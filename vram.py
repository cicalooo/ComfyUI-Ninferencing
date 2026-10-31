"""Free enough GPU memory for a native ninfer load after ComfyUI has been using the card."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import struct
import time

from comfy import model_management


LOGGER = logging.getLogger(__name__)
_NINFER_NAMES = frozenset({"ninfer", "ninfer.exe", "ninfer-serve", "ninfer-serve.exe"})


def free_bytes(device: int = 0) -> int | None:
    """Global free VRAM on the physical GPU index ninfer will use."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return int(free)
        except Exception:
            pass
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return None


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def orphan_ninfer_processes(exclude_pids: set[int] | None = None) -> list:
    """Return ninfer processes whose parent is gone (leftovers from crashed node runs)."""
    try:
        import psutil
    except ImportError:
        return []
    exclude = exclude_pids or set()
    found = []
    for process in psutil.process_iter(["pid", "name", "ppid"]):
        try:
            info = process.info
            name = (info.get("name") or "").lower()
            if name not in _NINFER_NAMES:
                continue
            pid = int(info["pid"])
            if pid in exclude or pid == os.getpid():
                continue
            parent = info.get("ppid")
            if parent and psutil.pid_exists(int(parent)):
                continue
            found.append(process)
        except (psutil.Error, TypeError, ValueError):
            continue
    return found


def reap_orphan_ninfer(exclude_pids: set[int] | None = None) -> list[int]:
    killed = []
    for process in orphan_ninfer_processes(exclude_pids):
        try:
            pid = process.pid
            process.kill()
            process.wait(timeout=10)
            killed.append(pid)
            LOGGER.warning("Ninferencing: stopped orphaned ninfer process pid=%s (parent already exited)", pid)
        except Exception as exc:
            LOGGER.warning("Ninferencing: could not stop orphaned ninfer pid=%s: %s", getattr(process, "pid", "?"), exc)
    return killed


def live_ninfer_pids(exclude_pids: set[int] | None = None) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    exclude = exclude_pids or set()
    pids = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            name = (process.info.get("name") or "").lower()
            if name not in _NINFER_NAMES:
                continue
            pid = int(process.info["pid"])
            if pid not in exclude and pid != os.getpid():
                pids.append(pid)
        except (psutil.Error, TypeError, ValueError):
            continue
    return pids


def _dead_real_model():
    """Stand-in so a later Comfy cleanup never calls a None real_model."""
    return None


def _detach_finalizer(loaded, name: str) -> None:
    finalizer = getattr(loaded, name, None)
    if finalizer is None:
        return
    detach = getattr(finalizer, "detach", None)
    if callable(detach):
        try:
            detach()
        except Exception:
            LOGGER.debug("Ninferencing: could not detach %s", name, exc_info=True)
    try:
        setattr(loaded, name, None)
    except Exception:
        LOGGER.debug("Ninferencing: could not clear %s", name, exc_info=True)


def release_comfy_models() -> int:
    """Unload Comfy's model registry without free_memory's index walk.

    free_memory saves indexes, then a weakref finalizer can pop the list
    (IndexError). A partial unload also sets real_model to None, and the
    prompt worker's later cleanup calls it (TypeError) and kills the thread.
    Clearing RAM does not repair that list, so the next run dies the same way.
    """
    registry = getattr(model_management, "current_loaded_models", None)
    if not isinstance(registry, list):
        return 0
    released = 0
    for _ in range(8):
        loaded = list(registry)
        if not loaded:
            break
        for entry in loaded:
            # Detach before any patcher.detach(). That call can collect a module
            # and run cleanup_models, which both shrinks the list and crashes if
            # real_model was already cleared.
            _detach_finalizer(entry, "model_finalizer")
            _detach_finalizer(entry, "_patcher_finalizer")
            try:
                entry.real_model = _dead_real_model
            except Exception:
                LOGGER.debug("Ninferencing: could not replace real_model", exc_info=True)
        for entry in loaded:
            try:
                model = entry.model
            except Exception:
                LOGGER.debug("Ninferencing: could not read loaded model", exc_info=True)
                continue
            if model is None:
                continue
            detach = getattr(model, "detach", None)
            if not callable(detach):
                continue
            try:
                detach(True)
            except Exception:
                LOGGER.warning(
                    "Ninferencing: failed to detach %s",
                    type(model).__name__,
                    exc_info=True,
                )
        gone = {id(entry) for entry in loaded}
        registry[:] = [entry for entry in registry if id(entry) not in gone]
        released += len(loaded)
    if registry:
        LOGGER.warning(
            "Ninferencing: %s Comfy model(s) still registered after unload",
            len(registry),
        )
    return released


def unload_comfy(device: int | None = None) -> None:
    """Drop Comfy-tracked weights before ninfer.

    device is unused: the registry covers every card. Do not call
    unload_all_models or free_memory; both index current_loaded_models and
    crash once a long session has left dead entries in it.
    """
    del device
    model_management.throw_exception_if_processing_interrupted()
    model_management.synchronize()
    released = release_comfy_models()
    if released:
        LOGGER.info("Ninferencing: released %s Comfy model(s)", released)
    for _ in range(2):
        gc.collect()
        model_management.soft_empty_cache(force=True)
    model_management.throw_exception_if_processing_interrupted()


def wait_for_free_vram(
    device: int,
    *,
    need_bytes: int | None,
    timeout_s: float,
    check_cancel,
    settle_s: float = 0.35,
) -> int | None:
    """Poll until free VRAM stops rising, or until need_bytes is met."""
    deadline = time.monotonic() + timeout_s
    best = free_bytes(device)
    stable_since = time.monotonic()
    while True:
        check_cancel()
        current = free_bytes(device)
        if current is not None and (best is None or current > best + 16 * 1024 * 1024):
            best = current
            stable_since = time.monotonic()
        if need_bytes is not None and current is not None and current >= need_bytes:
            return current
        if time.monotonic() - stable_since >= settle_s:
            return current
        if time.monotonic() >= deadline:
            return current
        time.sleep(0.1)


def prepare_for_ninfer(
    device: int,
    *,
    need_bytes: int | None,
    check_cancel,
    wait_s: float = 20.0,
) -> dict:
    """Unload ComfyUI-owned GPU state, reap orphan ninfer, wait for free VRAM."""
    killed = reap_orphan_ninfer()
    before = free_bytes(device)
    unload_comfy(device)
    if killed:
        # Driver often needs a beat after native process death before free counters move.
        time.sleep(0.5)
        unload_comfy(device)
    after = wait_for_free_vram(device, need_bytes=need_bytes, timeout_s=wait_s, check_cancel=check_cancel)
    if need_bytes is not None and (after is None or after < need_bytes):
        # Second pass: residual Python refs / delayed driver free.
        unload_comfy(device)
        after = wait_for_free_vram(device, need_bytes=need_bytes, timeout_s=min(10.0, wait_s), check_cancel=check_cancel)
    report = {
        "device": device,
        "before_free": before,
        "after_free": after,
        "need_bytes": need_bytes,
        "killed_orphans": killed,
        "live_ninfer": live_ninfer_pids(),
    }
    LOGGER.info(
        "Ninferencing VRAM handoff device=%s free %s → %s (need %s); orphans_killed=%s live_ninfer=%s",
        device,
        format_bytes(before),
        format_bytes(after),
        format_bytes(need_bytes),
        killed,
        report["live_ninfer"],
    )
    return report


def _legacy_need_bytes(artifact_path, has_images: bool = False, has_video: bool = False) -> int:
    """Weights on disk plus engine/KV headroom. Tuned so idle 24 GB cards pass the gate.

    Native video + --vision on the 27B artifact measures ~3.33 GiB engine floor after
    weights; an 8K INT8 KV is ~0.4 GiB. Overstating need (old +4 GiB) triggered useless
    unload loops when free was already ~20.25 GiB vs a 20.96 GiB estimate.
    """
    try:
        weights = int(artifact_path.stat().st_size)
    except OSError:
        weights = 0
    if has_video:
        # Engine floor ~3.33 GiB + modest KV (~0.4 GiB) + small fragment margin.
        headroom = int(3.8 * 1024 ** 3)
    elif has_images:
        # Still-image vision: engine + vision weights + auto KV headroom budget.
        headroom = 3 * 1024 ** 3 + 512 * 1024 ** 2
    else:
        headroom = 3 * 1024 ** 3
    return weights + headroom


# --- Qwen3.6-35B-A3B (family "qwen_moe") engine memory model -------------------------------
# Terms from ninfer-all 7a89828 docs/config-calculator.html (cross-checked there against the
# engine's own refusal arithmetic) plus per-component artifact bytes read with
# `ninfer inspect --objects` on the llmfan46 heretic artifact. Without --spec the engine loads
# no MTP/DFlash/proposal weights (docs/cli.md "Startup memory"); overlay Vision keeps the tower
# in pinned host memory and "costs no resident capacity". The file size is therefore NOT the
# device need: it includes DFlash (391 MiB), MTP (856 MiB), the proposal head (137 MiB) and the
# vision tower (268 MiB) that a spec-off overlay run never puts on the GPU.
_MIB = 1024 ** 2
_GIB = 1024 ** 3
QWEN_MOE_TEXT_WEIGHTS = 21_038_469_632        # resident text weights, spec none
QWEN_MOE_EXPERT_BYTES = int(17696.0 * _MIB)   # routed experts (movable with --expert-residency)
QWEN_MOE_EMBEDDING_BYTES = int(515.3 * _MIB)  # token embedding (movable with --embedding-residency)
QWEN_MOE_LAYER_BANK_BYTES = QWEN_MOE_EXPERT_BYTES // 40  # one layer's experts (host/disk banks)
QWEN_MOE_MTP_BYTES = int(856.3 * _MIB)
QWEN_MOE_MTP_Q4_SAVING = 350 * _MIB           # quality-trade-experiments.md: about -350 MiB
QWEN_MOE_PROPOSAL_BYTES = int(136.5 * _MIB)
QWEN_MOE_DFLASH_BYTES = int(391.1 * _MIB)
QWEN_MOE_VISION_BYTES = int(267.7 * _MIB)
# Resident vision also plans device execution for one 16,384-token item. Not in the
# calculator; 1 GiB is a deliberate over-estimate (overlay is the supported 24 GB path).
QWEN_MOE_VISION_RESIDENT_WORKSPACE = 1 * _GIB
QWEN_MOE_WORKSPACE = 109_453_312
QWEN_MOE_SPEC_TERMS = {
    # graph allowance, fixed sequence block, per-token sequence bytes, KV ratio, extra KV/token
    "none": (12_582_912, 70_579_968, 0.0625, 1.0, 0),
    "mtp": (90_177_536, 76_086_528, 0.125, 1.10078125, 0),
    "dflash": (167_772_160, 193_136_896, 0.125, 1.0, 4096),
}
QWEN_MOE_KV_BYTES = {
    "bf16": 20480, "int8": 10560, "fp8": 10320, "rk8v4": 8160,
    "rk4v4": 5600, "k8v4": 8040, "nvfp4": 5760,
}
QWEN_MOE_PAGE_TOKENS = 64
# ninfer's own CUDA context (calculator "Headless Linux" reserve, 307 MiB) plus a little slack.
QWEN_MOE_CUDA_CONTEXT = 320 * _MIB
QWEN_MOE_AUTO_HEADROOM = 1024 * _MIB

# The constants above are the llmfan46 heretic sm86 artifact and are only a fallback. Any other
# quant mix (e.g. Q8 expert layers) has different component sizes, so read them from the
# artifact's own v3 directory: entry header "<8sQ16s" (magic, JSON bytes, artifact id) then the
# directory JSON (tools/artifact/framing.py + reader.py in ninfer-all). Each tensor object's
# "bytes" is attributed to the component of the parameters bound to it.
_V3_MAGIC = b"NINFER\x00\x03"
_V3_HEADER = struct.Struct("<8sQ16s")
_V3_MAX_DIRECTORY_BYTES = 256 * _MIB
_EXPERT_PARAM_RE = re.compile(r"^text/layers/(\d+)/moe/experts/")
_QWEN_MOE_FALLBACK_BYTES = {
    "text": QWEN_MOE_TEXT_WEIGHTS, "experts": QWEN_MOE_EXPERT_BYTES,
    "embedding": QWEN_MOE_EMBEDDING_BYTES, "layer_bank": QWEN_MOE_LAYER_BANK_BYTES,
    "mtp": QWEN_MOE_MTP_BYTES, "proposal": QWEN_MOE_PROPOSAL_BYTES,
    "dflash": QWEN_MOE_DFLASH_BYTES, "vision": QWEN_MOE_VISION_BYTES, "source": "constants",
}
_COMPONENT_BYTES_CACHE: dict = {}


def _read_v3_component_bytes(path: str) -> dict:
    with open(path, "rb") as handle:
        raw = handle.read(_V3_HEADER.size)
        if len(raw) != _V3_HEADER.size:
            raise ValueError("truncated v3 header")
        magic, json_bytes, _artifact_id = _V3_HEADER.unpack(raw)
        if magic != _V3_MAGIC:
            raise ValueError("not a NInfer v3 artifact")
        if not 0 < json_bytes <= _V3_MAX_DIRECTORY_BYTES:
            raise ValueError(f"implausible directory size {json_bytes}")
        blob = handle.read(json_bytes)
    if len(blob) != json_bytes:
        raise ValueError("truncated v3 directory")
    directory = json.loads(blob)
    owner: dict[str, str] = {}
    for parameter, binding in directory["bindings"].items():
        if "object" in binding:
            ids = [binding["object"]]
        else:
            ids = [part["object"] for part in binding.get("parts", ())]
        for object_id in ids:
            # Lowest parameter name wins so the attribution is deterministic.
            if object_id not in owner or parameter < owner[object_id]:
                owner[object_id] = parameter
    sums = {"text": 0, "experts": 0, "embedding": 0, "mtp": 0, "proposal": 0, "dflash": 0,
            "vision": 0}
    layers: dict[int, int] = {}
    for obj in directory["objects"]:
        if obj.get("kind") != "tensor":
            continue
        size = int(obj["bytes"])
        parameter = owner.get(obj["id"])
        if parameter is None:
            continue  # unbound payload is never loaded
        component = parameter.split("/", 1)[0]
        expert = _EXPERT_PARAM_RE.match(parameter)
        if component == "text":
            sums["text"] += size  # text total includes experts and the embedding
            if expert:
                sums["experts"] += size
                layer = int(expert.group(1))
                layers[layer] = layers.get(layer, 0) + size
            elif parameter == "text/token_embedding":
                sums["embedding"] += size
        elif component in sums:
            sums[component] += size
    if sums["text"] <= 0 or sums["experts"] <= 0 or not layers:
        raise ValueError("no text/expert tensors bound")
    sums["layer_bank"] = max(layers.values())
    sums["source"] = "artifact"
    return sums


def qwen_moe_component_bytes(path) -> dict:
    """Per-component device bytes of a qwen_moe artifact, cached per (path, mtime, size).

    Falls back to the llmfan46 sm86 constants when the file is missing or unparsable.
    """
    if not path:
        return dict(_QWEN_MOE_FALLBACK_BYTES)
    path = str(path)
    try:
        st = os.stat(path)
    except OSError:
        return dict(_QWEN_MOE_FALLBACK_BYTES)
    key = (os.path.realpath(path), st.st_mtime_ns, st.st_size)
    cached = _COMPONENT_BYTES_CACHE.get(key)
    if cached is None:
        try:
            cached = _read_v3_component_bytes(path)
        except Exception as exc:  # noqa: BLE001 - any parse problem means "use the constants"
            LOGGER.warning("ninfer: could not read component sizes from %s (%s); using "
                           "built-in Qwen3.6-35B-A3B sizes", path, exc)
            cached = dict(_QWEN_MOE_FALLBACK_BYTES)
        _COMPONENT_BYTES_CACHE[key] = cached
    return dict(cached)


def _command_artifact(cmd):
    for arg in cmd[1:]:
        if arg.lower().endswith(".ninfer"):
            return arg
    return cmd[1] if len(cmd) > 1 and not cmd[1].startswith("-") else None


def _argv_value(command, flag, default=None):
    if flag in command:
        idx = command.index(flag)
        if idx + 1 < len(command):
            return command[idx + 1]
    for arg in command:
        if isinstance(arg, str) and arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _argv_has(command, flag):
    return flag in command or any(isinstance(a, str) and a.startswith(flag + "=") for a in command)


def qwen_moe_memory_terms(command) -> dict:
    """Device bytes ninfer reserves for this argv on the Qwen3.6-35B-A3B artifact.

    Returns a breakdown with ``fixed`` (weights + workspace + graph + fixed sequence block +
    CUDA context), ``per_token`` (KV + per-token sequence bytes, page-rounded by the caller),
    ``kv_tokens`` (the requested capacity, or None for auto) and ``total``.
    """
    cmd = [str(a) for a in command]
    spec = (_argv_value(cmd, "--spec", "none") or "none").lower()
    spec_key = "mtp" if spec == "mtp" else "dflash" if spec.startswith("dflash") or spec == "dspark" else "none"
    graph, seq_fixed, seq_pt, kv_ratio, kv_extra = QWEN_MOE_SPEC_TERMS[spec_key]
    if _argv_has(cmd, "--no-cuda-graph"):
        graph = 0

    sizes = qwen_moe_component_bytes(_command_artifact(cmd))
    weights = sizes["text"]
    if spec_key == "mtp":
        weights += sizes["mtp"]
        if _argv_has(cmd, "--mtp-experts-q4"):
            weights -= QWEN_MOE_MTP_Q4_SAVING
    elif spec_key == "dflash":
        weights += sizes["dflash"]
    if spec_key != "none" and not _argv_has(cmd, "--full-draft-head"):
        weights += sizes["proposal"]
    experts = (_argv_value(cmd, "--expert-residency", "device") or "device").lower()
    if experts in {"host", "disk"}:
        # Experts leave the device; two layer banks stay (one when memory is short). The
        # persistent expert cache (--expert-cache-mib auto) only takes what is left over.
        weights -= sizes["experts"]
        weights += 2 * sizes["layer_bank"]
    if (_argv_value(cmd, "--embedding-residency", "device") or "device").lower() == "host":
        weights -= sizes["embedding"]

    vision = 0
    if _argv_has(cmd, "--vision") or _argv_has(cmd, "--vision-cpu"):
        residency = (_argv_value(cmd, "--vision-residency", "resident") or "resident").lower()
        if _argv_has(cmd, "--vision-cpu"):
            residency = "cpu"
        if residency == "resident":
            vision = sizes["vision"] + QWEN_MOE_VISION_RESIDENT_WORKSPACE
        # overlay: window borrowed from free KV pages / evict-ranked weights; cpu: none.

    kv_dtype = (_argv_value(cmd, "--kv-dtype", "bf16") or "bf16").lower()
    kv_pt = QWEN_MOE_KV_BYTES.get(kv_dtype, QWEN_MOE_KV_BYTES["bf16"]) * kv_ratio + kv_extra
    try:
        context = int(_argv_value(cmd, "--max-context", "32768"))
    except ValueError:
        context = 32768
    cap_raw = (_argv_value(cmd, "--kv-capacity", str(context)) or str(context)).lower()
    kv_tokens = None if cap_raw == "auto" else int(cap_raw)

    fixed = (weights + QWEN_MOE_WORKSPACE + graph + seq_fixed + vision + QWEN_MOE_CUDA_CONTEXT)
    tokens = kv_tokens if kv_tokens is not None else context
    pages = -(-tokens // QWEN_MOE_PAGE_TOKENS)
    kv_bytes = int(kv_pt * pages * QWEN_MOE_PAGE_TOKENS + seq_pt * tokens)
    headroom = QWEN_MOE_AUTO_HEADROOM if kv_tokens is None else 0
    return {
        "spec": spec_key, "kv_dtype": kv_dtype, "weights": int(weights), "vision": vision,
        "sizes_source": sizes["source"],
        "runtime_fixed": int(QWEN_MOE_WORKSPACE + graph + seq_fixed),
        "fixed": int(fixed), "kv_per_token": float(kv_pt + seq_pt), "kv_tokens": kv_tokens,
        "kv_bytes": kv_bytes, "headroom": headroom,
        "total": int(fixed + kv_bytes + headroom),
    }


def estimate_need_bytes(artifact_path, has_images: bool = False, has_video: bool = False, *,
                        family: str | None = None, command=None) -> int:
    """Device bytes the ninfer run needs. See _legacy_need_bytes for non-MoE families.

    ``qwen_moe`` with an argv uses the engine terms above (spec-, KV- and residency-aware).
    """
    if family == "qwen_moe" and command:
        return qwen_moe_memory_terms(command)["total"]
    return _legacy_need_bytes(artifact_path, has_images, has_video)


_RESERVATION_RE = re.compile(
    r"requested Engine runtime reservation requires (\d+) bytes, but only (\d+) bytes are available"
)
_MINIMUM_RE = re.compile(
    r"minimum Engine runtime reservation requires (\d+) bytes in addition to (\d+) bytes of "
    r"automatic headroom, but only (\d+) bytes are available"
)
_AUTO_HEADROOM_RE = re.compile(
    r"automatic KV headroom requires (\d+) bytes, but only (\d+) bytes are available"
)


def parse_reservation_shortfall(text: str | None) -> dict | None:
    """Numbers from ninfer's KV-capacity refusal (src/runtime/engine/kv_capacity.cpp).

    ``available`` is what the engine had after weights for the runtime reservation.
    ``minimum`` is the smallest reservation it could make (only in the auto-headroom form).
    Returns None for raw allocation failures (cudaMalloc / out of memory), which carry no
    numbers and are not caused by the KV request.
    """
    if not text:
        return None
    m = _RESERVATION_RE.search(text)
    if m:
        return {"kind": "requested", "required": int(m.group(1)), "available": int(m.group(2))}
    m = _MINIMUM_RE.search(text)
    if m:
        return {"kind": "minimum", "required": int(m.group(1)), "headroom": int(m.group(2)),
                "available": int(m.group(3))}
    m = _AUTO_HEADROOM_RE.search(text)
    if m:
        return {"kind": "headroom", "required": int(m.group(1)), "available": int(m.group(2))}
    return None


def is_capacity_failure(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    # Phrase matches only. Do not use bare "oom" — ninfer --help mentions
    # "headroom", which contains that substring and misclassifies load errors
    # such as unknown tensor formats as capacity failures.
    return (
        any(
            token in lowered
            for token in (
                "runtime capacity",
                "runtime reservation",
                "out of memory",
                "cudamalloc",
                "cuda out of memory",
                # Driver-API OOM: overlay Vision backs the weight arena with cuMemCreate pieces
                # (EvictableWeightPool), so a weights shortfall reads
                # "cuMemCreate(&handle, bytes, &prop, 0) failed: CUDA_ERROR_OUT_OF_MEMORY".
                "cuda_error_out_of_memory",
            )
        )
        or (
            "model weights require" in lowered
            and "free before loading weights" in lowered
        )
    )


def is_prompt_context_failure(text: str | None) -> bool:
    """True when ninfer rejected the request because prompt+media > --max-context."""
    if not text:
        return False
    lowered = text.lower()
    return (
        "prepared prompt exceeds engine context capacity" in lowered
        or "prepared prompt exceeds" in lowered and "context capacity" in lowered
    )


def capacity_error_message(diagnostics: str, report: dict | None) -> str:
    parts = [
        "Ninferencing: ninfer could not allocate enough device memory for the model and engine.",
        "ComfyUI models were unloaded and orphaned ninfer processes were stopped when found.",
        "If this was native video on a 24 GB card, lower context_length (try 8192) or set "
        "Settings kv_capacity to match_context / 8192 — do not use kv-capacity auto with --vision video. "
        "If this was multimodal stills (identity + frames), prefer context_length 12288–16384 with "
        "kv_capacity=match_context rather than auto (auto may shortfall; a blind 8K retry can then "
        "fail with prepared prompt exceeds Engine context capacity).",
    ]
    if report:
        parts.append(
            f"VRAM free before handoff {format_bytes(report.get('before_free'))}, "
            f"after {format_bytes(report.get('after_free'))}, "
            f"estimated need {format_bytes(report.get('need_bytes'))}."
        )
        if report.get("killed_orphans"):
            parts.append(f"Stopped orphaned ninfer PIDs: {report['killed_orphans']}.")
        live = report.get("live_ninfer") or []
        if live:
            parts.append(
                f"Other live ninfer processes still hold the GPU: PIDs {live}. "
                "Stop them or use another device."
            )
    parts.append(diagnostics.strip())
    return "\n".join(parts)
