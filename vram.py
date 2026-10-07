"""Free enough GPU memory for a native ninfer load after ComfyUI has been using the card."""

from __future__ import annotations

import gc
import logging
import os
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


def unload_comfy(device: int | None = None) -> None:
    model_management.throw_exception_if_processing_interrupted()
    model_management.synchronize()
    # unload_all_models walks every torch device; free_memory again covers the ninfer index
    model_management.unload_all_models()
    try:
        import torch
        if torch.cuda.is_available() and device is not None:
            model_management.free_memory(1e30, torch.device("cuda", int(device)))
    except Exception:
        pass
    cleanup = getattr(model_management, "cleanup_models", None)
    if callable(cleanup):
        cleanup()
    cleanup_gc = getattr(model_management, "cleanup_models_gc", None)
    if callable(cleanup_gc):
        cleanup_gc()
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


def estimate_need_bytes(artifact_path, has_images: bool = False, has_video: bool = False) -> int:
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
