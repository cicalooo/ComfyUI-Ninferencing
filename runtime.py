from dataclasses import dataclass, field
import json
import logging
from math import isqrt
import os
from pathlib import Path
import re

from .process import run_process
from .vram import capacity_error_message, is_capacity_failure, is_prompt_context_failure

LOGGER = logging.getLogger(__name__)


# Media budgets compiled into the ninfer 0.6.x qwen3_6 frontend, enforced in
# frontend_internal::enforce_budget. Vision attention cost is the sum of
# patches_per_item ** 2 over every item, and one patch covers 16 image pixels.
VISION_ITEM_LIMIT = 16
VISION_PATCH_PIXELS = 16
VISION_PAIR_BUDGET = 134_217_728
VISION_PROMPT_BUDGET = 32_768
VISION_TOKEN_CHARS = 3  # conservative prompt characters per token when sizing images
# Native video + --vision on 24 GB / 27B: auto's fixed 1024 MiB KV headroom does not
# fit beside the vision engine floor (~3.33 GiB after weights). 8192 INT8 tokens leave
# ~80 MiB spare against the measured shortfall on an idle RTX 3090.
VIDEO_SAFE_CONTEXT = 8192
# --vision-residency overlay borrows device memory per encode window from
# evict-ranked weights (output head, embedding, proposal/MTP/drafter). Newer
# builds (e.g. 8319e8f) reject startup when one window needs more than those
# weights provide, e.g. 12288 needs ~816 MiB but only ~656 MiB is available.
# 8192 needs ~544 MiB and fits that budget while still covering the node's
# auto-sized stills (largest single still is ~2809 vision tokens).
VISION_MAX_MERGED = 8192
VISION_MAX_MERGED_LADDER = (8192, 4096, 2048, 1024)
# Still-image multimodal (identity + sparse frames) on 24 GB / 27B: auto's 1024 MiB
# KV headroom can fail once vision is loaded. Retry with an explicit mid context that
# still fits a long rewrite + vision tokens — do not slam max-context to VIDEO_SAFE
# (8K) or the prepared prompt can exceed Engine capacity.
IMAGE_SAFE_CONTEXT = 16384
IMAGE_MID_CONTEXT = 12288
# Step-down ladder for explicit KV capacity retries (largest-first below current).
CAPACITY_TIGHTEN_LADDER = (IMAGE_SAFE_CONTEXT, IMAGE_MID_CONTEXT, VIDEO_SAFE_CONTEXT)
# Step-up ladder when prepared prompt exceeds a too-tight --max-context.
PROMPT_EXPAND_LADDER = (IMAGE_MID_CONTEXT, IMAGE_SAFE_CONTEXT, 24576, VISION_PROMPT_BUDGET)
AUTO_KV_HEADROOM_BYTES = 1024 * 1024 * 1024
# Current ninfer CLI sampling bounds (see --help SAMPLING). Help text overrides
# these when a build advertises a different range. Values outside the range are
# clamped so Custom sampling cannot fail the run (the widget used to default
# top_k to 40, which this CLI rejects with "--top-k must be in [0,20]").
SAMPLING_LIMITS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0.0, 20.0),
    "min_p": (0.0, 1.0),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
}
# Qwen3.8-27B instruct / non-thinking (https://huggingface.co/Qwen/Qwen3.8-27B).
# Official repetition_penalty is 1.0 (a no-op). This CLI has frequency_penalty, left at 0.
QWEN38_NOTHINK_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
}


def default_sampling():
    """Sampler used when thinking is off. Matches Qwen3.8-27B non-thinking."""
    return {"sampling": "custom", **QWEN38_NOTHINK_SAMPLING}


_SAMPLING_RANGE = re.compile(
    r"(?m)^[ \t]*--(temperature|top-p|top-k|min-p|presence-penalty|frequency-penalty)"
    r"\b[^\n]*?(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)"
)


def vision_side_limit(frame_count, requested, context_length=VISION_PROMPT_BUDGET, text_tokens=0):
    """Longest image side that keeps a request inside the build's vision budgets."""
    if frame_count < 1:
        return requested
    side = isqrt(isqrt(VISION_PAIR_BUDGET // frame_count))
    side -= side % 2
    if (side + 1) ** 4 * frame_count > VISION_PAIR_BUDGET:
        side -= 2  # ninfer rounds each side to whole patches and may round up
    limit = side * VISION_PATCH_PIXELS
    room = min(context_length, VISION_PROMPT_BUDGET) - text_tokens - 3 * frame_count
    per_image = room // frame_count
    if per_image > 3:
        # A merged-token grid of m costs m * m tokens plus a newline row and control tokens per image.
        grid = (isqrt(4 * per_image - 11) - 1) // 2
        limit = min(limit, max(64, grid * 2 * VISION_PATCH_PIXELS))
    return min(requested, limit) if requested > 0 else limit


@dataclass(frozen=True)
class Settings:
    executable: str = ""
    device: int = -1
    seed: int = 0
    context_length: int = 32768
    kv_cache: str = "auto"
    # auto | match_context | integer token count as string (e.g. "8192")
    kv_capacity: str = "auto"
    cuda_graphs: str = "auto"
    thinking: str = "auto"
    speculation: dict = field(default_factory=lambda: {"speculation": "off"})
    sampling: dict = field(default_factory=default_sampling)
    image_max_side: int = 0
    timeout: float = 600
    extra_args: str = "[]"


def probe_runtime(executable, check_cancel):
    code, stdout, stderr = run_process(
        [str(executable), "--help"], cwd=executable.parent, timeout=15, check_cancel=check_cancel,
    )
    help_text = stdout + "\n" + stderr
    flags = set(re.findall(r"--[a-zA-Z][a-zA-Z0-9-]*", help_text))
    if not {"--messages", "--max-new"}.issubset(flags):
        detail = (
            f"Ninferencing: {executable} must be the ninfer CLI with --messages and --max-new support. "
            f"Its --help exited with code {code}."
        )
        if os.name != "nt" and "libcudart.so" in help_text:
            detail += " Install the CUDA runtime library required by this NInfer build; see the README."
        raise RuntimeError(detail + "\n" + help_text[-4000:])
    return help_text, flags


def runtime_supports_v3(help_text):
    return bool(re.search(r"--spec[^\n]*\bdflash2\b", help_text))


def check_artifact_runtime(artifact, help_text):
    from .models import artifact_container_version

    version = artifact_container_version(artifact)
    if version is None:
        return
    if version >= 3 and not runtime_supports_v3(help_text):
        raise RuntimeError(
            f"Ninferencing: {Path(artifact).name} is a NInfer v{version} artifact. "
            "This ninfer CLI only loads v1/v2. Install a v3 CLI "
            "(https://github.com/cicalooo/ninfer-linux for RTX 30/40) or point "
            "Settings at a ninfer-all build that loads v3 artifacts."
        )



def resolve_kv_capacity(settings, *, has_images=False, has_video=False):
    """Choose --kv-capacity value for this build.

    ``auto`` leaves a fixed 1024 MiB sizing headroom (ninfer 0.6.x). That headroom
    fits still-image vision on a 24 GB 3090 with the 27B artifact, but native
    video_url + --vision raises the engine floor enough that auto fails by ~0.55 GiB.
    For video we therefore default to an explicit token capacity matching context
    (official vision launcher style) instead of auto.
    """
    mode = str(getattr(settings, "kv_capacity", "auto") or "auto").strip().lower()
    context = max(1, int(settings.context_length))
    if mode in {"match_context", "match", "context"}:
        return str(context)
    if mode not in {"", "auto"}:
        if not mode.isdigit() or int(mode) < 1:
            raise ValueError(
                'Ninferencing: kv_capacity must be "auto", "match_context", or a positive token count.'
            )
        return str(int(mode))
    if has_video:
        return str(context)
    return "auto"


def _next_lower_capacity(current_tokens):
    """Largest ladder step strictly below current, else half (floor 1024)."""
    for step in CAPACITY_TIGHTEN_LADDER:
        if step < current_tokens:
            return step
    halved = max(1024, current_tokens // 2)
    return halved if halved < current_tokens else None


def _next_higher_context(current_tokens):
    """Smallest expand-ladder step strictly above current."""
    for step in PROMPT_EXPAND_LADDER:
        if step > current_tokens:
            return step
    return None


def is_vision_residency_failure(text: str | None) -> bool:
    """True when --vision-residency overlay cannot borrow one encode window.

    Newer builds fail startup with e.g. "--vision-residency overlay needs
    816 MiB of evict-ranked weights for one encode window, but ... provide
    656 MiB; lower --vision-max-merged, use --vision-residency resident, or
    drop a flag that shrinks those weights". Both the image and video nodes
    share build_command, so both need the same retry.

    Flag names alone are not enough: any usage error prints --help, and that
    text mentions both --vision-residency and --vision-max-merged. Matching
    those names misread "--top-k must be in [0,20]" as an overlay shortfall
    and burned the vision retry ladder.
    """
    if not text:
        return False
    lowered = text.lower()
    if "evict-ranked" in lowered:
        return True
    return "encode window" in lowered and (
        "vision-residency" in lowered or "vision-max-merged" in lowered
    )


def sampling_limits(help_text: str | None) -> dict[str, tuple[float, float]]:
    """Ranges for custom sampling flags. Help text wins; otherwise SAMPLING_LIMITS."""
    limits = dict(SAMPLING_LIMITS)
    if not help_text:
        return limits
    for match in _SAMPLING_RANGE.finditer(help_text):
        key = match.group(1).replace("-", "_")
        limits[key] = (float(match.group(2)), float(match.group(3)))
    return limits


def clamp_sampling_value(key: str, value, limits: dict[str, tuple[float, float]]):
    """Return value clamped to the build's range. In-range values are unchanged."""
    bound = limits.get(key)
    if bound is None or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    lo, hi = bound
    if lo <= number <= hi:
        return value
    clamped = min(max(number, lo), hi)
    if key == "top_k" or (isinstance(value, int) and not isinstance(value, bool)):
        clamped = int(round(clamped))
    LOGGER.warning(
        "Ninferencing: sampling --%s %s is outside %s..%s; using %s",
        key.replace("_", "-"),
        value,
        int(lo) if float(lo).is_integer() else lo,
        int(hi) if float(hi).is_integer() else hi,
        clamped,
    )
    return clamped


def _next_lower_merged(current_tokens):
    """Largest vision ladder step strictly below current.

    Stops at 1024 — below that the caller falls back to resident rather than
    crippling merged tokens further (a single auto-sized still is ~2809).
    """
    for step in VISION_MAX_MERGED_LADDER:
        if step < current_tokens:
            return step
    return None


def tighten_command_for_vision(command, diagnostics=None):
    """Build a retry argv when overlay cannot borrow one encode window.

    First lowers --vision-max-merged down the ladder (8192→4096→2048→1024),
    then falls back to --vision-residency resident. Returns None when no
    safer command can be derived.
    """
    if "--vision" not in command:
        return None
    new_cmd = list(command)
    if "--vision-max-merged" in new_cmd:
        idx = new_cmd.index("--vision-max-merged")
        try:
            current = int(new_cmd[idx + 1])
        except (ValueError, IndexError):
            return None
        lowered = _next_lower_merged(current)
        if lowered is not None:
            new_cmd[idx + 1] = str(lowered)
            LOGGER.warning(
                "Ninferencing: vision overlay shortfall; retrying with --vision-max-merged %s (was %s)",
                lowered,
                current,
            )
            return new_cmd
    if "--vision-residency" in new_cmd:
        idx = new_cmd.index("--vision-residency")
        try:
            current_residency = new_cmd[idx + 1]
        except IndexError:
            return None
        if current_residency.strip().lower() == "overlay":
            new_cmd[idx + 1] = "resident"
            LOGGER.warning(
                "Ninferencing: vision overlay shortfall; retrying with --vision-residency resident",
            )
            return new_cmd
    return None


def tighten_command_for_capacity(command, diagnostics=None):
    """Build a retry argv with a smaller explicit --kv-capacity.

    Used when auto headroom or an oversized explicit capacity fails after weights.
    Still-image multimodal prefers a mid explicit match (16K then 12K) before the
    native-video 8K floor — clamping max-context to 8K can make the prepared prompt
    exceed Engine capacity. Clamps --max-context down to the new capacity so the
    request stays coherent. Returns None when no safer command can be derived.
    """
    if "--kv-capacity" not in command:
        return None
    idx = command.index("--kv-capacity")
    current = command[idx + 1]
    ctx_idx = command.index("--max-context") if "--max-context" in command else None
    context = int(command[ctx_idx + 1]) if ctx_idx is not None else VIDEO_SAFE_CONTEXT
    has_vision = "--vision" in command

    if current == "auto":
        # Video path never leaves "auto" in argv (resolve_kv_capacity remaps it).
        # Image multimodal: first drop auto's 1024 MiB headroom by switching to an
        # explicit capacity at min(context, IMAGE_SAFE). Keep the widget's mid
        # budget (e.g. 16384) — do not shrink to 8K on this first step.
        if has_vision and context > VIDEO_SAFE_CONTEXT:
            new_cap = min(context, IMAGE_SAFE_CONTEXT)
        else:
            new_cap = min(context, VIDEO_SAFE_CONTEXT)
    else:
        try:
            cur = int(current)
        except ValueError:
            return None
        new_cap = _next_lower_capacity(cur)
        if new_cap is None:
            return None

    new_cmd = list(command)
    new_cmd[idx + 1] = str(new_cap)
    if ctx_idx is not None and int(new_cmd[ctx_idx + 1]) > new_cap:
        new_cmd[ctx_idx + 1] = str(new_cap)
    LOGGER.warning(
        "Ninferencing: capacity shortfall; retrying with --kv-capacity %s --max-context %s (was kv-capacity %s)",
        new_cmd[idx + 1],
        new_cmd[ctx_idx + 1] if ctx_idx is not None else context,
        current,
    )
    return new_cmd


def expand_command_for_prompt_capacity(command, diagnostics=None):
    """Raise --max-context/--kv-capacity when the prepared prompt does not fit.

    Triggered after a too-aggressive tighten (or an undersized widget) reports
    "prepared prompt exceeds Engine context capacity". Prefers the next mid
    rung (12K/16K) rather than stripping vision items.
    """
    if "--max-context" not in command:
        return None
    ctx_idx = command.index("--max-context")
    try:
        context = int(command[ctx_idx + 1])
    except ValueError:
        return None
    new_ctx = _next_higher_context(context)
    if new_ctx is None:
        return None

    new_cmd = list(command)
    new_cmd[ctx_idx + 1] = str(new_ctx)
    if "--kv-capacity" in new_cmd:
        kv_idx = new_cmd.index("--kv-capacity")
        kv = new_cmd[kv_idx + 1]
        if kv == "auto":
            new_cmd[kv_idx + 1] = str(new_ctx)
        else:
            try:
                if int(kv) < new_ctx:
                    new_cmd[kv_idx + 1] = str(new_ctx)
            except ValueError:
                new_cmd[kv_idx + 1] = str(new_ctx)
    LOGGER.warning(
        "Ninferencing: prepared prompt exceeds context; retrying with --kv-capacity %s --max-context %s (was max-context %s)",
        new_cmd[new_cmd.index("--kv-capacity") + 1] if "--kv-capacity" in new_cmd else "n/a",
        new_ctx,
        context,
    )
    return new_cmd


def build_command(executable, artifact, messages, settings, flags, help_text, *, max_tokens, seed, device, has_images, has_video=False, video_fps=2):
    extra = json.loads(settings.extra_args or "[]")
    if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
        raise ValueError('Ninferencing: extra_args must be a JSON array of strings, e.g. ["--prefill-chunk", "256"].')
    extra_flags = {arg.split("=", 1)[0] for arg in extra if arg.startswith("--")}
    reserved = extra_flags & {"--messages", "--prompt", "--max-new", "--seed", "--raw-output", "--print-token-ids", "--help"}
    if reserved or "--" in extra or (extra and not extra[0].startswith("--")):
        raise ValueError("Ninferencing: extra_args cannot replace the prompt, output, seed, or CLI invocation mode.")
    command = [str(executable), str(artifact)]

    def add(flag, value=None):
        if flag in extra_flags:
            return
        if flag not in flags:
            raise ValueError(f"Ninferencing: this build does not advertise {flag}. Change the corresponding setting or select another build.")
        command.append(flag)
        if value is not None:
            command.append(str(value))

    add("--messages", messages)
    add("--max-new", max_tokens)
    add("--seed", seed)
    if "--device" in flags or device != 0:
        add("--device", device)
    add("--max-context", settings.context_length)
    # Text/still: auto sizes KV to remaining VRAM (1024 MiB headroom). Native video: explicit
    # capacity matching context — auto's headroom does not fit beside vision engine on 24 GB.
    if "--kv-capacity" in flags:
        add("--kv-capacity", resolve_kv_capacity(settings, has_images=has_images, has_video=has_video))
    if "--prefill-chunk" in flags:
        add("--prefill-chunk", 512)

    if settings.kv_cache != "auto":
        add("--kv-dtype", settings.kv_cache)
    elif re.search(r"--kv-dtype[^\n\]]*\bint8\b", help_text):
        add("--kv-dtype", "int8")
    if "--gdn-state-fp16" in flags:
        add("--gdn-state-fp16")
    if settings.cuda_graphs == "off" or (settings.cuda_graphs == "auto" and "--no-cuda-graph" in flags):
        add("--no-cuda-graph")
    if settings.thinking == "off" or (settings.thinking == "auto" and "--no-thinking" in flags):
        add("--no-thinking")
    elif settings.thinking not in {"auto", "model default"}:
        add("--reasoning-effort", settings.thinking)
    spec = settings.speculation
    if spec["speculation"] != "off":
        add("--spec", spec["speculation"])
        add("--draft-tokens", spec["draft_tokens"])
        if spec["lm_head_draft"]:
            add("--lm-head-draft")
    if settings.sampling.get("sampling") == "custom":
        limits = sampling_limits(help_text)
        for key, value in settings.sampling.items():
            if key == "sampling" or value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if flag not in flags:
                LOGGER.warning("Ninferencing: this build does not advertise %s; leaving that sampler control at the artifact default.", flag)
                continue
            add(flag, clamp_sampling_value(key, value, limits))
    if has_images or has_video:
        add("--vision")
        if "--vision-residency" in flags:
            add("--vision-residency", "overlay")
        if "--vision-max-merged" in flags:
            add("--vision-max-merged", VISION_MAX_MERGED)
        try:
            fps = int(video_fps)
        except (TypeError, ValueError):
            fps = 2
        if has_video and fps != 2:
            if "--video-fps" not in flags:
                raise ValueError(
                    "Ninferencing: video_fps is not 2, but this ninfer build does not advertise "
                    "--video-fps. Install the cicalooo/ninfer-linux build with that flag, or leave video_fps at 2."
                )
            add("--video-fps", fps)
    command.extend(extra)
    return command


def generate(command, timeout, check_cancel, temp_dir, *, on_capacity=None, vram_report=None, tighten_on_capacity=True):
    """Run ninfer. On capacity failure: reclaim VRAM, then optionally retry with tighter KV.

    Still-image multimodal may need a mid-context tighten (16K/12K) before the 8K
    video floor. If a tighten leaves max-context below the prepared prompt, retry
    once more by expanding context/KV upward instead of stripping vision.
    Vision overlay shortfall (--vision-residency overlay needs N MiB of
    evict-ranked weights) retries with a smaller --vision-max-merged, then
    resident. Image and video nodes share build_command, so both paths retry.
    """
    last_report = vram_report
    active = list(command)
    has_vision = "--vision" in active
    # Vision requests may need overlay retries plus KV tighten/expand steps.
    # Text requests keep the original budgets so persistent-failure call counts
    # are unchanged. Vision ladder from 8192 is 4096→2048→1024→resident (4
    # retries); from a stale 12288 it needs one more, so allow 6 attempts.
    # With tighten: original + mid-step + second tighten or prompt-expand.
    # Reclaim-only (no tighten): original + one same-command retry.
    if tighten_on_capacity:
        attempts = 6 if has_vision else 3
    elif on_capacity is not None:
        attempts = 4 if has_vision else 2
    else:
        attempts = 3 if has_vision else 1
    diagnostics = ""
    for attempt in range(attempts):
        code, text, diagnostics = run_process(
            active, cwd=Path(active[0]).parent, timeout=timeout, check_cancel=check_cancel, temp_dir=temp_dir,
        )
        if code == 0:
            text = text.strip()
            if not text:
                raise RuntimeError(
                    "Ninferencing: ninfer returned no answer. Increase max_tokens or turn thinking off.\n" + diagnostics
                )
            return text
        if attempt + 1 < attempts:
            if is_vision_residency_failure(diagnostics):
                shrunk = tighten_command_for_vision(active, diagnostics)
                if shrunk is not None and shrunk != active:
                    active = shrunk
                    has_vision = "--vision" in active
                    continue
            if is_capacity_failure(diagnostics):
                tightened = tighten_command_for_capacity(active, diagnostics) if tighten_on_capacity else None
                if tightened is not None and tightened != active:
                    if on_capacity is not None:
                        last_report = on_capacity() or last_report
                    active = tightened
                    continue
                if on_capacity is not None:
                    last_report = on_capacity() or last_report
                    continue
            elif tighten_on_capacity and is_prompt_context_failure(diagnostics):
                expanded = expand_command_for_prompt_capacity(active, diagnostics)
                if expanded is not None and expanded != active:
                    if on_capacity is not None:
                        last_report = on_capacity() or last_report
                    active = expanded
                    continue
        if is_capacity_failure(diagnostics):
            raise RuntimeError(capacity_error_message(diagnostics, last_report))
        if is_prompt_context_failure(diagnostics):
            raise RuntimeError(
                "Ninferencing: prepared prompt exceeds Engine context capacity. "
                "Raise the image node's context_length (prefer 12288–16384 for multimodal "
                "stills on 24 GB) and set Settings kv_capacity=match_context (or 16384). "
                "Do not rely on auto→8K tighten for identity+frame grids.\n" + diagnostics
            )
        if is_vision_residency_failure(diagnostics):
            raise RuntimeError(
                "Ninferencing: --vision-residency overlay needs more evict-ranked weights "
                "than this artifact loads for one encode window. The node already retries "
                "with a smaller --vision-max-merged, then resident. If it still fails, pass "
                'Settings extra_args ["--vision-residency", "resident"] or '
                '["--vision-max-merged", "4096"] (or lower), or drop a flag that shrinks '
                "those weights (--embedding-q4/q6, --lm-head-q4/q6).\n" + diagnostics
            )
        if re.search(
            r"(media item count|vision raw patches|vision tokens|vision attention pairs|prompt tokens)"
            r" exceeds? processor budget",
            diagnostics,
        ):
            raise RuntimeError(
                f"Ninferencing: ninfer's media preprocessor budget is exhausted. This build takes at most "
                f"{VISION_ITEM_LIMIT} images or frames per request (native video_url counts separately as video media), "
                "and image size is capped automatically to fit. Send fewer images/frames, or connect VIDEO "
                "on Ninferencing instead of a large frame grid."
            )
        unsupported = re.search(r"artifact identity '([^']+)' has no registered target for this device", diagnostics)
        if unsupported:
            device = active[active.index("--device") + 1] if "--device" in active else 0
            raise RuntimeError(
                f"Ninferencing: artifact target '{unsupported.group(1)}' is not supported by this ninfer build "
                f"on device {device}. Install a ninfer build that registers this target for your GPU, or select "
                "a compatible .ninfer artifact. Sampling, context, and VRAM settings cannot fix a missing native target."
            )
        if re.search(r"artifact magic is not NInfer v1 or v2", diagnostics):
            raise RuntimeError(
                "Ninferencing: this .ninfer file is a v3 container (or not a NInfer artifact). "
                "The selected ninfer CLI only loads v1/v2. Install a v3 CLI "
                "(https://github.com/cicalooo/ninfer-linux for RTX 30/40) or point "
                "Settings at a ninfer-all build.\n" + diagnostics
            )
        raise RuntimeError(f"Ninferencing: ninfer exited with code {code}.\n{diagnostics}")
    if is_vision_residency_failure(diagnostics):
        raise RuntimeError(
            "Ninferencing: --vision-residency overlay needs more evict-ranked weights "
            "than this artifact loads for one encode window after retries. Pass Settings "
            'extra_args ["--vision-residency", "resident"] or ["--vision-max-merged", "4096"] '
            "(or lower), or drop --embedding-q4/q6 / --lm-head-q4/q6.\n" + diagnostics
        )
    if is_prompt_context_failure(diagnostics):
        raise RuntimeError(
            "Ninferencing: prepared prompt exceeds Engine context capacity after retries. "
            "Raise context_length / kv_capacity (12–16K match_context for multimodal stills).\n" + diagnostics
        )
    raise RuntimeError(capacity_error_message(diagnostics, last_report))
