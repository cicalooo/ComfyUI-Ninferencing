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
    sampling: dict = field(default_factory=lambda: {"sampling": "model defaults"})
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


def build_command(executable, artifact, messages, settings, flags, help_text, *, max_tokens, seed, device, has_images, has_video=False):
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
        for key, value in settings.sampling.items():
            if key != "sampling" and value is not None:
                add("--" + key.replace("_", "-"), value)
    if has_images or has_video:
        add("--vision")
        if "--vision-residency" in flags:
            add("--vision-residency", "overlay")
        if "--vision-max-merged" in flags:
            add("--vision-max-merged", 12288)
    command.extend(extra)
    return command


def generate(command, timeout, check_cancel, temp_dir, *, on_capacity=None, vram_report=None, tighten_on_capacity=True):
    """Run ninfer. On capacity failure: reclaim VRAM, then optionally retry with tighter KV.

    Still-image multimodal may need a mid-context tighten (16K/12K) before the 8K
    video floor. If a tighten leaves max-context below the prepared prompt, retry
    once more by expanding context/KV upward instead of stripping vision.
    """
    last_report = vram_report
    active = list(command)
    # With tighten: original + mid-step + second tighten or prompt-expand.
    # Reclaim-only (no tighten): original + one same-command retry.
    if tighten_on_capacity:
        attempts = 3
    elif on_capacity is not None:
        attempts = 2
    else:
        attempts = 1
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
        if re.search(
            r"(media item count|vision raw patches|vision tokens|vision attention pairs|prompt tokens)"
            r" exceeds? processor budget",
            diagnostics,
        ):
            raise RuntimeError(
                f"Ninferencing: ninfer's media preprocessor budget is exhausted. This build takes at most "
                f"{VISION_ITEM_LIMIT} images or frames per request (native video_url counts separately as video media), "
                "and image size is capped automatically to fit. Send fewer images/frames, or prefer Ninferencing Video "
                "with a single native video instead of a large frame grid."
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
    if is_prompt_context_failure(diagnostics):
        raise RuntimeError(
            "Ninferencing: prepared prompt exceeds Engine context capacity after retries. "
            "Raise context_length / kv_capacity (12–16K match_context for multimodal stills).\n" + diagnostics
        )
    raise RuntimeError(capacity_error_message(diagnostics, last_report))
