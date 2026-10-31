import json
import logging
from dataclasses import replace
from pathlib import Path
import tempfile
import threading

import numpy as np
from PIL import Image

import folder_paths
from comfy import model_management
from comfy_api.latest import io

from .models import default_executable, detect_artifact_family, resolve_executable, resolve_model
from .runtime import (
    QWEN38_NOTHINK_SAMPLING,
    VISION_ITEM_LIMIT,
    VISION_TOKEN_CHARS,
    Settings,
    build_command,
    check_artifact_runtime,
    estimate_still_tokens,
    fit_qwen_moe_residency,
    generate,
    probe_runtime,
    vision_side_limit,
)
from .videosample import (
    build_clip,
    fit_side_for_tokens,
    gemma_still_tokens,
    gemma_video_plan,
    ninfer_video_plan,
    probe_video,
)

LOGGER = logging.getLogger(__name__)
from .vram import estimate_need_bytes, prepare_for_ninfer


SettingsType = io.Custom("NINFERENCING_SETTINGS")
# CF Image Bin cable: ordered tuple of [1,H,W,C] IMAGE tensors. Match the string; do not import the other pack.
CF_REFS_TYPE = "CF_REFS"
_execution_lock = threading.Lock()


def first(value):
    return value[0] if isinstance(value, list) else value


def connected(value):
    """Unwrap an is_input_list optional. Missing or empty stays None."""
    value = first(value)
    if value is None or value == "":
        return None
    return value


def scalar_group(value):
    return {key: first(item) for key, item in value.items()}


def prepare_memory(device=0, need_bytes=None, check_cancel=None):
    return prepare_for_ninfer(
        int(device),
        need_bytes=need_bytes,
        check_cancel=check_cancel or model_management.throw_exception_if_processing_interrupted,
    )


def image_frames(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from image_frames(item)
    elif value is not None:
        yield from value


def _pending_file_trim(video):
    """True when Comfy trim metadata is not yet baked into the file bytes."""
    getter = getattr(video, "get_active_trim_window", None)
    if not callable(getter):
        return False
    try:
        start, duration = getter()
    except Exception:
        return False
    return float(start or 0.0) > 0.0 or float(duration or 0.0) > 0.0


def _on_disk_file(video):
    """Existing file path if the VIDEO already *is* that file (no encode).

    VideoInput.get_stream_source's default re-encodes into BytesIO — skip that.
    Also skip when the object still carries an unbaked start/duration window.
    """
    if _pending_file_trim(video):
        return None
    impl = getattr(type(video), "get_stream_source", None)
    try:
        from comfy_api.latest._input.video_types import VideoInput
        if impl is VideoInput.get_stream_source:
            return None
    except Exception:
        pass
    getter = getattr(video, "get_stream_source", None)
    if not callable(getter):
        return None
    try:
        source = getter()
    except Exception:
        return None
    if isinstance(source, str) and Path(source).is_file():
        return Path(source)
    return None


def _reuse_or_save_video(video, directory):
    """Prefer an existing on-disk stream; otherwise re-encode to directory."""
    from comfy_api.latest import VideoCodec, VideoContainer

    existing = _on_disk_file(video)
    if existing is not None:
        return existing
    path = Path(directory) / "input_video.mp4"
    model_management.throw_exception_if_processing_interrupted()
    video.save_to(str(path), format=VideoContainer.MP4, codec=VideoCodec.H264)
    return path


def spread_indices(count, budget):
    """Evenly cover ``count`` items with at most ``budget`` picks."""
    count = int(count)
    budget = max(1, int(budget))
    if count <= budget:
        return list(range(max(count, 0)))
    if budget == 1:
        return [count // 2]
    return [round(i * (count - 1) / (budget - 1)) for i in range(budget)]


def _as_ref_image(img):
    """One cable slot -> one or more [1,H,W,C] stills, batch order preserved."""
    if not hasattr(img, "ndim"):
        raise TypeError(
            f"Ninferencing refs item must be an IMAGE tensor, got {type(img).__name__}"
        )
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4 or int(img.shape[0]) < 1 or int(img.shape[-1]) < 3:
        raise ValueError(
            f"Ninferencing refs image must be [1,H,W,C] (or [H,W,C]); got shape {tuple(img.shape)}"
        )
    return tuple(img[index:index + 1].contiguous() for index in range(int(img.shape[0])))


def normalize_refs(refs):
    """CF Image Bin cable -> ordered tuple of [1,H,W,C] stills.

    Accepts None, one IMAGE, a tuple/list of IMAGEs, or the is_input_list wrap
    of that tuple. Bin order is preserved. Each still stays its own image.
    """
    if refs is None:
        return ()
    if hasattr(refs, "ndim"):
        return _as_ref_image(refs)
    if isinstance(refs, (list, tuple)):
        out = []
        for item in refs:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                for sub in item:
                    if sub is None:
                        continue
                    out.extend(_as_ref_image(sub))
            else:
                out.extend(_as_ref_image(item))
        return tuple(out)
    raise TypeError(
        f"Ninferencing refs expected a CF Image Bin cable (CF_REFS), got {type(refs).__name__}"
    )


def _as_batch(frame):
    if getattr(frame, "ndim", 0) == 3:
        return frame.unsqueeze(0)
    return frame


def split_ordered_frames(sources):
    """Split sources into pinned CF refs and everything else, in attachment order."""
    pinned = []
    rest = []
    for key, batch in (sources or {}).items():
        frames = list(image_frames(batch))
        if key == "refs":
            pinned.extend(frames)
        else:
            rest.extend(frames)
    return pinned, rest


def apply_still_budget(pinned, rest, budget):
    """Keep every ref. Only extra stills are spread into the leftover budget."""
    budget = max(0, int(budget))
    if len(pinned) > budget:
        raise ValueError(
            f"Ninferencing: {len(pinned)} CF Image Bin refs exceed the "
            f"{budget}-image vision budget. Refs stay in bin order and are not thinned; "
            "remove images from the bin."
        )
    room = budget - len(pinned)
    if len(rest) > room:
        chosen = [] if room == 0 else [rest[index] for index in spread_indices(len(rest), room)]
        LOGGER.info(
            "Ninferencing: %d extra images/frames exceed the remaining %d-item vision budget; sending %d spread across that input.",
            len(rest), room, len(chosen),
        )
        rest = chosen
    return pinned, rest


def limit_image_frames(images, budget):
    """Drop extra stills down to ``budget``. CF Image Bin refs are kept in order."""
    pinned, rest = split_ordered_frames(images)
    if len(pinned) + len(rest) <= budget:
        return images
    pinned, rest = apply_still_budget(pinned, rest, budget)
    limited = {}
    if pinned:
        limited["refs"] = [_as_batch(frame) for frame in pinned]
    if rest:
        limited["image_list"] = [_as_batch(frame) for frame in rest]
    return limited


# Full-clip video only: ninfer decodes the clip itself at its baked 2 fps with
# real timestamps; a native video_url is one media item.
VIDEO_FULL_DEFAULT_SIDE = 512


def fit_video_for_vision(video, directory, *, video_max_side=0,
                         token_budget=None, video_duration=0.0, family="qwen"):
    """Export the full clip as a video_url file. Returns (path, video_tokens).

    The loader owns all windowing: every input is treated as start-0 full
    input and the clip is sent whole — never trimmed or capped. ``video_max_side``
    downscales the longest edge (even dimensions) so large plates fit the
    vision token budget. ``video_duration`` (0 = probe the file) overrides the
    plate length used for planning and logging when container metadata is
    unreliable; it never cuts the clip.

    ``video_tokens`` is the planned native-video vision cost (ninfer samples
    the clip itself at its baked 2 fps), so stills can be sized against what
    remains of ``context_length``.
    """
    exported = export_video_path(video, directory)
    try:
        side = int(video_max_side or 0)
    except (TypeError, ValueError):
        side = 0
    if side < 0:
        side = 0
    info = probe_video(exported)
    if info is None:
        # Not a container we can inspect (or PyAV refused it): send as-is and
        # let the CLI report what it could not read.
        return exported, 0
    try:
        claimed = float(video_duration or 0.0)
    except (TypeError, ValueError):
        claimed = 0.0
    if claimed > 0 and abs(claimed - info.duration) / max(info.duration, 1e-6) > 0.05:
        LOGGER.info(
            "Ninferencing: video_duration %.2f s differs from probed %.2f s; "
            "reporting video_duration, planning from probed frames.",
            claimed, info.duration,
        )
    if info.frame_count <= 5 or info.duration < 1.0:
        LOGGER.info(
            "Ninferencing: image-as-video suspected (%d frames, %.3f s); sending as native video_url anyway.",
            info.frame_count, info.duration,
        )
    if side <= 0:
        side = VIDEO_FULL_DEFAULT_SIDE
    # Gemma soft-token cost is fixed per frame; Qwen cost scales with the grid, so
    # only the Qwen path shrinks max_side against token_budget.
    if token_budget and family != "gemma4":
        fitted = fit_side_for_tokens(
            info.width, info.height, info.frame_count, info.rate, side, token_budget,
        )
        if fitted < side and max(info.width, info.height) > fitted:
            LOGGER.info("Ninferencing: video_max_side %d -> %d to keep the clip within %d vision tokens.",
                        side, fitted, token_budget)
            side = fitted
    clip = build_clip(info, Path(directory) / "input_video_full.mp4", max_side=side)
    final = probe_video(clip) if clip is not None else info
    tokens = 0
    if final is not None:
        if family == "gemma4":
            frames, w, h, tokens, stamps = gemma_video_plan(final.frame_count, final.rate)
            LOGGER.info(
                "Ninferencing: full clip %.2f s @ %.3f fps, %dx%d -> Gemma samples %d frames evenly, "
                "~%d vision tokens, timestamps %.2f..%.2f s.",
                claimed if claimed > 0 else final.duration, float(final.rate), final.width, final.height, frames,
                tokens,
                stamps[0] if stamps else 0.0, stamps[-1] if stamps else 0.0,
            )
        else:
            frames, w, h, tokens, stamps = ninfer_video_plan(
                final.width, final.height, final.frame_count, final.rate,
            )
            LOGGER.info(
                "Ninferencing: full clip %.2f s @ %.3f fps, %dx%d -> ninfer samples %d frames at baked 2 fps, "
                "vision grid %dx%d, ~%d vision tokens, timestamps %.2f..%.2f s.",
                claimed if claimed > 0 else final.duration, float(final.rate), final.width, final.height, frames,
                w, h, tokens, stamps[0] if stamps else 0.0, stamps[-1] if stamps else 0.0,
            )
    return (clip if clip is not None else exported), tokens


def export_video_path(video, directory):
    """Materialize a local file path for ninfer video_url parts.

    The loader owns all windowing: every input is treated as start-0 full
    input, so this never trims. Prefer an existing on-disk stream; re-encode
    only when the object still carries unbaked trim metadata.
    """
    if video is None:
        raise ValueError("Ninferencing: connect a VIDEO input.")
    model_management.throw_exception_if_processing_interrupted()
    return _reuse_or_save_video(video, directory)


def write_messages(directory, prompt, system_prompt, images, max_side, context_length, video_paths=None,
                    video_token_reserve=0, family="qwen"):
    """Build messages.json for ninfer.

    Still images become image_url parts. Native video files become video_url
    parts (ninfer 0.6.x --vision). video_paths are written first so identity
    stills remain the later attachments when both are present.

    ``video_token_reserve`` is the planned native-video vision cost. It is
    spent from the prompt budget before stills are sized, so a clip plus
    stills cannot silently overflow ``context_length`` the way an unreserved
    video side could (the image-as-video 8192 failure).
    """
    pinned, rest = split_ordered_frames(images)
    videos = [Path(path) for path in (video_paths or []) if path]
    # A native video consumes one media item. CF Image Bin refs are pinned in
    # bin order and are never thinned; only extra stills are spread into what remains.
    frame_budget = max(0, VISION_ITEM_LIMIT - (1 if videos else 0))
    if videos or pinned:
        pinned, rest = apply_still_budget(pinned, rest, frame_budget)
    elif len(rest) > VISION_ITEM_LIMIT:
        LOGGER.info(
            "Ninferencing: %d images/frames exceed the %d-item vision budget; sending %d spread across the input.",
            len(rest), VISION_ITEM_LIMIT, VISION_ITEM_LIMIT,
        )
        rest = [rest[index] for index in spread_indices(len(rest), VISION_ITEM_LIMIT)]
    frames = pinned + rest
    if pinned:
        LOGGER.info(
            "Ninferencing: %d CF Image Bin refs first, in bin order, each saved as its own image.",
            len(pinned),
        )
    try:
        reserve = max(0, int(video_token_reserve or 0))
    except (TypeError, ValueError):
        reserve = 0
    text_tokens = (len(prompt) + len(system_prompt)) // VISION_TOKEN_CHARS + reserve
    # Native video is one media item for the preprocessor; still side caps only
    # apply to image/frame attachments sharing the request.
    if family == "gemma4":
        # Gemma soft-token budgets are fixed (default 280/image). Side caps still shrink
        # the carrier PNG; they do not change the soft-token ladder the way Qwen grids do.
        still_tokens = gemma_still_tokens(len(frames))
        if max_side <= 0:
            max_side = 1024  # practical carrier default; processor fits to soft-token budget
    else:
        max_side = vision_side_limit(len(frames), max_side, context_length, text_tokens)
        still_tokens = estimate_still_tokens(
            (max(int(frame.shape[-3]), int(frame.shape[-2])) for frame in frames), max_side,
        )
    if text_tokens + still_tokens > context_length:
        raise ValueError(
            "Ninferencing: planned prompt exceeds context_length "
            f"(text ~{text_tokens - reserve} + video ~{reserve} + stills ~{still_tokens} "
            f"= ~{text_tokens + still_tokens} > {context_length}). "
            "Raise context_length (video+stills: 12288–16384 on 24 GB, 64K for a 15 s plate with 4–6 stills), "
            "send fewer stills, or lower video_max_side/image_max_side."
        )
    content = []
    for path in videos:
        model_management.throw_exception_if_processing_interrupted()
        content.append({"type": "video_url", "video_url": {"url": str(path)}})
    for frame in frames:
        model_management.throw_exception_if_processing_interrupted()
        pixels = np.clip(frame.detach().cpu().numpy() * 255, 0, 255).round().astype(np.uint8)
        if pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        image = Image.fromarray(pixels).convert("RGB")
        if max_side > 0:
            image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        path = directory / f"image_{len(content)}.png"
        image.save(path, format="PNG")
        content.append({"type": "image_url", "image_url": {"url": str(path)}})
    has_media = bool(content)
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    if has_media:
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    path = directory / "messages.json"
    path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    return path, has_media


def collect_image_sources(images=None, image_list=None, refs=None):
    """Order: CF Image Bin refs first, then autogrow sockets, then image_list."""
    sources = {}
    normalized = normalize_refs(refs)
    if normalized:
        sources["refs"] = list(normalized)
    if images:
        sources.update(images)
    if image_list is not None:
        sources["image_list"] = image_list
    return sources


def refs_input():
    return io.Custom(CF_REFS_TYPE).Input(
        "refs",
        optional=True,
        tooltip=(
            "Ordered refs from CF Image Bin (CF_REFS). Bin order is kept and each "
            "still is saved as its own image, before image_list and images. "
            "Do not also wire the bin's image_list."
        ),
    )


def ninfer_model_input():
    names = [name for name in folder_paths.get_filename_list("ninfer") if name.lower().endswith(".ninfer")]
    return io.DynamicCombo.Input("model", options=[
        io.DynamicCombo.Option("library", [io.Combo.Input("name", options=names or ["No .ninfer models found"], display_name="model file")]),
        io.DynamicCombo.Option("file path", [io.String.Input("path", default="", display_name="model file", tooltip="Full path to any .ninfer artifact supported by your build.")]),
    ])


THINKING_OPTIONS = ("off", "auto", "model default", "low", "medium", "high", "xhigh")
REASONING_LOOP_OPTIONS = ("off", "stop", "conclude")
SPECULATION_MODES = ("mtp", "dflash", "dflash2")


def _override_thinking(settings, thinking=None, thinking_budget=None, reasoning_loop=None):
    """Apply Main-node thinking controls over Settings. Returns updated Settings."""
    node_thinking = first(thinking) if thinking is not None else None
    node_budget = first(thinking_budget) if thinking_budget is not None else None
    node_loop = first(reasoning_loop) if reasoning_loop is not None else None
    overrides = {}
    if node_thinking in THINKING_OPTIONS:
        overrides["thinking"] = node_thinking
    if node_budget is not None:
        try:
            overrides["thinking_budget"] = int(node_budget)
        except (TypeError, ValueError):
            pass
    if node_loop in REASONING_LOOP_OPTIONS:
        overrides["reasoning_loop"] = node_loop
    if overrides:
        settings = replace(settings, **overrides)
    return settings


def _override_speculation(settings, speculation=None):
    """Apply Main-node speculation over Settings. Returns updated Settings."""
    if speculation is None:
        return settings
    node_value = first(speculation) if isinstance(speculation, list) else speculation
    if isinstance(node_value, str):
        mode = node_value
        group = {"speculation": mode}
        if mode != "off":
            group.update({"draft_tokens": 3, "lm_head_draft": False})
    elif isinstance(node_value, dict):
        group = scalar_group(node_value)
        mode = group.get("speculation", "off")
        if isinstance(mode, list):
            mode = first(mode)
            group["speculation"] = mode
    else:
        return settings
    mode = group.get("speculation", "off")
    if mode in ("off", *SPECULATION_MODES):
        settings = replace(settings, speculation=dict(group))
    return settings


def run_ninfer(*, model, prompt, system_prompt, max_tokens, context_length, image_max_side,
               settings=None, images=None, image_list=None, refs=None,
               video=None, video_max_side=0,
               video_duration=0.0,
               thinking=None, thinking_budget=None, reasoning_loop=None,
               speculation=None):
    """CLI invocation. Optional video is a native video_url; stills stay image_url.

    Returns (text, thinking): the answer and the reasoning stream (stderr).
    """
    settings = replace(
        first(settings) or Settings(),
        context_length=first(context_length),
        image_max_side=first(image_max_side),
    )
    settings = _override_thinking(
        settings, thinking=thinking,
        thinking_budget=thinking_budget, reasoning_loop=reasoning_loop,
    )
    settings = _override_speculation(settings, speculation=speculation)
    artifact = resolve_model(scalar_group(model))
    family = detect_artifact_family(artifact)
    executable = resolve_executable(settings.executable)
    check_cancel = model_management.throw_exception_if_processing_interrupted
    while not _execution_lock.acquire(timeout=0.1):
        check_cancel()
    try:
        help_text, flags = probe_runtime(executable, check_cancel)
        check_artifact_runtime(artifact, help_text)
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ninferencing-", dir=temp_root) as temporary:
            directory = Path(temporary)
            sources = collect_image_sources(images, image_list, refs)
            paths = []
            clip = connected(video)
            text_prompt = first(prompt)
            video_tokens = 0
            if clip is not None:
                still_count = sum(1 for _ in image_frames(list(sources.values())))
                # A native video_url is ONE media item; stills keep the
                # VISION_ITEM_LIMIT - 1 budget.
                still_budget = max(0, VISION_ITEM_LIMIT - 1)
                if still_count > still_budget:
                    sources = limit_image_frames(sources, still_budget)
                clip_path, video_tokens = fit_video_for_vision(
                    clip, directory,
                    video_max_side=first(video_max_side) or 0,
                    # Leave room for text, stills and output inside context_length.
                    token_budget=max(512, min(8192, int(settings.context_length) // 3)),
                    video_duration=float(first(video_duration) or 0.0),
                    family=family,
                )
                paths.append(clip_path)
            messages, has_media = write_messages(
                directory, text_prompt, first(system_prompt),
                sources,
                settings.image_max_side, settings.context_length,
                video_paths=paths,
                video_token_reserve=video_tokens,
                family=family,
            )
            if clip is not None and not has_media:
                raise ValueError("Ninferencing: video export produced no media for ninfer.")
            comfy_device = model_management.get_torch_device()
            device = settings.device if settings.device >= 0 else (comfy_device.index or 0)
            use_video = bool(paths)
            command = build_command(
                executable, artifact, messages, settings, flags, help_text,
                max_tokens=first(max_tokens), seed=settings.seed, device=device,
                has_images=has_media, has_video=use_video, family=family,
            )
            need = estimate_need_bytes(artifact, has_media, has_video=use_video, family=family, command=command)
            report = prepare_memory(device, need_bytes=need, check_cancel=check_cancel)
            if family == "qwen_moe":
                command, _steps = fit_qwen_moe_residency(command, (report or {}).get("after_free"), flags)
            text, thinking = generate(
                command, settings.timeout, check_cancel, directory,
                on_capacity=lambda: prepare_memory(device, need_bytes=need, check_cancel=check_cancel),
                vram_report=report,
                tighten_on_capacity=True,
                family=family,
                flags=flags,
            )
        return text, thinking
    finally:
        _execution_lock.release()



class Ninferencing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ninferencing", display_name="Ninferencing", category="Ninferencing",
            description=(
                "Generate text with your local ninfer build. Optional VIDEO is a native video_url "
                "sent whole; refs, image_list and images are stills after the clip. "
                "Thinking and speculation on this node override Settings. The native "
                "process exits before the next node runs."
            ),
            is_input_list=True,
            inputs=[
                ninfer_model_input(),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", default="", multiline=True, advanced=True,
                                tooltip="Instructions for the model. Empty means no system message. Nothing is added behind your back."),
                io.Video.Input(
                    "video", optional=True,
                    tooltip=(
                        "Optional VIDEO. Sent whole as a native video_url; the loader owns "
                        "trimming (every input is start-0 full input). "
                        "Leave unconnected for stills or text."
                    ),
                ),
                io.Float.Input(
                    "video_duration", default=0.0, min=0.0, step=0.1, optional=True, advanced=True,
                    tooltip=(
                        "True plate length in seconds. 0 probes the VIDEO file. Wire your loader's "
                        "duration / duration_float here when container metadata is unreliable. "
                        "This never cuts the clip; it drives vision-token planning. Ignored when video is not connected."
                    ),
                ),
                io.Int.Input(
                    "video_max_side", default=0, min=0, max=2048, optional=True, advanced=True,
                    tooltip=(
                        "Downscale the clip's longest edge (even dimensions). 0 = 512 (about 2.2k vision "
                        "tokens for 15 s at 16:9); shrunk automatically to fit context_length/3."
                    ),
                ),
                io.Int.Input("max_tokens", default=4096, min=1, max=1048576, tooltip="Maximum generated tokens, including thinking when enabled."),
                io.Int.Input(
                    "context_length", default=32768, min=1, max=1048576, step=1024,
                    tooltip=(
                        "Prompt + media + output token capacity. Connecting video does not change this. "
                        "Default 32768. On 24 GB a connected clip makes kv_capacity auto match this context "
                        "(auto's 1024 MiB headroom does not fit native video). Lower to 8192–16384 for a clip. "
                        "For stills, 12288–16384 with Settings kv_capacity=match_context is safer than 32K auto."
                    ),
                ),
                io.Int.Input(
                    "image_max_side", default=0, min=0, max=32768,
                    tooltip="Downscale stills and image_list frames. 0 uses the largest side this frame count allows. Native video uses video_max_side.",
                ),
                io.Combo.Input(
                    "thinking", options=list(THINKING_OPTIONS), default="off", optional=True, advanced=True,
                    tooltip="Overrides Settings. off disables thinking, explicit efforts require runtime/model support.",
                ),
                io.Int.Input(
                    "thinking_budget", default=0, min=0, max=65536, step=256, optional=True, advanced=True,
                    tooltip="Overrides Settings. Cap model-origin thinking tokens (engine --thinking-budget). 0 leaves it unset. Ignored for Gemma 4.",
                ),
                io.Combo.Input(
                    "reasoning_loop", options=list(REASONING_LOOP_OPTIONS), default="off", optional=True, advanced=True,
                    tooltip="Overrides Settings. Engine --reasoning-loop: every 512 thinking tokens, test for repetition. off leaves it unset.",
                ),
                io.DynamicCombo.Input("speculation", options=[
                    io.DynamicCombo.Option("off", []),
                    *[io.DynamicCombo.Option(mode, [
                        io.Int.Input("draft_tokens", default=3, min=1, max=15),
                        io.Boolean.Input("lm_head_draft", default=False),
                    ]) for mode in SPECULATION_MODES],
                ], optional=True, tooltip="Overrides Settings. Opt in only if your model contains the required draft weights (engine range 1..15)."),
                SettingsType.Input("settings", optional=True),
                refs_input(),
                io.Image.Input(
                    "image_list", optional=True, advanced=True,
                    tooltip=(
                        "Backup IMAGE batch or image list (OUTPUT_IS_LIST). Continues after CF Image Bin refs. "
                        "Do not also wire the bin's image_list."
                    ),
                ),
                io.Autogrow.Input(
                    "images", optional=True, extra_dict={"advanced": True},
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image"), prefix="image", min=0, max=100,
                    ),
                    tooltip="Backup stills after CF Image Bin refs, in socket order. Do not also wire the bin's image_list.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, model, settings=None, **kwargs):
        path = resolve_model(scalar_group(model))
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns)
        if first(settings) is None:
            executable = resolve_executable("")
            stat = executable.stat()
            signature += (str(executable), stat.st_size, stat.st_mtime_ns)
        for key in ("thinking", "thinking_budget", "reasoning_loop", "speculation"):
            value = kwargs.get(key)
            if value is not None:
                try:
                    node_value = first(value)
                    if isinstance(node_value, dict):
                        node_value = first(scalar_group(node_value))
                    signature += (key, str(node_value))
                except Exception:
                    signature += (key, str(value))
        return signature

    @classmethod
    def execute(
        cls, model, prompt, system_prompt, max_tokens, context_length, image_max_side,
        settings=None, images=None, image_list=None, refs=None,
        video=None, video_duration=0.0, video_max_side=0,
        thinking="off", thinking_budget=0, reasoning_loop="off",
        speculation="off", **kwargs,
    ):
        text, thinking = run_ninfer(
            model=model, prompt=prompt, system_prompt=system_prompt,
            max_tokens=max_tokens, context_length=context_length, image_max_side=image_max_side,
            settings=settings, images=images, image_list=image_list, refs=refs,
            video=video, video_duration=video_duration,
            video_max_side=video_max_side,
            thinking=thinking, thinking_budget=thinking_budget,
            reasoning_loop=reasoning_loop, speculation=speculation,
        )
        return io.NodeOutput(text, thinking)


class NinferencingSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NinferencingSettings", display_name="Ninferencing Settings", category="Ninferencing",
            description="Optional runtime and sampling controls. Defaults favor predictable VRAM use. Auto options adapt to the selected binary's --help.",
            inputs=[
                io.String.Input("executable", default=default_executable(), tooltip="Path to ninfer, or its installation directory. Uses NINFER_EXECUTABLE / NINFER_HOME / PATH when blank."),
                io.Int.Input("device", default=-1, min=-1, max=255, tooltip="-1 uses ComfyUI's device. Other values are CUDA indices within the inherited CUDA_VISIBLE_DEVICES mapping."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=io.ControlAfterGenerate.fixed, tooltip="Manual sampling seed. It stays fixed after generation and defaults to 0."),
                io.Combo.Input("kv_cache", options=["auto", "bf16", "int8", "fp8", "k8v4", "nvfp4", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8"], default="auto", tooltip="Auto: INT8 for Qwen/Bonsai, rk4v4 for Qwen3.6-35B-A3B, rk8v4 for Gemma 4 A4B when the build advertises it (3090 recommendation). Prefer int8 on 24 GB for Qwen; rk8v4 for Gemma. fp8 / k8v4 / nvfp4 are smaller KV types some builds advertise for large contexts."),
                io.Combo.Input(
                    "kv_capacity",
                    options=["auto", "match_context", "4096", "8192", "12288", "16384", "32768"],
                    default="auto",
                    tooltip=(
                        "KV token capacity passed to ninfer. auto = remaining VRAM minus 1024 MiB headroom "
                        "(good for plain text / few stills). match_context = --kv-capacity equals context_length "
                        "(required for native video on 24 GB + 27B; also preferred for multimodal image "
                        "rewrite at 12–16K). Numeric values force that many tokens. "
                        "A connected clip treats Settings auto as match-context. Connecting video does not change context_length. On capacity shortfall, "
                        "image retries step through 16K→12K→8K explicit KV rather than slamming to 8K first."
                    ),
                ),
                io.Int.Input("prefill_chunk", default=1024, min=128, max=16384, step=128, advanced=True, tooltip="--prefill-chunk in tokens: larger chunks speed up prefill (must be a multiple of 128). 1024 is the engine default; 2048+ pairs with --prefill-cublas for long vision prompts."),
                io.Combo.Input("cuda_graphs", options=["auto", "on", "off"], default="auto", tooltip="Auto disables capture when the build supports it to reduce memory pressure. On uses the runtime default."),
                io.Combo.Input("thinking", options=["auto", "model default", "off", "low", "medium", "high", "xhigh"], default="auto", tooltip="Auto disables thinking when supported. Explicit efforts require support from your runtime and model."),
                io.Int.Input("thinking_budget", default=0, min=0, max=65536, step=256, advanced=True, tooltip="Cap model-origin thinking tokens (engine --thinking-budget; inserted control tokens count toward --max-new). 0 leaves it unset. Ignored for Gemma 4 (no early-close control)."),
                io.Combo.Input("reasoning_loop", options=["off", "stop", "conclude"], default="off", advanced=True, tooltip="Engine --reasoning-loop: every 512 thinking tokens, test whether thinking repeats whole passages. stop ends the reply, conclude closes thinking and lets the model answer. off leaves it unset."),
                io.DynamicCombo.Input("speculation", options=[
                    io.DynamicCombo.Option("off", []),
                    *[io.DynamicCombo.Option(mode, [
                        io.Int.Input("draft_tokens", default=3, min=1, max=15),
                        io.Boolean.Input("lm_head_draft", default=False),
                    ]) for mode in ("mtp", "dflash", "dflash2")],
                ], tooltip="Opt in only if your model contains the required draft weights (engine range 1..15). Gemma 4 uses MTP (engine default when off and the artifact has the drafter); dflash/dflash2 are remapped to MTP for Gemma. dflash2 needs the ninfer-all CLI and a Bonsai/v3 artifact with the adapter."),
                io.DynamicCombo.Input("sampling", options=[
                    io.DynamicCombo.Option("custom", [
                        io.Float.Input(
                            "temperature", default=QWEN38_NOTHINK_SAMPLING["temperature"], min=0, max=2, step=0.01,
                            tooltip="Qwen3.8-27B non-thinking default is 0.7. 0..2. Values outside this build's range are clamped.",
                        ),
                        io.Float.Input(
                            "top_p", default=QWEN38_NOTHINK_SAMPLING["top_p"], min=0, max=1, step=0.01,
                            tooltip="Qwen3.8-27B non-thinking default is 0.8. 0..1.",
                        ),
                        io.Int.Input(
                            "top_k", default=QWEN38_NOTHINK_SAMPLING["top_k"], min=0, max=20,
                            tooltip="Qwen3.8-27B non-thinking default is 20. 0..20. Higher values are clamped so the run does not fail.",
                        ),
                        io.Float.Input(
                            "min_p", default=QWEN38_NOTHINK_SAMPLING["min_p"], min=0, max=1, step=0.01,
                            tooltip="Qwen3.8-27B non-thinking default is 0. 0..1.",
                        ),
                        io.Float.Input(
                            "presence_penalty", default=QWEN38_NOTHINK_SAMPLING["presence_penalty"], min=-2, max=2, step=0.01,
                            tooltip="Qwen3.8-27B non-thinking default is 1.5. -2..2. Higher values cut repetition; too high can mix languages.",
                        ),
                        io.Float.Input(
                            "frequency_penalty", default=QWEN38_NOTHINK_SAMPLING["frequency_penalty"], min=-2, max=2, step=0.01,
                            tooltip="-2..2. Left at 0. Official repetition_penalty is 1.0 (no extra penalty); this CLI has no repetition-penalty flag.",
                        ),
                    ]),
                    io.DynamicCombo.Option("model defaults", []),
                ], tooltip=(
                    "Default is the Qwen3.8-27B non-thinking sampler (temperature 0.7, top_p 0.8, top_k 20, "
                    "min_p 0, presence_penalty 1.5). Thinking stays off. Model defaults uses the artifact sampler "
                    "(thinking profile). Values are clamped to the ranges this build advertises."
                )),
                io.Float.Input("timeout", default=600, min=1, max=86400, advanced=True, tooltip="Seconds allowed for model loading and generation together."),
                io.String.Input("extra_args", default="[]", multiline=True, advanced=True, tooltip='Build-specific CLI arguments as a JSON string array, e.g. ["--prefill-chunk", "256"]. These override matching runtime settings.'),
            ],
            outputs=[SettingsType.Output(display_name="settings")],
        )

    @classmethod
    def fingerprint_inputs(cls, executable, **kwargs):
        path = resolve_executable(executable)
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns

    @classmethod
    def execute(cls, **kwargs):
        cleaned = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = {name: item for name, item in value.items() if item is not None}
            cleaned[key] = value
        return io.NodeOutput(Settings(**cleaned))
