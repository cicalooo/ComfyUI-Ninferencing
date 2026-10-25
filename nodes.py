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

from .models import default_executable, resolve_executable, resolve_model
from .runtime import (
    QWEN38_NOTHINK_SAMPLING,
    VISION_ITEM_LIMIT,
    VISION_TOKEN_CHARS,
    Settings,
    build_command,
    check_artifact_runtime,
    generate,
    probe_runtime,
    vision_side_limit,
)
from .videosample import build_clip, fit_side_for_tokens, ninfer_video_plan, probe_video

LOGGER = logging.getLogger(__name__)
from .vram import estimate_need_bytes, prepare_for_ninfer


SettingsType = io.Custom("NINFERENCING_SETTINGS")
# CF Image Bin cable: ordered tuple of [1,H,W,C] IMAGE tensors. Match the string; do not import the other pack.
CF_REFS_TYPE = "CF_REFS"
_execution_lock = threading.Lock()
_SYSTEM_PROMPT = (
    "Improve the user's image-generation prompt. Preserve their intent and add concrete visual detail. "
    "Reply with only the finished prompt, without a preamble or analysis."
)


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


def _video_duration(video):
    """Best-effort source length in seconds, or None when unavailable."""
    try:
        duration = float(video.get_duration())
    except Exception:
        return None
    if duration != duration:  # NaN
        return None
    return duration


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


# Legacy "sampled" mode only: frames reserved for the thinned clip out of the
# still budget. 0 = original behaviour (stills keep VISION_ITEM_LIMIT - 1).
VIDEO_MIN_FRAMES = 0
# "full" mode: ninfer decodes the clip itself at a chosen fps (default 2, min 4, max
# 768 frames) with real timestamps; a native video_url is one media item.
VIDEO_FULL_MAX_SECONDS = 16.0
VIDEO_FULL_DEFAULT_SIDE = 512
VIDEO_MODES = ("full", "sampled (legacy)")
VIDEO_FPS_CHOICES = ("2", "3", "4", "5", "6")
DEFAULT_VIDEO_FPS = 2


def parse_video_fps(value):
    raw = first(value) if not isinstance(value, (int, float)) else value
    try:
        fps = int(raw)
    except (TypeError, ValueError):
        fps = DEFAULT_VIDEO_FPS
    if fps not in (2, 3, 4, 5, 6):
        raise ValueError(f"Ninferencing: video_fps must be 2, 3, 4, 5 or 6, not {value!r}.")
    return fps


def fit_video_for_vision(video, directory, *, start_seconds=0.0, duration_seconds=0.0, max_frames=VISION_ITEM_LIMIT, video_max_side=0,
                         mode="sampled (legacy)", max_seconds=VIDEO_FULL_MAX_SECONDS, token_budget=None,
                         video_fps=DEFAULT_VIDEO_FPS):
    """Export a video_url file whose frame count ninfer will accept.

    A long plate is not rejected. Frames are spread across the clip so a 15s
    plate still describes the whole action, and the result keeps the source
    length (``videosample`` picks the output rate), so an analysis timeline in
    ``[MM:SS.mmm]`` still maps onto the plate it came from. Decoding is
    streamed from the exported file — the plate is never pulled into a tensor
    batch just to be thinned.

    When ``video_max_side`` > 0 the clip is also downscaled (longest edge, even
    dimensions) so large plates do not exhaust vision context.
    """
    full = mode == "full"
    if full and max_seconds and max_seconds > 0:
        if not duration_seconds or duration_seconds <= 0 or duration_seconds > max_seconds:
            source = video_duration_seconds(video)
            remaining = None if source is None else max(0.0, source - float(start_seconds or 0.0))
            wanted = duration_seconds if duration_seconds and duration_seconds > 0 else remaining
            if wanted is None or wanted > max_seconds:
                LOGGER.info("Ninferencing: full-clip video capped at %.1f s (video_max_seconds).", max_seconds)
                duration_seconds = float(max_seconds)
    exported = export_video_path(
        video, directory, start_seconds=start_seconds, duration_seconds=duration_seconds,
    )
    try:
        side = int(video_max_side or 0)
    except (TypeError, ValueError):
        side = 0
    if side < 0:
        side = 0
    if max_frames < 1:
        return exported
    info = probe_video(exported)
    if info is None:
        # Not a container we can inspect (or PyAV refused it): send as-is and
        # let the CLI report what it could not read.
        return exported
    if full:
        if side <= 0:
            side = VIDEO_FULL_DEFAULT_SIDE
        if token_budget:
            fitted = fit_side_for_tokens(
                info.width, info.height, info.frame_count, info.rate, side, token_budget, fps=video_fps,
            )
            if fitted < side and max(info.width, info.height) > fitted:
                LOGGER.info("Ninferencing: video_max_side %d -> %d to keep the clip within %d vision tokens.",
                            side, fitted, token_budget)
                side = fitted
        clip = build_clip(info, Path(directory) / "input_video_full.mp4", max_frames=0, max_side=side, full=True)
        final = probe_video(clip) if clip is not None else info
        if final is not None:
            frames, w, h, tokens, stamps = ninfer_video_plan(
                final.width, final.height, final.frame_count, final.rate, fps=video_fps,
            )
            LOGGER.info(
                "Ninferencing: full clip %.2f s @ %.3f fps, %dx%d -> ninfer samples %d frames at %.0f fps, "
                "vision grid %dx%d, ~%d vision tokens, timestamps %.2f..%.2f s.",
                final.duration, float(final.rate), final.width, final.height, frames, video_fps,
                w, h, tokens, stamps[0] if stamps else 0.0, stamps[-1] if stamps else 0.0,
            )
        return clip if clip is not None else exported
    clip = build_clip(
        info, Path(directory) / "input_video_sampled.mp4",
        max_frames=max_frames, max_side=side,
    )
    return clip if clip is not None else exported


def video_duration_seconds(video):
    """Best-effort source duration of a Comfy VIDEO without decoding frames."""
    for name in ("get_duration",):
        getter = getattr(video, name, None)
        if callable(getter):
            try:
                value = float(getter())
                if value > 0:
                    return value
            except Exception:
                pass
    return None


def export_video_path(video, directory, *, start_seconds=0.0, duration_seconds=0.0):
    """Materialize a local file path for ninfer video_url parts.

    Prefer an existing on-disk stream when the clip is untrimmed. Trimmed windows
    are always re-encoded so ninfer sees the selected span (Comfy trim metadata
    is not part of the file URL contract).

    When source duration is known, start is clamped into ``[0, duration)`` and a
    positive requested length is clamped to the remaining span so the window is
    never empty. A request that already covers the full input skips
    ``as_trimmed`` (avoids double-trim when Prep already shortened the plate and
    Duration is still wired into this node). A start at or past EOF raises with
    diagnostics instead of Comfy's opaque empty-trim ``None``.
    """
    from comfy_api.latest import VideoCodec, VideoContainer

    if video is None:
        raise ValueError("Ninferencing: connect a VIDEO input.")
    start = max(0.0, float(start_seconds or 0.0))
    requested = max(0.0, float(duration_seconds or 0.0))
    duration = requested
    source_duration = _video_duration(video)

    if source_duration is not None:
        if source_duration <= 0.0:
            raise ValueError(
                f"Ninferencing: source duration is empty "
                f"(source_duration={source_duration:.4g}s)."
            )
        if start >= source_duration:
            raise ValueError(
                "Ninferencing: trim window is past end of clip "
                f"(source_duration={source_duration:.4g}s, start={start:.4g}s, "
                f"requested_duration={requested:.4g}s)."
            )
        remaining = source_duration - start
        if duration > 0.0:
            duration = min(duration, remaining)
        # Full input after clamp → skip as_trimmed (safe for Prep→Video double-wire).
        effective_end = source_duration if duration <= 0.0 else start + duration
        if start <= 0.0 and effective_end >= source_duration - 1e-6:
            return _reuse_or_save_video(video, directory)

    # Prefer the on-disk stream when untrimmed. Only call as_trimmed when a
    # window is set. Pass floats (never None): comfy VideoFromComponents.as_trimmed
    # does `start_time + duration` and TypeErrors on None; VideoFromFile treats
    # duration=0 as "to end".
    needs_trim = start > 0.0 or duration > 0.0
    if not needs_trim:
        return _reuse_or_save_video(video, directory)

    clip = video.as_trimmed(start_time=start, duration=duration)
    if clip is None:
        # After clamp this is rare (float / strict_duration quirks). Fall back to
        # the untrimmed clip only when the requested window already covered it.
        if (
            source_duration is not None
            and start <= 0.0
            and (requested <= 0.0 or duration >= source_duration - 1e-6)
        ):
            return _reuse_or_save_video(video, directory)
        detail = (
            f"source_duration={source_duration:.4g}s, "
            if source_duration is not None
            else "source_duration=unknown, "
        )
        raise ValueError(
            "Ninferencing: start/duration trim is empty "
            f"({detail}start={start:.4g}s, requested_duration={requested:.4g}s, "
            f"clamped_duration={duration:.4g}s). "
            "Set duration_seconds to 0 to send the full input."
        )

    path = Path(directory) / "input_video.mp4"
    model_management.throw_exception_if_processing_interrupted()
    clip.save_to(str(path), format=VideoContainer.MP4, codec=VideoCodec.H264)
    return path


def write_messages(directory, prompt, system_prompt, images, max_side, context_length, video_paths=None):
    """Build messages.json for ninfer.

    Still images become image_url parts. Native video files become video_url
    parts (ninfer 0.6.x --vision). video_paths are written first so identity
    stills remain the later attachments when both are present.
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
            "Ninferencing: %d CF Image Bin refs as <Picture 1>..<Picture %d>, each saved as its own image.",
            len(pinned), len(pinned),
        )
    text_tokens = (len(prompt) + len(system_prompt)) // VISION_TOKEN_CHARS
    # Native video is one media item for the preprocessor; still side caps only
    # apply to image/frame attachments sharing the request.
    max_side = vision_side_limit(len(frames), max_side, context_length, text_tokens)
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
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
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
    """Attachment order: CF Image Bin refs, then autogrow sockets, then image_list."""
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
            "Ordered refs from CF Image Bin (CF_REFS). Bin order is <Picture 1> onward. "
            "Each still is saved as its own image and is not thinned. "
            "If video is connected, the clip stays <Video 1>. "
            "image_list and the images sockets continue after the bin. "
            "Do not also wire the bin's image_list."
        ),
    )


def ninfer_model_input():
    names = [name for name in folder_paths.get_filename_list("ninfer") if name.lower().endswith(".ninfer")]
    return io.DynamicCombo.Input("model", options=[
        io.DynamicCombo.Option("library", [io.Combo.Input("name", options=names or ["No .ninfer models found"], display_name="model file")]),
        io.DynamicCombo.Option("file path", [io.String.Input("path", default="", display_name="model file", tooltip="Full path to any .ninfer artifact supported by your build.")]),
    ])


def run_ninfer(*, model, prompt, system_prompt, max_tokens, context_length, image_max_side,
               settings=None, images=None, image_list=None, refs=None,
               video=None, start_at_timestamp=0.0, duration_seconds=0.0, video_max_side=0,
               video_paths=None, has_video=False, video_mode="full", video_max_seconds=VIDEO_FULL_MAX_SECONDS,
               video_fps=DEFAULT_VIDEO_FPS):
    """CLI invocation. Optional video is a native video_url; stills stay image_url."""
    settings = replace(
        first(settings) or Settings(),
        context_length=first(context_length),
        image_max_side=first(image_max_side),
    )
    artifact = resolve_model(scalar_group(model))
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
            paths = [Path(path) for path in (video_paths or []) if path]
            clip = connected(video)
            if clip is not None:
                still_count = sum(1 for _ in image_frames(list(sources.values())))
                # A native video_url is ONE media item; stills keep the original
                # VISION_ITEM_LIMIT - 1 budget. VIDEO_MIN_FRAMES (legacy, opt-in) can
                # reserve frames for the thinned clip out of that budget.
                mode = first(video_mode) or "full"
                still_budget = max(0, VISION_ITEM_LIMIT - 1)
                if mode != "full" and VIDEO_MIN_FRAMES > 0:
                    pinned_count = len(split_ordered_frames(sources)[0])
                    still_budget = min(still_budget, max(VISION_ITEM_LIMIT - VIDEO_MIN_FRAMES, pinned_count))
                if still_count > still_budget:
                    sources = limit_image_frames(sources, still_budget)
                    still_count = still_budget
                fps = parse_video_fps(video_fps)
                paths.append(fit_video_for_vision(
                    clip, directory,
                    start_seconds=float(first(start_at_timestamp) or 0.0),
                    duration_seconds=float(first(duration_seconds) or 0.0),
                    max_frames=max(1, VISION_ITEM_LIMIT - still_count),
                    video_max_side=first(video_max_side) or 0,
                    mode=mode,
                    max_seconds=float(first(video_max_seconds) or 0.0),
                    # Leave room for text, stills and output inside context_length.
                    token_budget=max(512, min(8192, int(settings.context_length) // 3)),
                    video_fps=fps,
                ))
            messages, has_media = write_messages(
                directory, first(prompt), first(system_prompt),
                sources,
                settings.image_max_side, settings.context_length,
                video_paths=paths,
            )
            if clip is not None and not has_media:
                raise ValueError("Ninferencing: video export produced no media for ninfer.")
            comfy_device = model_management.get_torch_device()
            device = settings.device if settings.device >= 0 else (comfy_device.index or 0)
            use_video = bool(has_video or paths)
            command = build_command(
                executable, artifact, messages, settings, flags, help_text,
                max_tokens=first(max_tokens), seed=settings.seed, device=device,
                has_images=has_media, has_video=use_video,
                video_fps=parse_video_fps(video_fps) if use_video else DEFAULT_VIDEO_FPS,
            )
            need = estimate_need_bytes(artifact, has_media, has_video=use_video)
            report = prepare_memory(device, need_bytes=need, check_cancel=check_cancel)
            text = generate(
                command, settings.timeout, check_cancel, directory,
                on_capacity=lambda: prepare_memory(device, need_bytes=need, check_cancel=check_cancel),
                vram_report=report,
                tighten_on_capacity=True,
            )
        return text
    finally:
        _execution_lock.release()



class Ninferencing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ninferencing", display_name="Ninferencing", category="Ninferencing",
            description=(
                "Generate text with your local ninfer build. Optional VIDEO is a native video_url "
                "(<Video 1>). CF Image Bin refs are <Picture 1> onward. images and image_list are extra "
                "stills after the bin. Context stays at this widget; a connected clip makes kv_capacity "
                "auto match context. The native process exits before the next node runs."
            ),
            is_input_list=True,
            inputs=[
                ninfer_model_input(),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", default=_SYSTEM_PROMPT, multiline=True, advanced=True,
                                tooltip="Instructions for the model. Edit or clear it. Leave empty for no system message."),
                io.Video.Input(
                    "video", optional=True,
                    tooltip=(
                        "Optional VIDEO. Sent as a native video_url (<Video 1>). "
                        "A long plate is sampled across its full length instead of rejected. "
                        "Leave unconnected for stills or text."
                    ),
                ),
                io.Float.Input(
                    "start_at_timestamp", default=0.0, min=0.0, step=0.1, optional=True, advanced=True,
                    tooltip="Seconds to skip before the clip. 0 keeps the start. Ignored when video is not connected.",
                ),
                io.Float.Input(
                    "duration_seconds", default=0.0, min=0.0, step=0.1, optional=True, advanced=True,
                    tooltip="Clip length after the start offset. 0 = full input. Ignored when video is not connected.",
                ),
                io.Int.Input(
                    "video_max_side", default=0, min=0, max=2048, optional=True, advanced=True,
                    tooltip=(
                        "Downscale the clip's longest edge (even dimensions). full mode: 0 = 512 (about 2.2k vision "
                        "tokens for 15 s at 16:9); shrunk automatically to fit context_length/3. sampled mode: 0 = no resize."
                    ),
                ),
                io.Combo.Input(
                    "video_mode", options=list(VIDEO_MODES), default="full", optional=True, advanced=True,
                    tooltip=(
                        "full: send the whole clip (every frame, real fps) as one video_url; ninfer samples it itself "
                        "at video_fps (default 2, ~30 frames for 15 s) with true timestamps, and stills keep their full budget. "
                        "sampled (legacy): thin to at most 16 - stills frames before sending (old behaviour)."
                    ),
                ),
                io.Combo.Input(
                    "video_fps", options=list(VIDEO_FPS_CHOICES), default="2", optional=True, advanced=True,
                    tooltip=(
                        "Ninfer sample rate for a connected clip. 2 is the trained default and the cheap path. "
                        "3–6 decode more frames and cost vision tokens plus VRAM. 5 and 6 are for short plates "
                        "(under 10 s, side left at 512). Needs a ninfer build that advertises --video-fps."
                    ),
                ),
                io.Float.Input(
                    "video_max_seconds", default=VIDEO_FULL_MAX_SECONDS, min=0.0, max=600.0, step=0.5, optional=True, advanced=True,
                    tooltip="full mode: cap the clip length sent (after start_at_timestamp). 0 = no cap (ninfer allows up to 600 s / 768 sampled frames).",
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
                SettingsType.Input("settings", optional=True),
                refs_input(),
                io.Image.Input(
                    "image_list", optional=True, advanced=True,
                    tooltip=(
                        "Backup IMAGE batch or image list (OUTPUT_IS_LIST). Continues after CF Image Bin refs. "
                        "Do not also wire the bin's image_list or Picture numbers repeat."
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
            outputs=[io.String.Output(display_name="text")],
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
        return signature

    @classmethod
    def execute(
        cls, model, prompt, system_prompt, max_tokens, context_length, image_max_side,
        settings=None, images=None, image_list=None, refs=None,
        video=None, start_at_timestamp=0.0, duration_seconds=0.0, video_max_side=0,
        video_mode="full", video_max_seconds=VIDEO_FULL_MAX_SECONDS, video_fps="2", **kwargs,
    ):
        text = run_ninfer(
            model=model, prompt=prompt, system_prompt=system_prompt,
            max_tokens=max_tokens, context_length=context_length, image_max_side=image_max_side,
            settings=settings, images=images, image_list=image_list, refs=refs,
            video=video, start_at_timestamp=start_at_timestamp, duration_seconds=duration_seconds,
            video_max_side=video_max_side, video_mode=video_mode, video_max_seconds=video_max_seconds,
            video_fps=video_fps,
        )
        return io.NodeOutput(text)


def run_video_ninfer(
    *, model, video, prompt, system_prompt, start_at_timestamp, duration_seconds,
    max_tokens, context_length, image_max_side, settings=None, images=None, image_list=None,
    refs=None, video_max_side=0,
):
    """Analysis-node entry. Same path as Ninferencing with a required clip."""
    return run_ninfer(
        model=model, prompt=prompt, system_prompt=system_prompt,
        max_tokens=max_tokens, context_length=context_length, image_max_side=image_max_side,
        settings=settings, images=images, image_list=image_list, refs=refs,
        video=video, start_at_timestamp=start_at_timestamp, duration_seconds=duration_seconds,
        video_max_side=video_max_side, has_video=True,
    )


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
                io.Combo.Input("kv_cache", options=["auto", "bf16", "int8", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8"], default="auto", tooltip="Auto uses INT8 if advertised by the build; otherwise its native default. Prefer int8 on 24 GB cards. ninfer-all also advertises rk4v4 / rk4v4-e8 / rk2v4-e8."),
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
                io.Combo.Input("cuda_graphs", options=["auto", "on", "off"], default="auto", tooltip="Auto disables capture when the build supports it to reduce memory pressure. On uses the runtime default."),
                io.Combo.Input("thinking", options=["auto", "model default", "off", "low", "medium", "high", "xhigh"], default="auto", tooltip="Auto disables thinking when supported. Explicit efforts require support from your runtime and model."),
                io.DynamicCombo.Input("speculation", options=[
                    io.DynamicCombo.Option("off", []),
                    *[io.DynamicCombo.Option(mode, [
                        io.Int.Input("draft_tokens", default=3, min=1, max=256),
                        io.Boolean.Input("lm_head_draft", default=False),
                    ]) for mode in ("mtp", "dflash", "dflash2")],
                ], tooltip="Opt in only if your model contains the required draft weights. dflash2 needs the ninfer-all CLI and a Bonsai/v3 artifact with the adapter."),
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
