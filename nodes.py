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
    VIDEO_SAFE_CONTEXT,
    VISION_ITEM_LIMIT,
    VISION_TOKEN_CHARS,
    Settings,
    build_command,
    check_artifact_runtime,
    generate,
    probe_runtime,
    vision_side_limit,
)
from .videosample import build_clip, probe_video

LOGGER = logging.getLogger(__name__)
from .vram import estimate_need_bytes, prepare_for_ninfer


SettingsType = io.Custom("NINFERENCING_SETTINGS")
_execution_lock = threading.Lock()
_SYSTEM_PROMPT = (
    "Improve the user's image-generation prompt. Preserve their intent and add concrete visual detail. "
    "Reply with only the finished prompt, without a preamble or analysis."
)
_VIDEO_SYSTEM_PROMPT = (
    "Improve the user's video-generation or video-edit prompt using the attached video. "
    "Preserve their intent and match the source plate motion, timing, camera moves, "
    "blocking, cuts, and on-screen action 1:1 — do not rephrase into a different "
    "choreography or cinematic restage. "
    "Add concrete visual detail only when it is grounded in what the video shows. "
    "Reply with only the finished prompt, without a preamble or analysis."
)


def first(value):
    return value[0] if isinstance(value, list) else value


def resolve_clip_seconds(video, seconds=0, duration_seconds=0.0, start_seconds=0.0) -> int:
    """Whole seconds for the H3 timeline.

    An INT ``seconds`` wire wins. Otherwise a positive duration is rounded.
    Otherwise the video's own length, after the start offset, is rounded.
    """
    try:
        forced = int(seconds or 0)
    except (TypeError, ValueError):
        forced = 0
    if forced > 0:
        return forced
    try:
        requested = float(duration_seconds or 0.0)
    except (TypeError, ValueError):
        requested = 0.0
    if requested > 0.0:
        return max(1, int(round(requested)))
    duration = _video_duration(video)
    if duration is None:
        return 0
    remaining = duration - max(0.0, float(start_seconds or 0.0))
    if remaining <= 0.0:
        return 0
    return max(1, int(round(remaining)))


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


def limit_image_frames(images, budget):
    """Drop a frame grid down to the vision item budget, spread across the input."""
    frames = list(image_frames(list((images or {}).values())))
    if len(frames) <= budget:
        return images
    chosen = spread_indices(len(frames), budget)
    LOGGER.info(
        "Ninferencing: %d images/frames exceed the %d-item vision budget; sending %d spread across the input.",
        len(frames), VISION_ITEM_LIMIT, len(chosen),
    )
    return {"image_list": [frames[index] for index in chosen]}


def fit_video_for_vision(video, directory, *, start_seconds=0.0, duration_seconds=0.0, max_frames=VISION_ITEM_LIMIT, video_max_side=0):
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
    clip = build_clip(
        info, Path(directory) / "input_video_sampled.mp4",
        max_frames=max_frames, max_side=side,
    )
    return clip if clip is not None else exported


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
        raise ValueError("Ninferencing Video: connect a VIDEO input.")
    start = max(0.0, float(start_seconds or 0.0))
    requested = max(0.0, float(duration_seconds or 0.0))
    duration = requested
    source_duration = _video_duration(video)

    if source_duration is not None:
        if source_duration <= 0.0:
            raise ValueError(
                f"Ninferencing Video: source duration is empty "
                f"(source_duration={source_duration:.4g}s)."
            )
        if start >= source_duration:
            raise ValueError(
                "Ninferencing Video: trim window is past end of clip "
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
            "Ninferencing Video: start/duration trim is empty "
            f"({detail}start={start:.4g}s, requested_duration={requested:.4g}s, "
            f"clamped_duration={duration:.4g}s). "
            "If PrepRefVideo already trimmed the plate, set this node's "
            "duration_seconds to 0 (full input) instead of re-wiring H3J Duration."
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
    frames = [frame for batch in (images or {}).values() for frame in image_frames(batch)]
    videos = [Path(path) for path in (video_paths or []) if path]
    # A native video consumes the rest of the 16-item budget as sampled frames.
    # Still-only requests spend the whole budget on the frame grid.
    frame_budget = max(0, VISION_ITEM_LIMIT - (1 if videos else 0))
    if videos and len(frames) > frame_budget:
        frames = [frames[index] for index in spread_indices(len(frames), frame_budget)]
    elif not videos and len(frames) > VISION_ITEM_LIMIT:
        LOGGER.info(
            "Ninferencing: %d images/frames exceed the %d-item vision budget; sending %d spread across the input.",
            len(frames), VISION_ITEM_LIMIT, VISION_ITEM_LIMIT,
        )
        frames = [frames[index] for index in spread_indices(len(frames), VISION_ITEM_LIMIT)]
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


def collect_image_sources(images=None, image_list=None):
    sources = dict(images or {})
    if image_list is not None:
        sources["image_list"] = image_list
    return sources


def ninfer_model_input():
    names = [name for name in folder_paths.get_filename_list("ninfer") if name.lower().endswith(".ninfer")]
    return io.DynamicCombo.Input("model", options=[
        io.DynamicCombo.Option("library", [io.Combo.Input("name", options=names or ["No .ninfer models found"], display_name="model file")]),
        io.DynamicCombo.Option("file path", [io.String.Input("path", default="", display_name="model file", tooltip="Full path to any .ninfer artifact supported by your build.")]),
    ])


def run_ninfer(*, model, prompt, system_prompt, max_tokens, context_length, image_max_side,
               settings=None, images=None, image_list=None, video_paths=None, has_video=False):
    """Shared CLI invocation for image and video nodes."""
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
            messages, has_media = write_messages(
                directory, first(prompt), first(system_prompt),
                collect_image_sources(images, image_list),
                settings.image_max_side, settings.context_length,
                video_paths=video_paths,
            )
            comfy_device = model_management.get_torch_device()
            device = settings.device if settings.device >= 0 else (comfy_device.index or 0)
            use_video = bool(has_video or video_paths)
            command = build_command(
                executable, artifact, messages, settings, flags, help_text,
                max_tokens=first(max_tokens), seed=settings.seed, device=device,
                has_images=has_media, has_video=use_video,
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
            description="Generate text with your local ninfer build. Stills-only; for simple references and T2V with no timeline. Images enable vision automatically. The native process exits before the next node runs. Use Ninferencing Video for video swaps.",
            is_input_list=True,
            inputs=[
                ninfer_model_input(),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", default=_SYSTEM_PROMPT, multiline=True, advanced=True,
                                tooltip="Instructions for the model. Edit for general chat, captioning, or other tasks; leave empty for no system message."),
                io.Int.Input("max_tokens", default=4096, min=1, max=1048576, tooltip="Maximum generated tokens, including thinking when enabled."),
                io.Int.Input(
                    "context_length", default=32768, min=1, max=1048576, step=1024,
                    tooltip=(
                        "Prompt + images + output token capacity. Must be ≥ prepared prompt "
                        "(text + vision tokens). For RTX 3090 multimodal stills (identity + sparse "
                        "frames) prefer 12288–16384 with Settings kv_capacity=match_context; auto can "
                        "VRAM-shortfall then a too-tight retry may report prepared prompt exceeds capacity."
                    ),
                ),
                io.Int.Input("image_max_side", default=0, min=0, max=32768, tooltip="Downscale every input image or video frame so its longest side is at most this size. 0 uses the largest side this ninfer build's vision budget allows for the number of frames."),
                SettingsType.Input("settings", optional=True),
                io.Image.Input(
                    "image_list", optional=True,
                    tooltip=(
                        "Optional IMAGE batch or ComfyUI image list (OUTPUT_IS_LIST). "
                        "Use for multi-frame sources such as H3J Video Frames / Image Bin. "
                        "All frames join the Autogrow sockets in one request. "
                        "Stills attach in order as <Picture 1> onward; no timeline."
                    ),
                ),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image"), prefix="image", min=0, max=100,
                ), tooltip="Optional images, batches, or image lists. All frames are included in one request as <Picture> refs in attachment order. No timeline. 100 sockets is ComfyUI's native limit."),
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
    def execute(cls, model, prompt, system_prompt, max_tokens, context_length, image_max_side, settings=None, images=None, image_list=None, **kwargs):
        text = run_ninfer(
            model=model, prompt=prompt, system_prompt=system_prompt,
            max_tokens=max_tokens, context_length=context_length, image_max_side=image_max_side,
            settings=settings, images=images, image_list=image_list,
        )
        return io.NodeOutput(text)


def run_video_ninfer(
    *, model, video, prompt, system_prompt, start_at_timestamp, duration_seconds,
    max_tokens, context_length, image_max_side, settings=None, images=None, image_list=None,
    video_max_side=0,
):
    """Native video_url request. Shares the execution lock with still-image runs."""
    check_cancel = model_management.throw_exception_if_processing_interrupted
    settings_obj = replace(
        first(settings) or Settings(),
        context_length=first(context_length),
        image_max_side=first(image_max_side),
    )
    artifact = resolve_model(scalar_group(model))
    executable = resolve_executable(settings_obj.executable)
    while not _execution_lock.acquire(timeout=0.1):
        check_cancel()
    try:
        help_text, flags = probe_runtime(executable, check_cancel)
        check_artifact_runtime(artifact, help_text)
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ninferencing-", dir=temp_root) as temporary:
            directory = Path(temporary)
            sources = collect_image_sources(images, image_list)
            still_count = sum(1 for _ in image_frames(list(sources.values())))
            still_budget = max(0, VISION_ITEM_LIMIT - 1)
            if still_count > still_budget:
                sources = limit_image_frames(sources, still_budget)
                still_count = still_budget
            video_path = fit_video_for_vision(
                first(video), directory,
                start_seconds=first(start_at_timestamp),
                duration_seconds=first(duration_seconds),
                max_frames=max(1, VISION_ITEM_LIMIT - still_count),
                video_max_side=first(video_max_side),
            )
            messages, has_media = write_messages(
                directory, first(prompt), first(system_prompt),
                sources,
                settings_obj.image_max_side, settings_obj.context_length,
                video_paths=[video_path],
            )
            if not has_media:
                raise ValueError("Ninferencing Video: video export produced no media for ninfer.")
            comfy_device = model_management.get_torch_device()
            device = settings_obj.device if settings_obj.device >= 0 else (comfy_device.index or 0)
            command = build_command(
                executable, artifact, messages, settings_obj, flags, help_text,
                max_tokens=first(max_tokens), seed=settings_obj.seed, device=device,
                has_images=True, has_video=True,
            )
            need = estimate_need_bytes(artifact, True, has_video=True)
            report = prepare_memory(device, need_bytes=need, check_cancel=check_cancel)
            return generate(
                command, settings_obj.timeout, check_cancel, directory,
                on_capacity=lambda: prepare_memory(device, need_bytes=need, check_cancel=check_cancel),
                vram_report=report,
                tighten_on_capacity=True,
            )
    finally:
        _execution_lock.release()


class NinferencingVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NinferencingVideo",
            display_name="Ninferencing Video",
            category="Ninferencing",
            description=(
                "Generate text with ninfer using native VIDEO (video_url) when possible. "
                "Optional stills share the same request. Outputs text and enhanced_video "
                "for H3J struct handoff. Defaults to an 8K context with explicit KV capacity "
                "so 27B + --vision fits RTX 3090 24 GB. Prefer this over sparse frame grids."
            ),
            is_input_list=True,
            inputs=[
                ninfer_model_input(),
                io.Video.Input(
                    "video",
                    tooltip="Source VIDEO (LoadVideo / CreateVideo). Sent as a native video_url. A plate longer than this build's 16-item vision budget is sampled across its full length instead of rejected.",
                ),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input(
                    "system_prompt", default=_VIDEO_SYSTEM_PROMPT, multiline=True, advanced=True,
                    tooltip="Video rewrite instructions. Override for a specific edit dialect or target style.",
                ),
                io.Float.Input(
                    "start_at_timestamp", default=0.0, min=0.0, step=0.1,
                    tooltip="Seconds to skip before the clip sent to ninfer. 0 keeps the start.",
                ),
                io.Float.Input(
                    "duration_seconds", default=0.0, min=0.0, step=0.1,
                    tooltip=(
                        "Clip length after the start offset. 0 = full input (remainder). "
                        "When VIDEO already comes from PrepRefVideo (Duration wired there), "
                        "leave this at 0 — do not re-wire H3J Duration here or you double-trim."
                    ),
                ),
                io.Int.Input(
                    "seconds",
                    default=0,
                    min=0,
                    max=600,
                    optional=True,
                    tooltip=(
                        "Timeline length hint in whole seconds. 0 reads it from duration_seconds, "
                        "or from the video itself. Used for export length and logging only; "
                        "the rewrite timeline itself follows ref.md shots, not fixed beats."
                    ),
                ),
                io.Int.Input("max_tokens", default=4096, min=1, max=1048576, tooltip="Maximum generated tokens, including thinking when enabled."),
                io.Int.Input(
                    "context_length", default=VIDEO_SAFE_CONTEXT, min=1, max=1048576, step=1024,
                    tooltip=(
                        f"Prompt + media + output token capacity. Default {VIDEO_SAFE_CONTEXT} fits "
                        "RTX 3090 24GB + 27B + native --vision (auto's 1024 MiB KV headroom does not). "
                        "Raise only if VRAM is free; pair with Settings kv_capacity=match_context."
                    ),
                ),
                io.Int.Input(
                    "image_max_side", default=0, min=0, max=32768,
                    tooltip="Downscale optional still/image_list frames only. Native video uses video_max_side. 0 = automatic for stills.",
                ),
                io.Int.Input(
                    "video_max_side", default=0, min=0, max=2048,
                    tooltip="Downscale native VIDEO longest edge (even dimensions) before sending. 0 = no resize. Try 512-768 if a large plate exhausts context.",
                ),
                SettingsType.Input("settings", optional=True),
                io.Image.Input(
                    "image_list", optional=True,
                    tooltip=(
                        "Identity stills (IMAGE batch / OUTPUT_IS_LIST). When connected, they share "
                        "the same request as identity refs. First still is <Picture 1>. "
                        "The video is <Video 1>, not extra Pictures."
                    ),
                ),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image"), prefix="image", min=0, max=100,
                ), tooltip="Optional stills. Prefer native video for the plate; use stills for identity refs."),
            ],
            outputs=[
                io.String.Output(display_name="text", tooltip="Generated text (parity with Ninferencing)."),
                io.String.Output(
                    display_name="enhanced_video",
                    tooltip="Same text, named for H3J_STRUCT / image-pass handoff (H3J Enhanced Video).",
                ),
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
        return signature

    @classmethod
    def execute(
        cls, model, video, prompt, system_prompt, start_at_timestamp, duration_seconds,
        max_tokens, context_length, image_max_side, settings=None, images=None, image_list=None,
        seconds=0, video_max_side=0, **kwargs,
    ):
        video_obj = first(video)
        start = first(start_at_timestamp)
        requested = first(duration_seconds)
        clip_seconds = resolve_clip_seconds(
            video_obj, seconds=first(seconds), duration_seconds=requested, start_seconds=start,
        )
        has_still = any(
            frame is not None
            for frame in image_frames(list(collect_image_sources(images, image_list).values()))
        )
        if has_still:
            LOGGER.info(
                "Ninferencing Video: video swap, timeline %ds (ends %s). "
                "Video is <Video 1>, first still is <Picture 1>; shots follow ref.md.",
                clip_seconds,
                f"{clip_seconds // 60:02d}:{clip_seconds % 60:02d}.000" if clip_seconds else "last frame",
            )
        export_duration = requested
        if float(requested or 0.0) <= 0.0 and clip_seconds > 0:
            if has_still or int(first(seconds) or 0) > 0:
                export_duration = float(clip_seconds)
        text = run_video_ninfer(
            model=model, video=video, prompt=prompt, system_prompt=system_prompt,
            start_at_timestamp=start_at_timestamp, duration_seconds=export_duration,
            max_tokens=max_tokens, context_length=context_length, image_max_side=image_max_side,
            settings=settings, images=images, image_list=image_list,
            video_max_side=video_max_side,
        )
        return io.NodeOutput(text, text)




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
                        "Ninferencing Video treats Settings auto as match_context. On capacity shortfall, "
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
                    io.DynamicCombo.Option("model defaults", []),
                    io.DynamicCombo.Option("custom", [
                        io.Float.Input("temperature", default=0.6, min=0, max=10, step=0.01),
                        io.Float.Input("top_p", default=0.9, min=0, max=1, step=0.01),
                        io.Int.Input("top_k", default=40, min=0, max=1000000),
                        io.Float.Input("min_p", default=0, min=0, max=1, step=0.01),
                        io.Float.Input("presence_penalty", default=0, min=-10, max=10, step=0.01),
                        io.Float.Input("frequency_penalty", default=0, min=-10, max=10, step=0.01),
                    ]),
                ]),
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
