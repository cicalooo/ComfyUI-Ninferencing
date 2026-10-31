"""Probe and stream-encode the video ninfer actually receives.

Reads container metadata instead of decoding the whole clip. The full clip is
always sent whole (ninfer samples it itself at its baked 2 fps); this module
only downscales the longest edge when a size cap is set. Peak memory is one
frame plus the outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from fractions import Fraction
from pathlib import Path

from PIL import Image

# Container duration is in AV_TIME_BASE units (microseconds) on every PyAV
# release this pack supports; stream duration is time_base scaled.
_AV_TIME_BASE = 1_000_000


@dataclass(frozen=True)
class VideoInfo:
    """Container facts needed to plan a sample, without decoding frames."""

    path: Path
    width: int
    height: int
    rate: Fraction  # frames per second as declared by the container
    frame_count: int
    duration: float  # seconds


def lock_size(width, height, max_side, multiple):
    """Down-only longest edge snapped to ``multiple`` (never upscales).

    Moved from plate.py: video export needs even dimensions for yuv420p and
    ninfer's vision grid likes predictable edges.
    """
    multiple = max(1, int(multiple))
    width = max(1, int(width))
    height = max(1, int(height))
    cap = int(max_side)
    scale = min(1.0, cap / max(width, height)) if cap > 0 else 1.0

    def snap(value, original):
        snapped = max(multiple, int(value / multiple) * multiple)
        if snapped <= original:
            return snapped
        floor = (original // multiple) * multiple
        return floor if floor >= multiple else original

    return snap(width * scale, width), snap(height * scale, height)


def _stream_seconds(stream, container) -> float:
    try:
        if stream.duration is not None and stream.time_base:
            value = float(stream.duration * stream.time_base)
            if value > 0:
                return value
    except Exception:
        pass
    try:
        if container.duration:
            return float(container.duration) / _AV_TIME_BASE
    except Exception:
        pass
    return 0.0


def probe_video(path) -> VideoInfo | None:
    """Container metadata for an existing file, or None when unusable.

    Never raises: an unreadable file just means the caller sends it as-is and
    lets the CLI report what it could not parse.
    """
    try:
        import av
    except Exception:
        return None
    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if not streams:
                return None
            stream = streams[0]
            context = stream.codec_context
            width = int(getattr(context, "width", 0) or stream.width or 0)
            height = int(getattr(context, "height", 0) or stream.height or 0)
            rate = stream.average_rate or stream.base_rate or Fraction(24, 1)
            rate = Fraction(rate)
            if rate <= 0:
                rate = Fraction(24, 1)
            duration = _stream_seconds(stream, container)
            frame_count = int(stream.frames or 0)
            if frame_count < 1 and duration > 0:
                frame_count = int(round(duration * float(rate)))
            if frame_count < 1 or width < 2 or height < 2 or duration <= 0:
                return None
            return VideoInfo(
                path=Path(path),
                width=width,
                height=height,
                rate=rate.limit_denominator(1_000_000),
                frame_count=frame_count,
                duration=duration,
            )
    except Exception:
        return None


def _even(value: int) -> int:
    value = int(value)
    return max(2, value - (value % 2))


def sample_video(info: VideoInfo, destination, *, target_size=None) -> Path | None:
    """Re-encode every frame from ``info.path`` to mp4.

    ``target_size`` is (width, height); when given the frames are resized
    during decode. The output rate keeps the clip's wall-clock length so
    timestamps stay wall-clock true. Returns None when nothing could be kept —
    the caller then sends the unmodified file instead of an empty clip.
    """
    import av
    import numpy as np

    # Output rate is fixed up front so the clip plays as long as the source.
    # Frames are encoded as they are decoded: peak memory is one frame.
    duration = Fraction(info.duration).limit_denominator(1_000_000)
    out_rate = info.rate
    # Never slower than one frame per 4 s:
    # below that the encoder's own defaults start inserting duplicate frames.
    out_rate = max(out_rate, Fraction(1, 4)).limit_denominator(1_000_000)

    destination = Path(destination)
    output = None
    out_stream = None
    width = height = 0
    written = 0
    try:
        with av.open(str(info.path)) as container:
            for frame in container.decode(video=0):
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                if target_size is not None:
                    wanted = (_even(target_size[0]), _even(target_size[1]))
                    if image.size != wanted:
                        image = image.resize(wanted, Image.Resampling.BILINEAR)
                if output is None:
                    width, height = _even(image.width), _even(image.height)
                    output = av.open(str(destination), mode="w")
                    out_stream = output.add_stream("libx264", rate=out_rate)
                    out_stream.width = width
                    out_stream.height = height
                    out_stream.pix_fmt = "yuv420p"
                    # Throwaway carrier: ninfer re-samples this clip at its baked
                    # 2 fps before reading pixels, so encode fast, not pretty.
                    out_stream.options = {"preset": "superfast", "crf": "20"}
                if image.size != (width, height):
                    image = image.crop((0, 0, width, height))
                packet_source = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                for packet in out_stream.encode(packet_source):
                    output.mux(packet)
                written += 1
        if output is None:
            return None
        for packet in out_stream.encode():
            output.mux(packet)
    finally:
        if output is not None:
            output.close()
    if written == 0:
        return None
    return destination


def build_clip(
    info: VideoInfo,
    destination,
    *,
    max_side: int = 0,
) -> Path | None:
    """Plan and write the clip ninfer should see, or None when it is fine as-is.

    ``None`` means "no re-encode needed": the source already fits the size cap.
    The clip is always sent whole — never trimmed or capped.
    """
    if info is None:
        return None
    target_size = None
    if max_side > 0 and max(info.width, info.height) > max_side:
        target_w, target_h = lock_size(info.width, info.height, max_side, 2)
        if (target_w, target_h) != (info.width, info.height):
            target_size = (target_w, target_h)
    if target_size is None:
        return None
    return sample_video(info, destination, target_size=target_size)


# ninfer video frontend constants (src/models/qwen3_5/frontend/frontend.cpp:52-54,
# processor.cpp:42-45, decode.cpp:459). A native clip is always sampled at the
# baked 2 fps; there is no per-request fps flag.
NINFER_VIDEO_FPS = 2.0
NINFER_VIDEO_MIN_FRAMES = 4
NINFER_VIDEO_MAX_FRAMES = 768
_FACTOR = 32  # patch 16 x merge 2
_TEMPORAL = 2


def _round_even_half(value: float) -> int:
    return int(round(value))  # Python round is half-to-even, same as the C++ round_even helper


def ninfer_video_plan(width, height, frame_count, rate, max_pixels=8192 * _FACTOR * _FACTOR * _TEMPORAL,
                      min_pixels=128 * 32 * 32, fps=NINFER_VIDEO_FPS):
    """Mirror of ninfer's native video sampling + smart_resize_video for an exported clip.

    Returns (sampled_frames, grid_w, grid_h, vision_tokens, timestamps). max_pixels defaults to
    the bound --vision-max-merged 8192 puts on one video item (8192 merged tokens).
    """
    rate = float(rate)
    total = int(frame_count)
    target = float(fps) if fps else NINFER_VIDEO_FPS
    if total < 1 or rate <= 0 or target <= 0:
        return 0, 0, 0, 0, []
    count = int(total / rate * target)
    count = min(max(count, NINFER_VIDEO_MIN_FRAMES), NINFER_VIDEO_MAX_FRAMES, total)
    if count == 1:
        indices = [0]
    else:
        indices = [int(round(i * (total - 1) / (count - 1))) for i in range(count)]
    h = _round_even_half(height / _FACTOR) * _FACTOR
    w = _round_even_half(width / _FACTOR) * _FACTOR
    padded = ((count + _TEMPORAL - 1) // _TEMPORAL) * _TEMPORAL
    volume = padded * h * w
    if volume > max_pixels:
        beta = (count * height * width / max_pixels) ** 0.5
        h = max(_FACTOR, int(height / beta // _FACTOR) * _FACTOR)
        w = max(_FACTOR, int(width / beta // _FACTOR) * _FACTOR)
    elif volume < min_pixels:
        beta = (min_pixels / (count * height * width)) ** 0.5
        h = math.ceil(height * beta / _FACTOR) * _FACTOR
        w = math.ceil(width * beta / _FACTOR) * _FACTOR
    groups = padded // _TEMPORAL
    tokens = groups * (h // _FACTOR) * (w // _FACTOR)
    pad = indices + ([indices[-1]] if len(indices) % _TEMPORAL else [])
    stamps = [(pad[2 * t] + pad[2 * t + 1]) / (2.0 * rate) for t in range(groups)]
    return count, w, h, tokens, stamps



# Gemma 4 A4B vision (processor.h GemmaMediaOptions; docs/gemma4.md). Native
# video is up to 32 frames spaced evenly over the clip (FrameSpacing::Floor),
# not the Qwen baked-2-fps temporal-pair path. Soft tokens per frame stay on
# the 70/140/280/560/1120 ladder; video defaults to 70 each.
GEMMA_VIDEO_FRAMES = 32
GEMMA_VIDEO_FRAME_TOKENS = 70
GEMMA_IMAGE_TOKENS = 280
GEMMA_SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def gemma_video_plan(frame_count, rate, *, frames=GEMMA_VIDEO_FRAMES,
                     tokens_per_frame=GEMMA_VIDEO_FRAME_TOKENS):
    """Gemma 4 native video plan. Tuple shape matches ``ninfer_video_plan``.

    Returns (sampled_frames, 0, 0, vision_tokens, timestamps_seconds). Width/height
    are unused: the processor fits each frame into the soft-token budget itself.
    """
    total = int(frame_count)
    if total < 1:
        return 0, 0, 0, 0, []
    count = min(max(1, int(frames)), total)
    if count == 1:
        indices = [0]
    else:
        # Floor spacing: i * total / count (matches media::decode FrameSpacing::Floor).
        indices = [int(i * total / count) for i in range(count)]
    rate = float(rate) if rate and float(rate) > 0 else 1.0
    stamps = [index / rate for index in indices]
    tokens = count * int(tokens_per_frame)
    return count, 0, 0, tokens, stamps


def gemma_still_tokens(image_count, tokens_per_image=GEMMA_IMAGE_TOKENS):
    """Default soft-token cost of ``image_count`` Gemma stills at auto detail."""
    return max(0, int(image_count)) * int(tokens_per_image)


def fit_side_for_tokens(width, height, frame_count, rate, max_side, token_budget, fps=NINFER_VIDEO_FPS):
    """Largest even max_side <= max_side whose ninfer video plan stays within token_budget."""
    side = int(max_side)
    while side > 64:
        tw, th = lock_size(width, height, side, 2) if max(width, height) > side else (width, height)
        if ninfer_video_plan(tw, th, frame_count, rate, fps=fps)[3] <= token_budget:
            return side
        side -= 32
    return 64
