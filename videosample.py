"""Probe and stream-encode the video ninfer actually receives.

The old path pulled a whole plate into a tensor batch (``get_components``)
just to decide whether to drop frames, so peak memory grew with the clip.
This module reads container metadata instead and decodes only the frames it
keeps: peak memory is one frame plus at most ``max_frames`` outputs.

Sampling keeps the plate's wall-clock length. ``n`` frames taken from an
``s`` second clip are written at ``n/s`` fps, so an 8 s plate sampled to 16
frames plays as 8 s, not as 8 s of 2 fps stretched to whatever. A character
or object swap analysis asks for ``[MM:SS.mmm]`` coverage of the source
plate, so those stamps must still map 1:1 onto the original timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def sample_video(info: VideoInfo, destination, *, keep, target_size=None) -> Path | None:
    """Write ``keep`` (ascending frame indices) from ``info.path`` to mp4.

    ``target_size`` is (width, height); when given the kept frames are resized
    during decode. The output rate is chosen so the clip plays as long as the
    source window. Returns None when nothing could be kept — the caller then
    sends the unmodified file instead of an empty clip.
    """
    import av
    import numpy as np

    keep = sorted({int(index) for index in keep if int(index) >= 0})
    if not keep:
        return None
    images = []
    with av.open(str(info.path)) as container:
        stream = container.streams.video[0]
        pointer = 0
        last = keep[-1]
        for index, frame in enumerate(container.decode(video=0)):
            while pointer < len(keep) and keep[pointer] < index:
                pointer += 1
            if pointer >= len(keep) or index > last:
                break
            if index != keep[pointer]:
                continue
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            width, height = _even(image.width), _even(image.height)
            if target_size is not None:
                wanted = (_even(target_size[0]), _even(target_size[1]))
                if image.size != wanted:
                    image = image.resize(wanted, Image.Resampling.LANCZOS)
            elif image.size != (width, height):
                image = image.crop((0, 0, width, height))
            images.append(image)
            pointer += 1
    if not images:
        return None

    duration = Fraction(info.duration).limit_denominator(1_000_000)
    out_rate = Fraction(len(images)) / duration
    if out_rate <= 0:
        out_rate = info.rate
    # Never faster than the source, and never slower than one frame per 4 s:
    # below that the encoder's own defaults start inserting duplicate frames.
    out_rate = min(out_rate, info.rate)
    out_rate = max(out_rate, Fraction(1, 4))

    destination = Path(destination)
    width, height = _even(images[0].width), _even(images[0].height)
    with av.open(str(destination), mode="w") as output:
        out_stream = output.add_stream("libx264", rate=out_rate)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        for image in images:
            array = np.asarray(image)
            if array.shape[1] != width or array.shape[0] != height:
                array = np.asarray(
                    Image.fromarray(array).crop((0, 0, width, height))
                )
            packet_source = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in out_stream.encode(packet_source):
                output.mux(packet)
        for packet in out_stream.encode():
            output.mux(packet)
    return destination


def build_clip(
    info: VideoInfo,
    destination,
    *,
    max_frames: int,
    max_side: int = 0,
) -> Path | None:
    """Plan and write the clip ninfer should see, or None when it is fine as-is.

    ``None`` means "no re-encode needed": the source already fits the frame
    budget and (when set) the size cap.
    """
    if info is None:
        return None
    needs_sample = info.frame_count > max_frames
    target_size = None
    if max_side > 0 and max(info.width, info.height) > max_side:
        target_w, target_h = lock_size(info.width, info.height, max_side, 2)
        if (target_w, target_h) != (info.width, info.height):
            target_size = (target_w, target_h)
    if not needs_sample and target_size is None:
        return None
    keep = list(range(info.frame_count)) if not needs_sample else _spread(info.frame_count, max_frames)
    return sample_video(info, destination, keep=keep, target_size=target_size)


def _spread(count: int, budget: int):
    """Evenly cover ``count`` items with at most ``budget`` picks.

    Mirrors nodes.spread_indices; kept local so the module has no import
    cycle back into the node file.
    """
    count = int(count)
    budget = max(1, int(budget))
    if count <= budget:
        return list(range(max(count, 0)))
    if budget == 1:
        return [count // 2]
    return [round(i * (count - 1) / (budget - 1)) for i in range(budget)]
