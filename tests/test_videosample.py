"""Streamed sampling: frame budget, size cap, and plate-length preservation."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from Ninferencing import videosample
from Ninferencing.videosample import build_clip, lock_size, probe_video, sample_video


def _write_clip(path, *, frames=40, fps=24, width=128, height=96):
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(fps, 1))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for index in range(frames):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (index * 5) % 256
        container.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
    container.mux(stream.encode())
    container.close()
    return Path(path)


def _measure(path):
    """Decoded frame count and container duration for an encoded file."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video=0))
        duration = float(stream.duration * stream.time_base)
        size = (stream.codec_context.width, stream.codec_context.height)
    return frames, duration, size


def test_probe_reads_container_metadata_without_decoding(tmp_path):
    source = _write_clip(tmp_path / "source.mp4", frames=40, fps=24)
    info = probe_video(source)
    assert info is not None
    assert info.frame_count == 40
    assert info.rate == Fraction(24, 1)
    assert info.duration == pytest.approx(40 / 24, abs=0.05)
    assert (info.width, info.height) == (128, 96)


def test_probe_refuses_non_video_input(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("not a clip")
    assert probe_video(text) is None


def test_clip_inside_the_budget_is_not_reencoded(tmp_path):
    source = _write_clip(tmp_path / "short.mp4", frames=8, fps=24)
    info = probe_video(source)
    # 8 frames within a 16-frame budget and no size cap → send the file as-is.
    assert build_clip(info, tmp_path / "out.mp4", max_frames=16, max_side=0) is None


def test_long_plate_is_thinned_and_keeps_its_wall_clock_length(tmp_path):
    source = _write_clip(tmp_path / "plate.mp4", frames=192, fps=24)  # 8 s
    info = probe_video(source)
    out = build_clip(info, tmp_path / "sampled.mp4", max_frames=16, max_side=0)
    assert out is not None
    frames, duration, size = _measure(out)
    assert frames == 16
    # The analysis timeline must still map 1:1 onto the plate: 8 s in, 8 s out.
    assert duration == pytest.approx(8.0, abs=0.1)
    assert size == (128, 96)


def test_sampling_spreads_over_the_whole_clip(tmp_path):
    source = _write_clip(tmp_path / "plate.mp4", frames=40, fps=24)
    info = probe_video(source)
    out = build_clip(info, tmp_path / "sampled.mp4", max_frames=15, max_side=0)
    frames, duration, _ = _measure(out)
    assert frames == 15
    assert duration == pytest.approx(40 / 24, abs=0.1)


def test_size_cap_resizes_during_decode(tmp_path):
    source = _write_clip(tmp_path / "hd.mp4", frames=4, fps=24, width=1920, height=1080)
    info = probe_video(source)
    out = build_clip(info, tmp_path / "small.mp4", max_frames=16, max_side=640)
    frames, duration, size = _measure(out)
    assert frames == 4
    assert size == (640, 360)  # down-only, even, 16:9 kept
    assert duration == pytest.approx(4 / 24, abs=0.1)


def test_sample_and_resize_together(tmp_path):
    source = _write_clip(tmp_path / "plate.mp4", frames=48, fps=24, width=1920, height=1080)
    info = probe_video(source)
    out = build_clip(info, tmp_path / "both.mp4", max_frames=8, max_side=512)
    frames, duration, size = _measure(out)
    assert frames == 8
    assert size == (512, 288)
    assert duration == pytest.approx(48 / 24, abs=0.1)


def test_lock_size_never_upscales():
    assert lock_size(640, 360, 768, 2) == (640, 360)
    assert lock_size(1920, 1080, 768, 2) == (768, 432)
    assert lock_size(128, 96, 0, 2) == (128, 96)


def test_empty_keep_set_returns_none(tmp_path):
    source = _write_clip(tmp_path / "source.mp4", frames=4, fps=24)
    info = probe_video(source)
    assert sample_video(info, tmp_path / "never.mp4", keep=[]) is None
    assert not (tmp_path / "never.mp4").exists()


def test_ninfer_video_plan_scales_sample_count_with_fps():
    frames_2, *_ = videosample.ninfer_video_plan(512, 288, 192, 24, fps=2)
    frames_6, *_ = videosample.ninfer_video_plan(512, 288, 192, 24, fps=6)
    assert frames_2 == 16
    assert frames_6 == 48


def test_samplers_share_the_nodes_entry_point():
    from Ninferencing import nodes

    assert nodes.probe_video is probe_video
    assert nodes.build_clip is build_clip
