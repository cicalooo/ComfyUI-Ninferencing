import pytest

from Ninferencing.replace_brief import build_attachment_map, lint_swap_output
from Ninferencing.swap_nodes import NinferencingH3AttachmentMap

OBS = "Frames sampled from <Video 1> are observations of <Video 1>, not Pictures. No other media is attached."


def test_one_still_plus_video_golden():
    assert build_attachment_map(1, True) == "\n".join([
        "ATTACHMENTS:",
        "<Picture 1> = replacement still 1 of 1",
        "<Video 1> = source plate (native video)",
        OBS,
    ])


def test_two_stills_full_with_duration_and_note():
    assert build_attachment_map(2, True, duration_s=5.0, note="  same <Picture 3> outfit\n twice ") == "\n".join([
        "ATTACHMENTS:",
        "<Picture 1> = replacement still 1 of 2",
        "<Picture 2> = replacement still 2 of 2",
        "<Video 1> = source plate (native video), 5 s",
        OBS,
        "NOTE: same the attached still outfit twice",
    ])


def test_illegal_maps_raise():
    with pytest.raises(ValueError):
        build_attachment_map(0, True)  # video-only cannot swap
    with pytest.raises(ValueError):
        build_attachment_map(1, False)


def test_node():
    schema = NinferencingH3AttachmentMap.define_schema()
    schema.finalize()
    schema.validate()
    assert NinferencingH3AttachmentMap.execute(1, True, 0.0, "")[0] == build_attachment_map(1, True)


def test_lint_swap_output():
    good = "The Subject from <Picture 1> replaces the performer in <Video 1>; keep the source audio fully_copy."
    assert lint_swap_output(good, 1, True) == []
    assert lint_swap_output(good, 1, True, audio_muxed_later=False) == []
    issues = lint_swap_output(good, 1, True, audio_muxed_later=True)
    assert any("fully_copy" in i for i in issues)
    assert any("+AUDIO" in i for i in lint_swap_output("keep +AUDIO", 1, True, audio_muxed_later=True))
    assert any("Picture 3" in i for i in lint_swap_output("<Picture 3> hat", 2, True))
    assert any("Video 2" in i for i in lint_swap_output("match Video 2 framing", 1, True))
    assert any("no video" in i for i in lint_swap_output("<Video 1>", 1, False))
