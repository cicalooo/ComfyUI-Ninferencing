import torch

from Ninferencing import nodes
from Ninferencing.replace_brief import (
    IMAGE_ANALYSIS_SYSTEM,
    IMAGE_ANALYSIS_USER,
    VIDEO_ANALYSIS_SYSTEM,
    sanitize_analysis,
    video_analysis_user_prompt,
)
from Ninferencing.replace_nodes import (
    NinferencingImageAnalysis,
    NinferencingVideoAnalysis,
)
from Ninferencing.runtime import IMAGE_MID_CONTEXT, VIDEO_SAFE_CONTEXT


def test_clip_seconds_prefers_int_then_duration_then_video():
    from Ninferencing.nodes import resolve_clip_seconds

    class Clip:
        def get_duration(self):
            return 14.6

    assert resolve_clip_seconds(Clip(), seconds=9, duration_seconds=15) == 9
    assert resolve_clip_seconds(Clip(), seconds=0, duration_seconds=15.2) == 15
    assert resolve_clip_seconds(Clip(), seconds=0, duration_seconds=0, start_seconds=2) == 13


def test_analysis_schemas_keep_the_two_views_separate():
    image = NinferencingImageAnalysis.define_schema()
    video = NinferencingVideoAnalysis.define_schema()
    for schema in (image, video):
        schema.finalize()
        schema.validate()
    image_ids = {item.id for item in image.inputs}
    video_ids = {item.id for item in video.inputs}
    assert "image" in image_ids and "video" not in image_ids
    assert "video" in video_ids and "image" not in video_ids and "images" not in video_ids
    assert {item.id: item for item in image.inputs}["context_length"].default == IMAGE_MID_CONTEXT
    assert {item.id: item for item in video.inputs}["context_length"].default == VIDEO_SAFE_CONTEXT
    assert {item.id: item for item in video.inputs}["duration_seconds"].default == 0.0
    assert IMAGE_ANALYSIS_SYSTEM.startswith("You describe one attached still")
    assert "Do not invent a replacement appearance" in VIDEO_ANALYSIS_SYSTEM


def test_no_node_is_an_output_node_or_carries_a_preview():
    """Text leaves through sockets only, and nothing runs without a consumer."""
    for node in (
        nodes.Ninferencing,
        nodes.NinferencingVideo,
        nodes.NinferencingSettings,
        NinferencingImageAnalysis,
        NinferencingVideoAnalysis,
    ):
        schema = node.define_schema()
        assert not schema.is_output_node, node.__name__


def test_video_analysis_sends_a_downscaled_plate_by_default():
    schema = NinferencingVideoAnalysis.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["video_max_side"].default == 768
    assert inputs["video_max_side"].min == 0


def test_image_analysis_sends_one_still_and_does_not_rewrite_a_video_prompt(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "blue coat"

    monkeypatch.setattr("Ninferencing.replace_nodes.run_ninfer", fake)
    image = torch.zeros(1, 8, 8, 3)
    output = NinferencingImageAnalysis.execute(
        {"model": "library", "name": "x.ninfer"}, image, 32, 1024, 64,
    )
    assert seen["system_prompt"] == IMAGE_ANALYSIS_SYSTEM
    assert seen["prompt"] == IMAGE_ANALYSIS_USER
    assert seen["images"]["image"].shape == (1, 8, 8, 3)
    assert output[0] == "blue coat"


def test_image_analysis_rejects_a_frame_batch():
    try:
        NinferencingImageAnalysis.execute(
            {"model": "library", "name": "x.ninfer"}, torch.zeros(4, 8, 8, 3), 32, 1024, 64,
        )
    except ValueError as error:
        assert "exactly one still" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_video_analysis_passes_the_size_cap_through(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "scene and timeline"

    monkeypatch.setattr("Ninferencing.replace_nodes.run_video_ninfer", fake)
    output = NinferencingVideoAnalysis.execute(
        {"model": "library", "name": "x.ninfer"}, "video", "the cup", 0.0, 0.0, 128, 8192, 768,
    )
    assert seen["video_max_side"] == 768
    assert seen["system_prompt"] == VIDEO_ANALYSIS_SYSTEM
    assert seen["prompt"] == "Target to replace: the cup"
    assert seen["image_max_side"] == 0
    assert output[0] == "scene and timeline"


def test_analysis_text_cannot_invent_extra_media_labels():
    assert sanitize_analysis("Face from <Picture 2>, coat from <Video 1>.") == (
        "Face from the attached still, coat from the source plate."
    )
    assert sanitize_analysis("<Subject 2> stands") == "the replacement stands"
    assert sanitize_analysis(None) == ""
    assert video_analysis_user_prompt("<Picture 3>") == "Target to replace: the attached still"
    assert video_analysis_user_prompt("") == "Target to replace: the named target"
