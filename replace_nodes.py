"""Oneshot H3 replacement analysis: a still view and a plate view, kept apart."""
from comfy_api.latest import io

from .nodes import first, ninfer_model_input, run_ninfer, run_video_ninfer
from .replace_brief import (
    IMAGE_ANALYSIS_SYSTEM,
    IMAGE_ANALYSIS_USER,
    VIDEO_ANALYSIS_SYSTEM,
    video_analysis_user_prompt,
)
from .runtime import IMAGE_MID_CONTEXT, VIDEO_SAFE_CONTEXT


def _text(text):
    return io.NodeOutput(text)


def _one_still(image):
    image = first(image)
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[0]) != 1 or int(shape[-1]) < 3:
        raise ValueError(
            "Ninferencing Image Analysis expects exactly one still. "
            "Connect the replacement image, not the source plate."
        )
    return image


class NinferencingImageAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NinferencingImageAnalysis",
            display_name="Ninferencing Image Analysis",
            category="Ninferencing",
            description=(
                "Describes one replacement still for an H3 oneshot swap. "
                "Does not see the source video and does not write the H3 prompt."
            ),
            is_input_list=True,
            inputs=[
                ninfer_model_input(),
                io.Image.Input(
                    "image",
                    tooltip="Replacement still only. Do not connect plate frames.",
                ),
                io.Int.Input("max_tokens", default=2048, min=1, max=1048576),
                io.Int.Input(
                    "context_length",
                    default=IMAGE_MID_CONTEXT,
                    min=1,
                    max=1048576,
                    step=1024,
                    tooltip="Prompt plus one still. Pair with Settings kv_capacity=match_context on a 24 GB card.",
                ),
                io.Int.Input(
                    "image_max_side",
                    default=1024,
                    min=0,
                    max=32768,
                    tooltip="Longest-side cap for the still. 0 uses the largest side this frame count allows.",
                ),
                io.Custom("NINFERENCING_SETTINGS").Input("settings", optional=True),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def fingerprint_inputs(cls, model, settings=None, **kwargs):
        from .nodes import Ninferencing
        return Ninferencing.fingerprint_inputs(model, settings=settings, **kwargs)

    @classmethod
    def execute(cls, model, image, max_tokens, context_length, image_max_side, settings=None):
        text = run_ninfer(
            model=model,
            prompt=IMAGE_ANALYSIS_USER,
            system_prompt=IMAGE_ANALYSIS_SYSTEM,
            max_tokens=max_tokens,
            context_length=context_length,
            image_max_side=image_max_side,
            settings=settings,
            images={"image": _one_still(image)},
        )
        return _text(text)


class NinferencingVideoAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NinferencingVideoAnalysis",
            display_name="Ninferencing Video Analysis",
            category="Ninferencing",
            description=(
                "Observes one source video for an H3 oneshot swap. "
                "Does not see the replacement still and does not write the H3 prompt."
            ),
            is_input_list=True,
            inputs=[
                ninfer_model_input(),
                io.Video.Input(
                    "video",
                    tooltip="Prepared source plate. Leave duration at 0 when Reference Video already trimmed it.",
                ),
                io.String.Input(
                    "target",
                    default="the main performer",
                    multiline=True,
                    tooltip="Who or what to track in the plate. This node does not see the replacement still.",
                ),
                io.Float.Input(
                    "start_at_timestamp",
                    default=0.0,
                    min=0.0,
                    step=0.1,
                    tooltip="Seconds to skip. 0 keeps the start. Leave at 0 when the video is already windowed.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.1,
                    tooltip="0 uses the full input. Do not trim again when Reference Video already owns the window.",
                ),
                io.Int.Input("max_tokens", default=4096, min=1, max=1048576),
                io.Int.Input(
                    "context_length",
                    default=VIDEO_SAFE_CONTEXT,
                    min=1,
                    max=1048576,
                    step=1024,
                    tooltip="Native video budget. 8192 with explicit KV capacity fits 27B vision on 24 GB.",
                ),
                io.Int.Input(
                    "video_max_side",
                    default=768,
                    min=0,
                    max=2048,
                    step=32,
                    tooltip=(
                        "Downscale the plate's longest edge before sending (even dimensions). "
                        "768 matches a typical H3 encode size and keeps vision context small; "
                        "0 sends the plate as-is. Analysis only needs enough detail to read the "
                        "scene, camera, and contacts."
                    ),
                ),
                io.Custom("NINFERENCING_SETTINGS").Input("settings", optional=True),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def fingerprint_inputs(cls, model, settings=None, **kwargs):
        from .nodes import NinferencingVideo
        return NinferencingVideo.fingerprint_inputs(model, settings=settings, **kwargs)

    @classmethod
    def execute(
        cls, model, video, target, start_at_timestamp, duration_seconds,
        max_tokens, context_length, video_max_side=768, settings=None,
    ):
        text = run_video_ninfer(
            model=model,
            video=video,
            prompt=video_analysis_user_prompt(first(target)),
            system_prompt=VIDEO_ANALYSIS_SYSTEM,
            start_at_timestamp=start_at_timestamp,
            duration_seconds=duration_seconds,
            max_tokens=max_tokens,
            context_length=context_length,
            image_max_side=0,
            video_max_side=video_max_side,
            settings=settings,
        )
        return _text(text)
