"""Wave 1 H3 swap nodes: pack-owned system dialect and attachment map."""
from comfy_api.latest import io

from .replace_brief import build_attachment_map
from .swap_dialects import DIALECTS, swap_system_prompt


class NinferencingH3SwapSystem(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NinferencingH3SwapSystem",
            display_name="Ninferencing H3 Swap System",
            category="Ninferencing",
            description=(
                "Pack-owned H3 swap system prompt (locked or delta). Wire into Ninferencing `system_prompt`. "
                "Improves on: external markdown files that can go missing; this text ships with the pack "
                "and reads a plain user prompt plus the attachment map. Locks intent, not pixels."
            ),
            inputs=[
                io.Combo.Input(
                    "dialect",
                    options=DIALECTS,
                    default="locked",
                    tooltip="locked: only the swap, plate otherwise unchanged. delta: swap plus changes the user names (motion, look, speech).",
                ),
            ],
            outputs=[io.String.Output(display_name="system_prompt", tooltip="Wire into Ninferencing `system_prompt`.")],
        )

    @classmethod
    def execute(cls, dialect="locked"):
        return io.NodeOutput(swap_system_prompt(dialect))


class NinferencingH3AttachmentMap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NinferencingH3AttachmentMap",
            display_name="Ninferencing H3 Attachment Map",
            category="Ninferencing",
            description=(
                "Deterministic ATTACHMENTS block for the H3 swap prompt: stills are <Picture N>, the plate is <Video 1>, "
                "sampled frames are observations, not Pictures. Improves on: freeform media descriptions that let the "
                "model invent extra Pictures. Put it before your plain request in Ninferencing `prompt`."
            ),
            inputs=[
                io.Int.Input("picture_count", default=1, min=1, max=16,
                             tooltip="Stills Ninferencing sends (refs + image_list + images), in order."),
                io.Boolean.Input("has_video", default=True, tooltip="Plate connected to Ninferencing `video`. Required for a swap."),
                io.Float.Input("duration_s", default=0.0, min=0.0, step=0.1, optional=True,
                               tooltip="Plate length in seconds for the map. 0 omits it."),
                io.String.Input("note", default="", multiline=True, optional=True,
                                tooltip="Optional one-line note. Media labels are stripped."),
            ],
            outputs=[io.String.Output(display_name="attachment_map")],
        )

    @classmethod
    def execute(cls, picture_count=1, has_video=True, duration_s=0.0, note=""):
        text = build_attachment_map(picture_count, has_video, duration_s=duration_s or None, note=note or "")
        return io.NodeOutput(text)
