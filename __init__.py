from comfy_api.latest import ComfyExtension

from .models import register_models

__version__ = "1.0.0"
from .nodes import Ninferencing, NinferencingSettings
from .swap_nodes import NinferencingH3AttachmentMap, NinferencingH3SwapSystem
from .replace_nodes import (
    NinferencingImageAnalysis,
    NinferencingVideoAnalysis,
)

register_models()


class NinferencingExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            Ninferencing,
            NinferencingSettings,
            NinferencingImageAnalysis,
            NinferencingVideoAnalysis,
            NinferencingH3SwapSystem,
            NinferencingH3AttachmentMap,
        ]


async def comfy_entrypoint():
    return NinferencingExtension()
