from comfy_api.latest import ComfyExtension

from .models import register_models

__version__ = "1.0.0"
from .nodes import Ninferencing, NinferencingSettings

register_models()


class NinferencingExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            Ninferencing,
            NinferencingSettings,
        ]


async def comfy_entrypoint():
    return NinferencingExtension()
