import os
from pathlib import Path
import sys


COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# All tests, including the real subprocess tests, are CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from comfy.cli_args import args

args.cpu = True
