import os
import re
from pathlib import Path
import shutil
import struct

import folder_paths


# Linux prefixes checked for a `ninfer` binary, first hit wins.
# /opt/ninfer is the ninfer-linux installer default (RTX 30/40).
_LINUX_NINFER_HOMES = (
    Path("/opt/ninfer"),
    Path("/opt/ninfer/current"),
    Path("/opt/ninfer-3090/current"),
    Path.home() / ".local" / "ninfer",
)


def default_executable():
    configured = os.environ.get("NINFER_EXECUTABLE", "").strip()
    if configured:
        return configured
    home = os.environ.get("NINFER_HOME", "").strip()
    if home:
        return home
    if os.name != "nt":
        for root in _LINUX_NINFER_HOMES:
            installed = root / "ninfer"
            if installed.is_file():
                return str(installed)
    found = shutil.which("ninfer")
    if found:
        return found
    if os.name == "nt" and Path(r"C:\ninfer\ninfer.exe").is_file():
        return r"C:\ninfer\ninfer.exe"
    return "ninfer"


def artifact_container_version(path):
    """NInfer container version from the file magic, or None if unreadable/unknown."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(8)
    except OSError:
        return None
    if len(header) < 8 or header[:6] != b"NINFER":
        return None
    return struct.unpack_from(">H", header, 6)[0]


def detect_artifact_family(path):
    """Best-effort family for node defaults (KV, spec, vision planning).

    Returns ``"gemma4"`` for Gemma 4 A4B artifacts, ``"qwen_moe"`` for the
    Qwen3.6-35B-A3B MoE (same Qwen frontend, but its own VRAM model and KV/MTP
    defaults), otherwise ``"qwen"`` (Qwen3.x / Bonsai / Flash-Next share the
    Qwen frontend path this pack already targets). Filename heuristics cover
    the WaveCut/Google names; a short ASCII scan of the artifact directory blob
    catches renames. Callers test ``family == "gemma4"``, so ``qwen_moe``
    follows every Qwen code path.
    """
    name = Path(path).name.lower()
    if re.search(r"gemma[-_]?4|gemma4", name):
        return "gemma4"
    moe_name = bool(re.search(r"35b[-_.]?a3b", name))
    try:
        with open(path, "rb") as handle:
            blob = handle.read(256 * 1024)
    except OSError:
        return "qwen_moe" if moe_name else "qwen"
    markers = (b"Gemma4ForCausalLM", b"gemma4_gguf", b"Gemma4ForConditionalGeneration")
    if any(marker in blob for marker in markers):
        return "gemma4"
    if moe_name or b"Qwen3_5MoeForCausalLM" in blob or b"qwen3_5_moe" in blob:
        return "qwen_moe"
    return "qwen"


def resolve_executable(value):
    value = os.path.expandvars(value.strip().strip('"') or default_executable())
    path = Path(value).expanduser()
    if path.is_dir():
        path /= "ninfer.exe" if os.name == "nt" else "ninfer"
    # ninfer-all ships one multicall binary; argv[0] must be "ninfer" or it treats
    # --help as a program name ("unknown program '--help'").
    if path.name.lower() in {"ninfer-serve", "ninfer-serve.exe", "ninfer-multicall", "ninfer-multicall.exe"}:
        path = path.with_name("ninfer.exe" if path.suffix.lower() == ".exe" else "ninfer")
    found = str(path) if path.is_file() else shutil.which(str(path))
    if not found:
        raise FileNotFoundError("Ninferencing: set executable to your ninfer CLI or its installation folder in Ninferencing Settings.")
    path = Path(found)
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise PermissionError(f"Ninferencing: ninfer CLI is not executable: {path}")
    resolved = path.resolve()
    if resolved.name.lower() in {"ninfer-multicall", "ninfer-multicall.exe"}:
        cli = resolved.with_name("ninfer.exe" if os.name == "nt" else "ninfer")
        if cli.exists():
            return cli
        raise FileNotFoundError("Ninferencing: ninfer-multicall needs a sibling ninfer symlink so the CLI dispatches on argv[0].")
    return resolved


def register_models():
    folder_paths.add_model_folder_path("ninfer", str(Path(folder_paths.models_dir) / "ninfer"))
    folder_paths.folder_names_and_paths["ninfer"][1].add(".ninfer")
    configured = os.environ.get("NINFER_MODELS_DIR", "").strip()
    if configured:
        folder_paths.add_model_folder_path("ninfer", str(Path(os.path.expandvars(configured)).expanduser()))
    home_models = Path.home() / "models" / "ninfer"
    folder_paths.add_model_folder_path("ninfer", str(home_models))
    try:
        home = resolve_executable("").parent
    except (FileNotFoundError, PermissionError):
        return
    for directory in (home / "models", home / "artifacts"):
        if directory.is_dir():
            folder_paths.add_model_folder_path("ninfer", str(directory))


def resolve_model(selection):
    if selection["model"] == "file path":
        path = Path(os.path.expandvars(selection["path"].strip().strip('"'))).expanduser().resolve()
    elif selection["model"] == "library":
        name = selection["name"]
        if Path(name).is_absolute() or ".." in name.replace("\\", "/").split("/"):
            raise ValueError("Ninferencing: select a model from the library, or use file path.")
        path = Path(folder_paths.get_full_path_or_raise("ninfer", name)).resolve()
        if not any(path.is_relative_to(Path(root).resolve()) for root in folder_paths.get_folder_paths("ninfer")):
            raise ValueError("Ninferencing: library model resolves outside its registered folders.")
    else:
        raise ValueError(f"Ninferencing: unknown model source {selection['model']!r}.")
    if path.suffix.lower() != ".ninfer" or not path.is_file():
        raise ValueError(f"Ninferencing: select an existing .ninfer model file: {path}")
    return path
