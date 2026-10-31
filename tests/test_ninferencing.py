import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from PIL import Image
import psutil
import pytest
import torch

import folder_paths
import execution as comfy_execution
from Ninferencing import models, nodes, process, runtime


HELP = """ninfer model.ninfer --messages FILE --max-new N --seed N
--max-context N --kv-capacity N --prefill-chunk N --device N
--kv-dtype bf16|int8|rk8v4 --no-cuda-graph --no-thinking --vision
--reasoning-effort low|medium|xhigh --spec mtp|dflash --draft-tokens N
--lm-head-draft --temperature F --top-p F --top-k N --min-p F
--presence-penalty F --frequency-penalty F
"""
FLAGS = set(re.findall(r"--[\w-]+", HELP))
HELP_ALL = HELP + """
--kv-dtype bf16|int8|fp8|k8v4|nvfp4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8
--thinking-budget N --reasoning-loop off|stop|conclude
--spec mtp|dflash|dflash2 --gdn-state-fp16
--vision-residency resident|overlay|cpu --vision-max-merged N
"""
FLAGS_ALL = set(re.findall(r"--[\w-]+", HELP_ALL))


def command(settings=runtime.Settings(), flags=FLAGS, help_text=HELP, images=False, video=False):
    return runtime.build_command(
        Path("/runtime/ninfer"), Path("/models/any-family.ninfer"), Path("/temp/messages.json"),
        settings, flags, help_text, max_tokens=512, seed=123, device=0, has_images=images, has_video=video,
    )


def test_default_command_is_model_neutral_and_explicit_about_memory():
    argv = command()
    assert argv[1] == str(Path("/models/any-family.ninfer"))
    assert argv[argv.index("--kv-capacity") + 1] == "auto"
    assert argv[argv.index("--max-context") + 1] == "32768"
    assert argv[argv.index("--kv-dtype") + 1] == "int8"
    assert "--no-cuda-graph" in argv and "--no-thinking" in argv
    assert "--spec" not in argv and "--model-id" not in argv
    assert "--vision" not in argv
    assert argv[argv.index("--temperature") + 1] == "0.7"
    assert argv[argv.index("--top-p") + 1] == "0.8"
    assert argv[argv.index("--top-k") + 1] == "20"
    assert argv[argv.index("--min-p") + 1] == "0.0"
    assert argv[argv.index("--presence-penalty") + 1] == "1.5"
    assert argv[argv.index("--frequency-penalty") + 1] == "0.0"
    assert argv[argv.index("--prefill-chunk") + 1] == "1024"
    assert "--thinking-budget" not in argv and "--reasoning-loop" not in argv


def test_prefill_chunk_is_configurable_and_extra_args_wins():
    settings = replace(runtime.Settings(), prefill_chunk=2048)
    argv = command(settings)
    assert argv[argv.index("--prefill-chunk") + 1] == "2048"
    with pytest.raises(ValueError, match="multiple of 128"):
        command(replace(runtime.Settings(), prefill_chunk=100))
    settings = replace(runtime.Settings(), extra_args='["--prefill-chunk", "4096"]')
    argv = command(settings)
    assert argv.count("--prefill-chunk") == 1
    assert argv[argv.index("--prefill-chunk") + 1] == "4096"


def test_thinking_budget_and_reasoning_loop_are_opt_in():
    settings = replace(runtime.Settings(), thinking_budget=512, reasoning_loop="conclude")
    argv = command(settings, flags=FLAGS_ALL, help_text=HELP_ALL)
    assert argv[argv.index("--thinking-budget") + 1] == "512"
    assert argv[argv.index("--reasoning-loop") + 1] == "conclude"
    with pytest.raises(ValueError, match="reasoning_loop"):
        command(replace(runtime.Settings(), reasoning_loop="sometimes"))


def test_draft_tokens_follow_the_engine_limit():
    schema = nodes.NinferencingSettings.define_schema()
    speculation = next(item for item in schema.inputs if item.id == "speculation")
    for option in speculation.options:
        for widget in option.inputs:
            if widget.id == "draft_tokens":
                assert (widget.min, widget.max) == (1, 15)
    with pytest.raises(ValueError, match="1\\.\\.15"):
        command(replace(runtime.Settings(),
                        speculation={"speculation": "mtp", "draft_tokens": 16, "lm_head_draft": False}))
    argv = command(replace(runtime.Settings(),
                           speculation={"speculation": "mtp", "draft_tokens": 15, "lm_head_draft": False}))
    assert argv[argv.index("--draft-tokens") + 1] == "15"


def test_kv_cache_offers_the_small_context_types():
    schema = nodes.NinferencingSettings.define_schema()
    kv_cache = next(item for item in schema.inputs if item.id == "kv_cache")
    for dtype in ("auto", "bf16", "int8", "fp8", "k8v4", "nvfp4"):
        assert dtype in kv_cache.options
    argv = command(replace(runtime.Settings(), kv_cache="fp8"), flags=FLAGS_ALL, help_text=HELP_ALL)
    assert argv[argv.index("--kv-dtype") + 1] == "fp8"


def test_still_image_vision_keeps_auto_kv_capacity():
    argv = command(images=True)
    assert "--vision" in argv
    assert argv[argv.index("--kv-capacity") + 1] == "auto"
    assert argv[argv.index("--max-context") + 1] == "32768"


def test_native_video_uses_explicit_kv_capacity_matching_context():
    settings = replace(runtime.Settings(), context_length=runtime.VIDEO_SAFE_CONTEXT)
    argv = command(settings, images=True, video=True)
    assert "--vision" in argv
    assert argv[argv.index("--max-context") + 1] == str(runtime.VIDEO_SAFE_CONTEXT)
    assert argv[argv.index("--kv-capacity") + 1] == str(runtime.VIDEO_SAFE_CONTEXT)


def test_kv_capacity_setting_overrides_video_default():
    settings = replace(runtime.Settings(), context_length=16384, kv_capacity="4096")
    argv = command(settings, images=True, video=True)
    assert argv[argv.index("--kv-capacity") + 1] == "4096"
    assert argv[argv.index("--max-context") + 1] == "16384"
    settings = replace(runtime.Settings(), context_length=12288, kv_capacity="match_context")
    argv = command(settings, images=False)
    assert argv[argv.index("--kv-capacity") + 1] == "12288"


def test_video_is_optional_and_context_stays_put():
    schema = nodes.Ninferencing.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["video"].optional
    assert inputs["video_duration"].optional and inputs["video_duration"].default == 0.0
    assert inputs["video_max_side"].optional and inputs["video_max_side"].advanced
    for removed in ("start_at_timestamp", "duration_seconds", "video_fps", "video_max_seconds", "video_mode"):
        assert removed not in inputs
    assert [item.display_name for item in schema.outputs] == ["text", "thinking"]
    settings_inputs = {item.id: item for item in nodes.NinferencingSettings.define_schema().inputs}
    assert settings_inputs["kv_capacity"].default == "auto"
    assert inputs["system_prompt"].default == ""
    assert inputs["thinking"].default == "off"
    assert inputs["reasoning_loop"].default == "off"
    assert inputs["thinking_budget"].default == 0
    speculation = next(item for item in schema.inputs if item.id == "speculation")
    assert speculation.options[0].key == "off"


def test_main_thinking_and_speculation_override_settings(tmp_path):
    assert nodes.THINKING_OPTIONS[0] == "off"
    assert nodes.REASONING_LOOP_OPTIONS == ("off", "stop", "conclude")
    assert nodes.SPECULATION_MODES == ("mtp", "dflash", "dflash2")
    base = runtime.Settings()
    assert nodes._override_thinking(base, thinking=["off"]).thinking == "off"
    assert nodes._override_thinking(base, thinking=["low"]).thinking == "low"
    assert nodes._override_thinking(base, thinking_budget=[512]).thinking_budget == 512
    assert nodes._override_thinking(base, reasoning_loop=["conclude"]).reasoning_loop == "conclude"
    assert nodes._override_speculation(
        base, {"speculation": ["mtp"], "draft_tokens": [5], "lm_head_draft": [True]},
    ).speculation == {"speculation": "mtp", "draft_tokens": 5, "lm_head_draft": True}
    assert nodes._override_speculation(base, {"speculation": ["off"]}).speculation == {"speculation": "off"}
    assert nodes._override_speculation(base, "off").speculation == {"speculation": "off"}
    # Main fingerprint covers the overrides so toggle changes invalidate the cache.
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    selection = {"model": "file path", "path": str(artifact)}
    plain = nodes.Ninferencing.fingerprint_inputs(selection, thinking=["off"])
    assert nodes.Ninferencing.fingerprint_inputs(selection, thinking=["low"]) != plain
    assert nodes.Ninferencing.fingerprint_inputs(
        selection, speculation={"speculation": ["mtp"], "draft_tokens": [5]}) != plain


def test_automatic_options_adapt_to_older_build():
    minimal = {"--messages", "--max-new", "--seed", "--max-context"}
    argv = command(flags=minimal, help_text=" ".join(minimal))
    assert {item for item in argv if item.startswith("--")} == minimal
    with pytest.raises(ValueError, match="--vision"):
        command(flags=minimal, help_text="", images=True)


def test_explicit_settings_and_build_specific_arguments():
    settings = replace(
        runtime.Settings(), context_length=12345, thinking="xhigh", cuda_graphs="on", kv_cache="rk8v4",
        speculation={"speculation": "mtp", "draft_tokens": 8, "lm_head_draft": True},
        sampling={"sampling": "custom", "temperature": 0.3, "presence_penalty": 0.8},
        extra_args='["--prefill-chunk", "128", "--future-build-flag", "a path with spaces"]',
    )
    argv = command(settings, images=True)
    assert "--no-cuda-graph" not in argv and "--no-thinking" not in argv
    assert argv[argv.index("--reasoning-effort") + 1] == "xhigh"
    assert argv[argv.index("--max-context") + 1] == "12345"
    assert argv.count("--prefill-chunk") == 1
    assert argv[argv.index("--prefill-chunk") + 1] == "128"
    assert argv[-1] == "a path with spaces"
    assert "--lm-head-draft" in argv and "--vision" in argv
    assert argv[argv.index("--temperature") + 1] == "0.3"
    assert argv[argv.index("--presence-penalty") + 1] == "0.8"


def test_custom_sampling_clamps_values_the_cli_rejects():
    settings = replace(
        runtime.Settings(),
        sampling={
            "sampling": "custom",
            "temperature": 5.0,
            "top_p": 1.5,
            "top_k": 40,
            "min_p": -0.2,
            "presence_penalty": 9.0,
            "frequency_penalty": -9.0,
        },
    )
    argv = command(settings)
    assert argv[argv.index("--temperature") + 1] == "2.0"
    assert argv[argv.index("--top-p") + 1] == "1.0"
    assert argv[argv.index("--top-k") + 1] == "20"
    assert argv[argv.index("--min-p") + 1] == "0.0"
    assert argv[argv.index("--presence-penalty") + 1] == "2.0"
    assert argv[argv.index("--frequency-penalty") + 1] == "-2.0"


def test_custom_sampling_honors_a_wider_range_from_help():
    help_text = HELP + "\n--top-k N 0..40\n--temperature F 0..2\n"
    flags = set(re.findall(r"--[\w-]+", help_text))
    settings = replace(
        runtime.Settings(),
        sampling={"sampling": "custom", "top_k": 40, "temperature": 5.0},
    )
    argv = command(settings, flags=flags, help_text=help_text)
    assert argv[argv.index("--top-k") + 1] == "40"
    assert argv[argv.index("--temperature") + 1] == "2.0"


def test_custom_sampling_widgets_match_cli_ranges():
    sampling = next(item for item in nodes.NinferencingSettings.define_schema().inputs if item.id == "sampling")
    custom = next(option for option in sampling.options if option.key == "custom")
    widgets = {item.id: item for item in custom.inputs}
    assert (widgets["temperature"].min, widgets["temperature"].max) == (0, 2)
    assert (widgets["top_p"].min, widgets["top_p"].max) == (0, 1)
    assert widgets["temperature"].default == 0.7
    assert widgets["top_p"].default == 0.8
    assert widgets["top_k"].default == 20
    assert widgets["min_p"].default == 0.0
    assert widgets["presence_penalty"].default == 1.5
    assert widgets["frequency_penalty"].default == 0.0
    assert sampling.options[0].key == "custom"
    assert (widgets["top_k"].min, widgets["top_k"].max) == (0, 20)
    assert (widgets["min_p"].min, widgets["min_p"].max) == (0, 1)
    assert (widgets["presence_penalty"].min, widgets["presence_penalty"].max) == (-2, 2)
    assert (widgets["frequency_penalty"].min, widgets["frequency_penalty"].max) == (-2, 2)


def test_usage_dump_is_not_a_vision_residency_failure():
    usage = (
        "error: --top-k must be in [0,20]\n"
        "usage: ninfer\n"
        "  --vision-residency resident|overlay|cpu\n"
        "  --vision-max-merged N\n"
    )
    assert runtime.is_vision_residency_failure(usage) is False
    assert runtime.is_vision_residency_failure(
        "--vision-residency overlay needs 816 MiB of evict-ranked weights for one encode window"
    )
    assert runtime.is_vision_residency_failure(
        "lower --vision-max-merged; one encode window still does not fit"
    )


@pytest.mark.parametrize("args", ['["--messages=x"]', '["--raw-output"]', '["--"]', '["some-other-model.ninfer"]', '{"x": 1}'])
def test_extra_arguments_cannot_change_invocation_contract(args):
    with pytest.raises(ValueError):
        command(replace(runtime.Settings(), extra_args=args))


def test_library_containment_and_arbitrary_file_path(tmp_path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    (root / "nested").mkdir()
    artifact = root / "nested" / "completely-new-model.ninfer"
    artifact.write_bytes(b"artifact")
    external = tmp_path / "external.NINFER"
    external.write_bytes(b"different artifact")
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "ninfer", ([str(root)], {".ninfer"}))
    assert models.resolve_model({"model": "library", "name": "nested/completely-new-model.ninfer"}) == artifact
    assert models.resolve_model({"model": "file path", "path": str(external)}) == external
    for name in ("../external.NINFER", "..\\external.NINFER", str(external)):
        with pytest.raises(ValueError):
            models.resolve_model({"model": "library", "name": name})
    external.rename(tmp_path / "external.txt")
    with pytest.raises(ValueError):
        models.resolve_model({"model": "file path", "path": str(tmp_path / "external.txt")})


def test_executable_directory_and_server_path_resolve_to_cli(tmp_path):
    executable = tmp_path / ("ninfer.exe" if os.name == "nt" else "ninfer")
    executable.touch()
    executable.chmod(0o755)
    assert models.resolve_executable(str(tmp_path)) == executable
    serve = tmp_path / ("ninfer-serve.exe" if os.name == "nt" else "ninfer-serve")
    assert models.resolve_executable(str(serve)) == executable


@pytest.mark.skipif(os.name == "nt", reason="POSIX multicall argv0")
def test_multicall_symlink_keeps_ninfer_argv0(tmp_path):
    binary = tmp_path / "ninfer-multicall"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    cli = tmp_path / "ninfer"
    cli.symlink_to("ninfer-multicall")
    for selection in (str(cli), str(binary), str(tmp_path)):
        resolved = models.resolve_executable(selection)
        assert resolved.name == "ninfer"
        assert resolved.parent == tmp_path


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permissions")
def test_non_executable_linux_runtime_is_rejected(tmp_path):
    executable = tmp_path / "ninfer"
    executable.touch()
    executable.chmod(0o644)

    with pytest.raises(PermissionError, match="not executable"):
        models.resolve_executable(str(executable))


@pytest.mark.skipif(os.name == "nt", reason="Linux installation default")
def test_linux_install_executable_is_discovered(tmp_path, monkeypatch):
    runtime = tmp_path / "current"
    runtime.mkdir()
    executable = runtime / "ninfer"
    executable.touch()
    executable.chmod(0o755)
    monkeypatch.delenv("NINFER_EXECUTABLE", raising=False)
    monkeypatch.delenv("NINFER_HOME", raising=False)
    monkeypatch.setattr(models, "_LINUX_NINFER_HOMES", (runtime,))

    assert models.default_executable() == str(executable)
    assert models.resolve_executable("") == executable


def test_home_ninfer_model_directory_is_registered(tmp_path, monkeypatch):
    home = tmp_path / "home"
    model_dir = home / "models" / "ninfer"
    model_dir.mkdir(parents=True)
    registered = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NINFER_MODELS_DIR", raising=False)
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "comfy-models"))
    monkeypatch.setattr(folder_paths, "add_model_folder_path", lambda _name, path, **_kwargs: registered.append(Path(path)))
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "ninfer", ([], set()))
    monkeypatch.setattr(models, "resolve_executable", lambda _value: home / "runtime" / "ninfer")

    models.register_models()

    assert model_dir in registered


@pytest.mark.skipif(os.name == "nt", reason="POSIX shared-library error")
def test_probe_missing_cuda_runtime_points_to_setup_instructions(monkeypatch):
    diagnostic = "/opt/ninfer/current/ninfer: error while loading shared libraries: libcudart.so.12: cannot open shared object file"
    monkeypatch.setattr(runtime, "run_process", lambda *_args, **_kwargs: (127, "", diagnostic))

    with pytest.raises(RuntimeError, match="CUDA runtime.*README"):
        runtime.probe_runtime(Path("/opt/ninfer/current/ninfer"), lambda: None)


def test_memory_handoff_unloads_and_waits(monkeypatch):
    import Ninferencing.vram as vram

    calls = []
    monkeypatch.setattr(vram, "reap_orphan_ninfer", lambda: calls.append("reap") or [42])
    monkeypatch.setattr(vram, "free_bytes", lambda device=0: 8 * 1024 ** 3)
    monkeypatch.setattr(vram.model_management, "throw_exception_if_processing_interrupted", lambda: calls.append("check"))
    monkeypatch.setattr(vram.model_management, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(vram.model_management, "current_loaded_models", [])
    monkeypatch.setattr(vram.model_management, "unload_all_models", lambda: calls.append("unload"))
    monkeypatch.setattr(vram.model_management, "free_memory", lambda *_args, **_kwargs: calls.append("free"))
    monkeypatch.setattr(vram.model_management, "cleanup_models", lambda: calls.append("cleanup"))
    monkeypatch.setattr(vram.model_management, "cleanup_models_gc", lambda: calls.append("cleanup_gc"))
    monkeypatch.setattr(vram.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(vram.model_management, "soft_empty_cache", lambda force: calls.append(("empty", force)))
    monkeypatch.setattr(vram.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    report = nodes.prepare_memory(0, need_bytes=4 * 1024 ** 3, check_cancel=lambda: None)
    assert report["killed_orphans"] == [42]
    assert report["after_free"] == 8 * 1024 ** 3
    assert calls[0] == "reap"
    assert ("empty", True) in calls
    assert "unload" not in calls and "free" not in calls
    assert "cleanup" not in calls and "cleanup_gc" not in calls


def test_unload_comfy_survives_shrinking_registry(monkeypatch):
    import Ninferencing.vram as vram

    registry = []

    class Finalizer:
        def __init__(self):
            self.detached = False

        def detach(self):
            self.detached = True

    class Patcher:
        def __init__(self):
            self.detached = False

        def detach(self, unpatch_all=True):
            assert unpatch_all is True
            # cleanup_models runs on GC during detach and calls real_model().
            for entry in list(registry):
                entry.real_model()
            # The Comfy bug: that finalizer also pops the list while free_memory holds indexes.
            if registry:
                registry.pop(0)
            self.detached = True

    live = type("Loaded", (), {})()
    live.model = Patcher()
    live.real_model = lambda: object()
    live_finalizer = Finalizer()
    live_patcher_finalizer = Finalizer()
    live.model_finalizer = live_finalizer
    live._patcher_finalizer = live_patcher_finalizer

    dead = type("Loaded", (), {})()
    dead.model = None
    dead.real_model = None
    dead.model_finalizer = None
    dead._patcher_finalizer = None

    registry.extend([live, dead])
    forbidden = []
    monkeypatch.setattr(vram.model_management, "current_loaded_models", registry)
    monkeypatch.setattr(vram.model_management, "unload_all_models", lambda: forbidden.append("unload"))
    monkeypatch.setattr(vram.model_management, "free_memory", lambda *_args, **_kwargs: forbidden.append("free"))
    monkeypatch.setattr(vram.model_management, "cleanup_models", lambda: forbidden.append("cleanup"))
    monkeypatch.setattr(vram.model_management, "cleanup_models_gc", lambda: forbidden.append("cleanup_gc"))
    monkeypatch.setattr(vram.model_management, "throw_exception_if_processing_interrupted", lambda: None)
    monkeypatch.setattr(vram.model_management, "synchronize", lambda: None)
    monkeypatch.setattr(vram.model_management, "soft_empty_cache", lambda force=False: None)
    monkeypatch.setattr(vram.gc, "collect", lambda: None)

    vram.unload_comfy(0)

    assert live.model.detached
    assert live_finalizer.detached and live.model_finalizer is None
    assert live_patcher_finalizer.detached and live._patcher_finalizer is None
    assert live.real_model() is None
    assert dead.real_model() is None
    assert registry == []
    assert forbidden == []



class FakeVideo:
    """Minimal VIDEO stand-in: the loader owns trimming, export is full-input only."""

    def __init__(self, source=None, trim=(0.0, 0.0)):
        self._source = source
        self._trim = trim
        self.save_calls = []

    def get_stream_source(self):
        return self._source

    def get_duration(self):
        return 10.0

    def get_active_trim_window(self):
        return self._trim

    def save_to(self, path, format=None, codec=None):
        self.save_calls.append(path)
        Path(path).write_bytes(b"exported")


def test_export_video_path_reuses_on_disk_when_untrimmed(tmp_path):
    source = tmp_path / "plate.mp4"
    source.write_bytes(b"on-disk")
    video = FakeVideo(str(source))
    assert nodes.export_video_path(video, tmp_path) == source
    assert video.save_calls == []


def test_export_video_path_reencodes_without_on_disk_source(tmp_path):
    video = FakeVideo()
    path = nodes.export_video_path(video, tmp_path)
    assert path == tmp_path / "input_video.mp4"
    assert video.save_calls == [str(path)]
    assert path.is_file()


def test_export_video_path_reencodes_with_pending_trim(tmp_path):
    """Unbaked trim metadata: the file bytes are not the presented clip."""
    source = tmp_path / "full_source.mp4"
    source.write_bytes(b"longer-than-presented")
    video = FakeVideo(str(source), trim=(0.0, 8.0))
    path = nodes.export_video_path(video, tmp_path)
    assert path == tmp_path / "input_video.mp4"
    assert video.save_calls == [str(path)]
    assert path.read_bytes() == b"exported"


def test_export_video_path_rejects_missing_video(tmp_path):
    with pytest.raises(ValueError, match="connect a VIDEO input"):
        nodes.export_video_path(None, tmp_path)


def test_full_clip_is_sent_whole_with_planned_tokens(tmp_path):
    import av
    from fractions import Fraction

    from comfy_api.latest import VideoFromComponents, VideoComponents

    count = 40
    images = torch.zeros(count, 32, 48, 3)
    images[:, 0, 0, 0] = torch.arange(count)
    video = VideoFromComponents(VideoComponents(images=images, audio=None, frame_rate=Fraction(24, 1)))
    path, video_tokens = nodes.fit_video_for_vision(video, tmp_path, video_max_side=0)
    assert video_tokens >= 0
    container = av.open(str(path))
    try:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()
    assert len(decoded) == count
    assert nodes.spread_indices(count, 15)[0] == 0
    assert nodes.spread_indices(count, 15)[-1] == count - 1


def test_write_messages_accepts_native_video_url(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    path, has_media = nodes.write_messages(tmp_path, "describe", "sys", {}, 0, 32768, video_paths=[video])
    assert has_media
    messages = json.loads(path.read_text())
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    assert content[0] == {"type": "video_url", "video_url": {"url": str(video)}}
    assert content[-1]["text"] == "describe"


def test_write_messages_video_before_optional_stills(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    images = {"image0": torch.zeros(1, 8, 8, 3)}
    path, has_media = nodes.write_messages(tmp_path, "x", "", images, 0, 32768, video_paths=[video])
    assert has_media
    content = json.loads(path.read_text())[0]["content"]
    assert content[0]["type"] == "video_url"
    assert content[1]["type"] == "image_url"
    assert content[-1]["text"] == "x"


def test_images_preserve_dimensions_and_include_every_batch_item(tmp_path):
    images = {"image0": [torch.zeros(2, 300, 200, 3)], "image1": [torch.ones(1, 19, 27, 4)]}
    path, has_images = nodes.write_messages(tmp_path, "look", "system", images, 0, 32768)
    messages = json.loads(path.read_text(encoding="utf-8"))
    assert has_images
    assert messages[0] == {"role": "system", "content": "system"}
    content = messages[1]["content"]
    assert len(content) == 4 and content[-1]["text"] == "look"
    sizes = []
    for item in content[:-1]:
        with Image.open(item["image_url"]["url"]) as image:
            assert image.mode == "RGB" and image.format == "PNG"
            sizes.append(image.size)
    assert sizes == [(200, 300), (200, 300), (27, 19)]


def test_image_list_socket_accepts_video_batch_and_output_is_list(tmp_path):
    # Video-frames style: one IMAGE batch. Image-bin style: OUTPUT_IS_LIST.
    sources = {
        "image_list": [torch.zeros(3, 40, 30, 3), torch.ones(1, 20, 10, 3)],
        "image0": torch.full((1, 8, 8, 3), 0.5),
    }
    path, has_images = nodes.write_messages(tmp_path, "frames", "", sources, 0, 32768)
    content = json.loads(path.read_text(encoding="utf-8"))[0]["content"]
    assert has_images and len(content) == 6 and content[-1]["text"] == "frames"
    schema_inputs = {item.id: item for item in nodes.Ninferencing.define_schema().inputs}
    assert schema_inputs["image_list"].optional


def test_refs_cable_preserves_order_and_separate_images(tmp_path):
    red = torch.zeros(1, 8, 6, 3)
    red[..., 0] = 1
    green = torch.zeros(1, 5, 9, 3)
    green[..., 1] = 1
    # is_input_list wraps the bin tuple. Extra sockets must follow, not replace, bin order.
    sources = nodes.collect_image_sources(
        images={"image0": torch.full((1, 2, 2, 3), 0.25)},
        image_list=torch.zeros(1, 3, 3, 3),
        refs=[(red, green)],
    )
    assert list(sources) == ["refs", "image0", "image_list"]
    path, has_images = nodes.write_messages(tmp_path, "look", "", sources, 0, 32768)
    content = json.loads(path.read_text(encoding="utf-8"))[0]["content"]
    assert has_images and content[-1]["text"] == "look"
    sizes = []
    colors = []
    for item in content[:-1]:
        assert item["type"] == "image_url"
        with Image.open(item["image_url"]["url"]) as image:
            assert image.format == "PNG" and image.mode == "RGB"
            sizes.append(image.size)
            colors.append(image.getpixel((0, 0)))
    assert sizes == [(6, 8), (9, 5), (2, 2), (3, 3)]
    assert colors[0][0] > 200 and colors[1][1] > 200
    assert colors[2][0] > 40 and colors[3] == (0, 0, 0)


def test_refs_follow_video_and_are_not_thinned(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    refs = tuple(torch.full((1, 4, 4, 3), index / 16) for index in range(1, 10))
    sources = nodes.collect_image_sources(
        images={"image0": torch.zeros(20, 4, 4, 3)},
        refs=refs,
    )
    path, _ = nodes.write_messages(tmp_path, "x", "", sources, 0, 32768, video_paths=[video])
    content = json.loads(path.read_text(encoding="utf-8"))[0]["content"]
    assert content[0]["type"] == "video_url"
    images = content[1:-1]
    assert len(images) == 15  # 9 pinned refs + 6 spread extras; video takes the 16th slot
    with Image.open(images[0]["image_url"]["url"]) as first_image:
        first_pixel = first_image.getpixel((0, 0))[0]
    with Image.open(images[8]["image_url"]["url"]) as last_ref:
        last_pixel = last_ref.getpixel((0, 0))[0]
    assert first_pixel > 10 and last_pixel > first_pixel


def test_refs_over_vision_budget_are_not_thinned(tmp_path):
    refs = tuple(torch.zeros(1, 4, 4, 3) for _ in range(17))
    sources = nodes.collect_image_sources(refs=refs)
    with pytest.raises(ValueError, match="not thinned"):
        nodes.write_messages(tmp_path, "", "", sources, 0, 32768)


def test_video_token_reserve_shrinks_stills_and_prefetch_overflow_raises(tmp_path):
    # 6 large stills alone fit 8192; reserving a 15 s plate (~2.2k tokens) must
    # shrink them instead of overflowing like the old image-as-video 8192 failure.
    sources = {"image_list": [torch.zeros(1, 1600, 1600, 3) for _ in range(6)]}
    plain, _ = nodes.write_messages(tmp_path, "", "", sources, 0, 8192)
    reserved_dir = tmp_path / "reserved"
    reserved_dir.mkdir()
    reserved, _ = nodes.write_messages(reserved_dir, "", "", sources, 0, 8192, video_token_reserve=2160)
    sizes = []
    for path in (plain, reserved):
        content = json.loads(path.read_text())[0]["content"]
        with Image.open(content[0]["image_url"]["url"]) as image:
            sizes.append(max(image.size))
    assert sizes[1] < sizes[0]
    with pytest.raises(ValueError, match="planned prompt exceeds context_length"):
        nodes.write_messages(tmp_path, "", "", sources, 0, 2048, video_token_reserve=2160)


def test_fit_video_for_vision_returns_tokens_and_honors_duration_override(tmp_path):
    import av
    from fractions import Fraction

    import numpy as np

    container = av.open(str(tmp_path / "plate.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(24, 1))
    stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
    for _ in range(48):
        frame = av.VideoFrame.from_ndarray(np.zeros((96, 128, 3), dtype=np.uint8), format="rgb24")
        container.mux(stream.encode(frame))
    container.mux(stream.encode())
    container.close()

    from comfy_api.latest import VideoFromFile

    video = VideoFromFile(str(tmp_path / "plate.mp4"))
    path, tokens = nodes.fit_video_for_vision(video, tmp_path, video_max_side=512)
    assert Path(path).is_file() and tokens > 0
    _, claimed_tokens = nodes.fit_video_for_vision(video, tmp_path, video_max_side=512, video_duration=2.0)
    assert claimed_tokens == tokens


def test_image_resize_is_only_opt_in(tmp_path):
    path, _ = nodes.write_messages(tmp_path, "", "", {"image0": torch.zeros(1, 300, 200, 1)}, 150, 32768)
    source = json.loads(path.read_text())[0]["content"][0]["image_url"]["url"]
    with Image.open(source) as image:
        assert image.size == (100, 150)


def test_schemas_are_valid_and_have_no_required_image():
    for node in (nodes.Ninferencing, nodes.NinferencingSettings):
        schema = node.define_schema()
        schema.finalize()
        schema.validate()
    main_inputs = {item.id: item for item in nodes.Ninferencing.define_schema().inputs}
    settings_inputs = {item.id: item for item in nodes.NinferencingSettings.define_schema().inputs}
    images = main_inputs["images"]
    assert images.optional and images.template.min == 0
    refs = main_inputs["refs"]
    assert refs.optional and refs.get_io_type() == "CF_REFS"
    assert main_inputs["image_list"].optional and main_inputs["image_list"].advanced
    assert main_inputs["context_length"].default == 32768
    assert "seed" not in main_inputs and settings_inputs["seed"].default == 0
    assert settings_inputs["seed"].control_after_generate == nodes.io.ControlAfterGenerate.fixed
    assert settings_inputs["kv_capacity"].default == "auto"


def test_node_orders_handoff_and_cleans_temp_files_on_failure(tmp_path, monkeypatch):
    artifact = tmp_path / "new-family.ninfer"
    artifact.touch()
    monkeypatch.setattr(nodes, "resolve_executable", lambda value: Path(sys.executable))
    monkeypatch.setattr(nodes, "probe_runtime", lambda *args: (HELP, FLAGS))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    calls = []
    monkeypatch.setattr(nodes, "prepare_memory", lambda *args, **kwargs: calls.append("prepare") or {"after_free": 1})

    def fail(argv, timeout, check_cancel, directory, **kwargs):
        assert calls == ["prepare"]
        assert Path(argv[argv.index("--messages") + 1]).is_file()
        assert kwargs.get("on_capacity") is not None
        calls.append("generate")
        raise RuntimeError("native load failed")

    monkeypatch.setattr(nodes, "generate", fail)
    with pytest.raises(RuntimeError, match="native load failed"):
        nodes.Ninferencing.execute(
            {"model": ["file path"], "path": [str(artifact)]}, ["hello"], [""], [32], [32768], [0],
        )
    assert calls == ["prepare", "generate"]
    assert not list(tmp_path.glob("ninferencing-*"))
    assert nodes._execution_lock.acquire(blocking=False)
    nodes._execution_lock.release()


def test_capacity_failure_retries_with_tighter_kv_capacity(monkeypatch):
    calls = []
    reclaims = []

    def fail(command, **kwargs):
        calls.append(list(command))
        return 1, "", (
            "minimum Engine runtime reservation requires 3575176704 bytes in addition to "
            "1073741824 bytes of automatic headroom, but only 4061181440 bytes are available after weights"
        )

    monkeypatch.setattr(runtime, "run_process", fail)
    argv = command(replace(runtime.Settings(), context_length=17000))
    assert argv[argv.index("--kv-capacity") + 1] == "auto"
    with pytest.raises(RuntimeError, match="could not allocate enough device memory"):
        runtime.generate(
            argv, 5, lambda: None, None,
            on_capacity=lambda: reclaims.append("reclaim") or {"after_free": 0, "need_bytes": 1, "killed_orphans": [], "live_ninfer": [9]},
            vram_report={"before_free": 1, "after_free": 2, "need_bytes": 3, "killed_orphans": [], "live_ninfer": []},
        )
    # Text (no --vision): auto → 8K, then half to 4K across up to 3 attempts.
    assert len(calls) == 3
    assert calls[0] == argv
    assert calls[1][calls[1].index("--kv-capacity") + 1] == str(runtime.VIDEO_SAFE_CONTEXT)
    assert calls[1][calls[1].index("--max-context") + 1] == str(runtime.VIDEO_SAFE_CONTEXT)
    assert calls[2][calls[2].index("--kv-capacity") + 1] == "4096"
    assert reclaims == ["reclaim", "reclaim"]


def test_capacity_failure_without_tighten_keeps_same_command(monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(list(command))
        return 1, "", "requested Engine runtime reservation requires 1947393536 bytes, but only 0 bytes are available for runtime capacity"

    monkeypatch.setattr(runtime, "run_process", fail)
    argv = command(replace(runtime.Settings(), context_length=17000))
    with pytest.raises(RuntimeError, match="could not allocate enough device memory"):
        runtime.generate(
            argv, 5, lambda: None, None,
            on_capacity=lambda: {"after_free": 0},
            vram_report={"before_free": 1, "after_free": 2, "need_bytes": 3},
            tighten_on_capacity=False,
        )
    assert len(calls) == 2 and calls[0] == calls[1] == argv


def test_baked_2fps_is_never_passed_to_the_cli():
    from Ninferencing import videosample

    assert videosample.NINFER_VIDEO_FPS == 2.0
    assert "--video-fps" not in command(flags=FLAGS_ALL, help_text=HELP_ALL, video=True)
    assert "--video-fps" not in command(flags=FLAGS_ALL, help_text=HELP_ALL, images=True, video=True)


def test_ninfer_all_adds_gdn_and_vision_overlay():
    argv = command(flags=FLAGS_ALL, help_text=HELP_ALL, images=True)
    assert "--gdn-state-fp16" in argv
    assert "--vision" in argv
    assert argv[argv.index("--vision-residency") + 1] == "overlay"
    assert argv[argv.index("--vision-max-merged") + 1] == str(runtime.VISION_MAX_MERGED)
    text = command(flags=FLAGS_ALL, help_text=HELP_ALL)
    assert "--gdn-state-fp16" in text
    assert "--vision" not in text and "--vision-residency" not in text


def test_dflash2_speculation_is_passed_when_advertised():
    settings = replace(
        runtime.Settings(),
        speculation={"speculation": "dflash2", "draft_tokens": 5, "lm_head_draft": False},
        kv_cache="rk4v4",
    )
    argv = command(settings, flags=FLAGS_ALL, help_text=HELP_ALL)
    assert argv[argv.index("--spec") + 1] == "dflash2"
    assert argv[argv.index("--draft-tokens") + 1] == "5"
    assert argv[argv.index("--kv-dtype") + 1] == "rk4v4"


def test_v3_artifact_requires_ninfer_all_cli(tmp_path):
    artifact = tmp_path / "bonsai.ninfer"
    artifact.write_bytes(b"NINFER\x00\x03" + b"\0" * 32)
    with pytest.raises(RuntimeError, match="NInfer v3 artifact"):
        runtime.check_artifact_runtime(artifact, HELP)
    runtime.check_artifact_runtime(artifact, HELP_ALL)
    v2 = tmp_path / "qwen.ninfer"
    v2.write_bytes(b"NINFER\x00\x02" + b"\0" * 32)
    runtime.check_artifact_runtime(v2, HELP)
    assert models.artifact_container_version(artifact) == 3
    assert models.artifact_container_version(v2) == 2


def test_v1_v2_magic_failure_names_ninfer_all(monkeypatch):
    monkeypatch.setattr(
        runtime, "run_process",
        lambda *args, **kwargs: (1, "", "error: artifact magic is not NInfer v1 or v2\nusage: ninfer"),
    )
    with pytest.raises(RuntimeError, match="v3 container") as error:
        runtime.generate(command(), 5, lambda: None, None)
    assert "ninfer-linux" in str(error.value)
    assert "usage:" in str(error.value)


def test_sampling_usage_error_is_not_retried_as_vision(monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(list(command))
        return 1, "", (
            "error: --top-k must be in [0,20]\n"
            "usage: ninfer\n"
            "  --vision-residency resident|overlay|cpu\n"
            "  --vision-max-merged N\n"
        )

    monkeypatch.setattr(runtime, "run_process", fail)
    argv = command(flags=FLAGS_ALL, help_text=HELP_ALL, images=True)
    with pytest.raises(RuntimeError, match="top-k must be in") as error:
        runtime.generate(argv, 5, lambda: None, None)
    assert "evict-ranked weights" not in str(error.value)
    assert len(calls) == 1


def test_vision_overlay_shortfall_still_retries(monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(list(command))
        return 1, "", (
            "--vision-residency overlay needs 816 MiB of evict-ranked weights "
            "for one encode window, but provide 656 MiB; lower --vision-max-merged"
        )

    monkeypatch.setattr(runtime, "run_process", fail)
    argv = command(flags=FLAGS_ALL, help_text=HELP_ALL, images=True)
    with pytest.raises(RuntimeError, match="evict-ranked"):
        runtime.generate(argv, 5, lambda: None, None)
    assert calls[0][calls[0].index("--vision-max-merged") + 1] == "8192"
    assert calls[1][calls[1].index("--vision-max-merged") + 1] == "4096"


def test_non_capacity_failure_is_not_retried(monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(1)
        return 7, "", "artifact magic is not NInfer"

    monkeypatch.setattr(runtime, "run_process", fail)
    with pytest.raises(RuntimeError, match="exited with code 7"):
        runtime.generate(command(), 5, lambda: None, None, on_capacity=lambda: calls.append("nope"))
    assert calls == [1]


def test_missing_native_target_has_actionable_error(monkeypatch):
    diagnostics = (
        "phase       detail\n"
        "error: artifact identity 'qwen3.5-9b/groupwise-int' has no registered target for this device\n"
        "usage: ninfer.exe <model.ninfer> (--prompt <text>|--messages <messages.json>)"
    )
    monkeypatch.setattr(runtime, "run_process", lambda *args, **kwargs: (1, "", diagnostics))
    with pytest.raises(RuntimeError, match="Install a ninfer build that registers this target for your GPU") as error:
        runtime.generate(command(), 5, lambda: None, None)
    assert "qwen3.5-9b/groupwise-int" in str(error.value)
    assert "Sampling, context, and VRAM settings cannot fix" in str(error.value)
    assert "usage:" not in str(error.value)


def test_vision_processor_budget_names_the_item_limit(monkeypatch):
    diagnostics = (
        "phase       detail\n"
        "summary     tensors/resources        1104 / 6\n"
        "error: media item count exceeds processor budget\n"
        "usage: ninfer <model.ninfer> (--prompt <text>|--messages <messages.json>)"
    )
    monkeypatch.setattr(runtime, "run_process", lambda *args, **kwargs: (1, "", diagnostics))
    with pytest.raises(RuntimeError, match="at most 16 images or frames per request") as error:
        runtime.generate(command(), 5, lambda: None, None)
    assert "capped automatically" in str(error.value)
    assert "usage:" not in str(error.value)


def test_frame_count_decides_the_automatic_image_cap(tmp_path):
    # Vision cost is the square of each image's patch grid, so more frames buy fewer pixels.
    assert runtime.vision_side_limit(1, 0) == 1696
    assert runtime.vision_side_limit(8, 0) == 992
    assert runtime.vision_side_limit(16, 0) == 832
    assert runtime.vision_side_limit(16, 512) == 512
    assert runtime.vision_side_limit(0, 640) == 640
    for count in range(1, 17):
        side = runtime.vision_side_limit(count, 0)
        grid = side // 16
        assert (grid + 1) ** 4 * count <= runtime.VISION_PAIR_BUDGET  # ninfer may round a side up

    frames = [torch.zeros(1, 2048, 2048, 3)] * 16
    path, _ = nodes.write_messages(tmp_path, "", "", {"image_list": frames}, 0, 32768)
    content = json.loads(path.read_text(encoding="utf-8"))[0]["content"]
    with Image.open(content[0]["image_url"]["url"]) as image:
        assert image.size == (832, 832)


def test_a_large_system_prompt_shrinks_images_before_it_overflows_the_context():
    # 16 frames at the pair cap cost 10,816 vision tokens; a very large system prompt cannot fit beside them.
    assert runtime.vision_side_limit(16, 0, 32768, 0) == 832
    shrunk = runtime.vision_side_limit(16, 0, 32768, 30_000)
    assert shrunk == 384
    grid = shrunk // 32
    assert 16 * (grid * grid + grid + 3) + 30_000 <= 32768
    # The prompt budget is the hard frontend limit, so a wider context does not raise it.
    assert runtime.vision_side_limit(16, 0, 262_144, 30_000) == shrunk
    # One image still rides the pair cap, minus what the prompt needs.
    single = runtime.vision_side_limit(1, 0, 32768, 30_000)
    assert single == 1664
    assert (single // 32) ** 2 + single // 32 + 3 + 30_000 <= 32768


def test_estimate_need_bytes_video_is_tighter_than_old_image_pad(tmp_path):
    import Ninferencing.vram as vram

    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"x" * 1000)
    video_need = vram.estimate_need_bytes(artifact, True, has_video=True)
    image_need = vram.estimate_need_bytes(artifact, True, has_video=False)
    text_need = vram.estimate_need_bytes(artifact, False)
    # Video estimate must not revive the old weights+4GiB overstatement (~20.96 GiB on 27B).
    assert video_need == 1000 + int(3.8 * 1024 ** 3)
    assert image_need == 1000 + 3 * 1024 ** 3 + 512 * 1024 ** 2
    assert text_need == 1000 + 3 * 1024 ** 3
    assert video_need < 1000 + 4 * 1024 ** 3


def test_unknown_tensor_format_is_not_capacity_failure():
    import Ninferencing.vram as vram

    # ninfer dumps usage after load errors; "headroom" must not match as OOM.
    diagnostics = (
        "error: unknown tensor format: Q3G64_F16S\n"
        "usage: ninfer.exe <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
        "--kv-capacity auto leaves 1024 MiB of sizing headroom.\n"
    )
    assert not vram.is_capacity_failure(diagnostics)
    assert vram.is_capacity_failure(
        "requested Engine runtime reservation requires 1 bytes, but only 0 bytes are available for runtime capacity"
    )


def test_weight_allocation_shortfall_is_capacity_failure():
    import Ninferencing.vram as vram

    diagnostic = (
        "error: model weights require 17093490688 bytes of device memory, "
        "but only 13107724288 bytes are free before loading weights"
    )
    assert vram.is_capacity_failure(diagnostic)


def test_orphan_ninfer_reap_kills_dead_parent_only(monkeypatch):
    import Ninferencing.vram as vram

    class Fake:
        def __init__(self, pid):
            self.pid = pid
            self.killed = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return 0

    orphan = Fake(11)
    monkeypatch.setattr(vram, "orphan_ninfer_processes", lambda exclude_pids=None: [orphan])
    assert vram.reap_orphan_ninfer() == [11]
    assert orphan.killed


def test_orphan_detection_skips_processes_with_live_parents(monkeypatch):
    import Ninferencing.vram as vram

    class Proc:
        def __init__(self, pid, name, ppid):
            self.info = {"pid": pid, "name": name, "ppid": ppid}

    class FakePsutil:
        class Error(Exception):
            pass

        @staticmethod
        def process_iter(attrs):
            return [
                Proc(10, "ninfer-serve.exe", 1),
                Proc(11, "ninfer.exe", 99999),
                Proc(12, "chrome.exe", 1),
            ]

        @staticmethod
        def pid_exists(pid):
            return pid == 1

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    orphans = vram.orphan_ninfer_processes()
    assert [p.info["pid"] for p in orphans] == [11]


def test_comfy_execution_expands_settings_and_aggregates_image_lists(tmp_path, monkeypatch):
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    monkeypatch.setattr(nodes, "resolve_executable", lambda value: Path(sys.executable))
    monkeypatch.setattr(nodes, "probe_runtime", lambda *args: (HELP, FLAGS))
    monkeypatch.setattr(nodes, "prepare_memory", lambda *args, **kwargs: {"after_free": 9 * 1024 ** 3})
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    settings_data, _, settings_v3 = comfy_execution.get_input_data({
        "executable": "a custom build", "seed": 9,
        "speculation": "mtp", "speculation.draft_tokens": 3, "speculation.lm_head_draft": True,
        "sampling": "custom", "sampling.temperature": 0.4,
    }, nodes.NinferencingSettings, "settings")
    settings_result = asyncio.run(comfy_execution._async_map_node_over_list(
        "test", "settings", nodes.NinferencingSettings, settings_data, "execute", v3_data=settings_v3,
    ))
    settings = settings_result[0].result[0]
    assert settings.speculation == {"speculation": "mtp", "draft_tokens": 3, "lm_head_draft": True}
    assert settings.sampling == {"sampling": "custom", "temperature": 0.4}
    assert "top_p" not in settings.sampling
    inputs, missing, v3_data = comfy_execution.get_input_data({
        "model": "file path", "model.path": str(artifact), "prompt": "look", "system_prompt": "",
        "max_tokens": 123, "context_length": 8192, "image_max_side": 4,
        "thinking": "off", "thinking_budget": 0, "reasoning_loop": "off",
        "speculation": "mtp", "speculation.draft_tokens": 3, "speculation.lm_head_draft": True,
        "images.image0": ["image-node", 0],
        "image_list": ["video-frames", 0], "settings": ["settings", 0],
    }, nodes.Ninferencing, "generate")
    assert missing.keys() == {"settings", "images.image0", "image_list"}
    inputs["images.image0"] = [torch.zeros(2, 2, 3, 3), torch.zeros(1, 3, 4, 3)]
    # OUTPUT_IS_LIST of single-frame tensors (e.g. image-bin image_list)
    inputs["image_list"] = [torch.zeros(1, 5, 5, 3), torch.ones(1, 6, 6, 3)]
    inputs["settings"] = [settings]
    calls = []

    def complete(argv, timeout, check_cancel, directory, **kwargs):
        messages = json.loads(Path(argv[argv.index("--messages") + 1]).read_text())
        # 3 from Autogrow batches + 2 from image_list + text
        assert len(messages[0]["content"]) == 6
        for item in messages[0]["content"][:-1]:
            with Image.open(item["image_url"]["url"]) as image:
                assert max(image.size) <= 4
        assert "--vision" in argv and "--lm-head-draft" in argv
        assert argv[argv.index("--seed") + 1] == "9"
        assert argv[argv.index("--max-context") + 1] == "8192"
        assert argv[argv.index("--kv-capacity") + 1] == "auto"
        assert argv[argv.index("--temperature") + 1] == "0.4"
        calls.append(argv)
        return "one answer for every image", "reasoning trace"

    monkeypatch.setattr(nodes, "generate", complete)
    results = asyncio.run(comfy_execution._async_map_node_over_list(
        "test", "generate", nodes.Ninferencing, inputs, "execute", v3_data=v3_data,
    ))
    assert results[0].result == ("one answer for every image", "reasoning trace")
    assert results[0].ui is None  # no on-node preview; the text goes to the output socket
    assert len(calls) == 1
    assert not list(tmp_path.glob("ninferencing-*"))


def test_replacing_artifact_or_default_runtime_invalidates_comfy_cache(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.ninfer"
    executable = tmp_path / "ninfer"
    artifact.write_bytes(b"v1")
    executable.write_bytes(b"v1")
    monkeypatch.setattr(nodes, "resolve_executable", lambda value: executable)
    selection = {"model": "file path", "path": str(artifact)}
    original = nodes.Ninferencing.fingerprint_inputs(selection)
    artifact.write_bytes(b"v2 model")
    changed_model = nodes.Ninferencing.fingerprint_inputs(selection)
    assert changed_model != original
    executable.write_bytes(b"v2 binary")
    assert nodes.Ninferencing.fingerprint_inputs(selection) != changed_model


def run_python(code, tmp_path, **kwargs):
    return process.run_process([sys.executable, "-c", code], cwd=tmp_path, temp_dir=tmp_path, **kwargs)


def test_real_process_exit_separates_answer_from_diagnostics(tmp_path):
    code, text, tail = run_python(
        "import sys; print('answer'); print('diagnostic', file=sys.stderr)", tmp_path, timeout=10, check_cancel=lambda: None,
    )
    assert code == 0 and text.strip() == "answer" and tail == "diagnostic"


def test_large_diagnostics_do_not_deadlock_and_tail_is_bounded(tmp_path):
    code, text, tail = run_python(
        "import sys; sys.stderr.write('x' * 1000000); print('answer')", tmp_path, timeout=10, check_cancel=lambda: None,
    )
    assert code == 0 and text.strip() == "answer" and len(tail) == 12288


def test_full_stderr_is_available_for_the_thinking_output(tmp_path):
    code, text, full = run_python(
        "import sys; sys.stderr.write('thinking-trace'); print('answer')", tmp_path, timeout=10,
        check_cancel=lambda: None, tail_limit=None,
    )
    assert code == 0 and text.strip() == "answer" and full == "thinking-trace"


def test_generate_returns_answer_and_thinking_stream(monkeypatch):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return 0, "the answer", (
            "engine ready | qwen3.8-27b | total 48.1 s | weights 17.00 GiB | CUDA sync on\n"
            "the reasoning trace\n"
            "phase       detail                      elapsed/progress\n"
            "summary     finish reason               stop-token\n"
        )

    monkeypatch.setattr(runtime, "run_process", fake)
    text, thinking = runtime.generate(command(), 5, lambda: None, None)
    assert text == "the answer"
    assert thinking == "the reasoning trace"
    # The thinking output keeps the whole stderr, not the bounded error tail.
    assert seen.get("tail_limit") is None


def test_extract_reasoning_strips_startup_logs_and_summary():
    stderr = (
        "build 7a89828a3446cd1fa7bb1465ad10386469718e05\n"
        "starting engine\n"
        "loading weights | 17.00 GiB\n"
        "  loading weights [======>.....] 45% | 7.65 GiB/17.00 GiB | 390.12 MiB/s | ETA 24.5 s\n"
        "weights ready | 17.00 GiB | 45.2 s | 390.12 MiB/s\n"
        "engine ready | qwen3.8-27b | total 48.1 s | weights 17.00 GiB | CUDA sync on\n"
        "First thought line.\n"
        "engine | transfer 12.0 ms | prefill 3.1 s\n"
        "Second thought line.\n"
        "phase       detail                      elapsed/progress\n"
        "prepare     render/preprocess           3.1 s\n"
        "summary     sampling                    temp=0.70 top_p=0.80 top_k=20 min_p=0.00 presence=1.50 freq=0.00 seed=0\n"
    )
    assert runtime.extract_reasoning(stderr) == "First thought line.\nSecond thought line."


def test_extract_reasoning_keeps_quoted_markup_and_strips_prefixed_lines():
    stderr = (
        "The </think> tag closes reasoning.\n"
        "debug: frontend ready\n"
        "media | swscaler: deprecated pixel format\n"
        "warning: engine | prefix cache not restored | none\n"
        "error: this line is a log record, not thinking\n"
        "actual thought\n"
    )
    assert runtime.extract_reasoning(stderr) == (
        "The </think> tag closes reasoning.\nactual thought"
    )


def test_extract_reasoning_strips_ansi_before_matching():
    stderr = "\x1b[31m\x1b[1merror: \x1b[mkaput\nactual thought\n"
    assert runtime.extract_reasoning(stderr) == "actual thought"


def test_extract_reasoning_empty_without_stderr():
    assert runtime.extract_reasoning("") == ""
    assert runtime.extract_reasoning(None) == ""
    assert runtime.extract_reasoning(
        "engine ready | qwen3.8-27b | total 48.1 s | weights 17.00 GiB | CUDA sync on\n"
        "phase       detail                      elapsed/progress\n"
        "summary     finish reason               stop-token\n"
    ) == ""


def test_extract_reasoning_drops_barless_progress_and_following_startup_lines():
    # Real Gemma4 startup tail: the bar-less persistent progress line (no `[==>]`)
    # used to break the leading block, leaking the lines after it too.
    stderr = (
        "build 7a89828a3446cd1fa7bb1465ad10386469718e05\n"
        "starting engine\n"
        "loading weights | 16.9 GiB\n"
        "  loading weights 97.2% | 16.4 GiB/16.9 GiB | 1.76 GiB/s | ETA 270 ms\n"
        "weights ready | 16.9 GiB | 10.3s | 1.64 GiB/s\n"
        "CUDA graphs ready | 519 ms\n"
        "engine ready | xdavxd-gemma4-26B-A4B-qat-heretic-ega-MTP-sm86 | total 12.0s | weights 15.8 GiB | CUDA sync auto\n"
        "phase       detail                      elapsed/progress\n"
        "summary     finish reason               stop-token\n"
    )
    assert runtime.extract_reasoning(stderr) == ""


def test_extract_reasoning_keeps_trace_after_barless_progress():
    stderr = (
        "loading weights | 16.9 GiB\n"
        "  loading weights 97.2% | 16.4 GiB/16.9 GiB | 1.76 GiB/s | ETA 270 ms\n"
        "weights ready | 16.9 GiB | 10.3s | 1.64 GiB/s\n"
        "engine ready | xdavxd-gemma4-26B-A4B-qat-heretic-ega-MTP-sm86 | total 12.0s | weights 15.8 GiB | CUDA sync auto\n"
        "Let me think through this.\n"
        "  loading weights [======>.....] 45% | bracketed TTY form never matches prose\n"
        "phase       detail                      elapsed/progress\n"
        "summary     finish reason               stop-token\n"
    )
    assert runtime.extract_reasoning(stderr) == (
        "Let me think through this.\n"
        "  loading weights [======>.....] 45% | bracketed TTY form never matches prose"
    )


def test_node_execute_returns_text_and_thinking(tmp_path, monkeypatch):
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    monkeypatch.setattr(nodes, "resolve_executable", lambda value: Path(sys.executable))
    monkeypatch.setattr(nodes, "probe_runtime", lambda *args: (HELP, FLAGS))
    monkeypatch.setattr(nodes, "prepare_memory", lambda *args, **kwargs: {"after_free": 9 * 1024 ** 3})
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes, "generate", lambda *args, **kwargs: ("the answer", "the trace"))
    out = nodes.Ninferencing.execute(
        {"model": ["file path"], "path": [str(artifact)]}, ["hello"], [""], [32], [32768], [0],
    )
    assert tuple(out) == ("the answer", "the trace")


class Cancelled(BaseException):
    pass


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_comfy_style_interrupt_reap_child(tmp_path, cancel):
    pidfile = tmp_path / "pid.txt"
    code = f"import os,time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(60)"

    def check():
        if cancel and pidfile.exists():
            raise Cancelled()

    with pytest.raises(Cancelled if cancel else TimeoutError):
        run_python(code, tmp_path, timeout=0.5 if not cancel else 10, check_cancel=check)
    pid = int(pidfile.read_text())
    assert not psutil.pid_exists(pid)


def test_descendants_die_even_when_root_exits_successfully(tmp_path):
    pidfile = tmp_path / "descendant.txt"
    code = (
        "import subprocess,sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))"
    )
    result = run_python(code, tmp_path, timeout=10, check_cancel=lambda: None)
    assert result[0] == 0
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while psutil.pid_exists(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(child)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_hard_parent_exit_closes_job_and_releases_child(tmp_path):
    # Import only process.py so the disposable parent never imports torch/ComfyUI.
    pidfile = tmp_path / "owned-child.txt"
    child_script = tmp_path / "child.py"
    child_script.write_text(f"import os,time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import importlib.util,sys\n"
        f"spec = importlib.util.spec_from_file_location('owned_process', {process.__file__!r})\n"
        "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
        f"module.run_process([sys.executable, {str(child_script)!r}], cwd={str(tmp_path)!r}, timeout=60, check_cancel=lambda: None)\n"
    )
    parent = subprocess.Popen([sys.executable, str(parent_script)])
    child = None
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(pidfile.read_text())
        parent.kill()
        parent.wait(timeout=10)
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not psutil.pid_exists(child)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if child is not None and psutil.pid_exists(child):
            psutil.Process(child).kill()


def test_waiting_for_another_execution_is_interruptible(monkeypatch, tmp_path):
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    monkeypatch.setattr(nodes, "resolve_executable", lambda value: Path(sys.executable))
    monkeypatch.setattr(nodes.model_management, "throw_exception_if_processing_interrupted", lambda: (_ for _ in ()).throw(Cancelled()))
    lock = threading.Lock()
    lock.acquire()
    monkeypatch.setattr(nodes, "_execution_lock", lock)
    try:
        with pytest.raises(Cancelled):
            nodes.Ninferencing.execute({"model": "file path", "path": str(artifact)}, ["hi"], [""], [8], [32768], [0])
    finally:
        lock.release()


def test_tests_run_with_cuda_devices_hidden():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_image_vision_capacity_retry_uses_mid_context_not_video_floor(monkeypatch):
    """Multimodal stills: auto shortfall must not slam max-context to 8K first."""
    calls = []

    def fail_then_ok(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            return 1, "", (
                "minimum Engine runtime reservation requires 3575176704 bytes in addition to "
                "1073741824 bytes of automatic headroom, but only 4061181440 bytes are available after weights"
            )
        return 0, "rewritten prompt", ""

    monkeypatch.setattr(runtime, "run_process", fail_then_ok)
    argv = command(replace(runtime.Settings(), context_length=32768), images=True)
    assert "--vision" in argv and argv[argv.index("--kv-capacity") + 1] == "auto"
    text, thinking = runtime.generate(argv, 5, lambda: None, None)
    assert text == "rewritten prompt"
    assert thinking == ""
    assert len(calls) == 2
    assert calls[1][calls[1].index("--kv-capacity") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
    assert calls[1][calls[1].index("--max-context") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)


def test_tighten_from_explicit_high_context_steps_through_mid_ladder():
    argv = command(replace(runtime.Settings(), context_length=32768, kv_capacity="match_context"), images=True)
    assert argv[argv.index("--kv-capacity") + 1] == "32768"
    tightened = runtime.tighten_command_for_capacity(argv)
    assert tightened[tightened.index("--kv-capacity") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
    assert tightened[tightened.index("--max-context") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
    again = runtime.tighten_command_for_capacity(tightened)
    assert again[again.index("--kv-capacity") + 1] == str(runtime.IMAGE_MID_CONTEXT)
    third = runtime.tighten_command_for_capacity(again)
    assert third[third.index("--kv-capacity") + 1] == str(runtime.VIDEO_SAFE_CONTEXT)


def test_prepared_prompt_exceeds_retries_with_higher_context(monkeypatch):
    calls = []

    def fail_then_ok(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            return 1, "", "error: prepared prompt exceeds Engine context capacity"
        return 0, "ok", ""

    monkeypatch.setattr(runtime, "run_process", fail_then_ok)
    settings = replace(runtime.Settings(), context_length=runtime.VIDEO_SAFE_CONTEXT, kv_capacity="match_context")
    argv = command(settings, images=True)
    assert argv[argv.index("--max-context") + 1] == str(runtime.VIDEO_SAFE_CONTEXT)
    text, _thinking = runtime.generate(argv, 5, lambda: None, None)
    assert text == "ok"
    assert len(calls) == 2
    assert calls[1][calls[1].index("--max-context") + 1] == str(runtime.IMAGE_MID_CONTEXT)
    assert calls[1][calls[1].index("--kv-capacity") + 1] == str(runtime.IMAGE_MID_CONTEXT)


def test_auto_capacity_then_prompt_exceed_expands_instead_of_hard_fail(monkeypatch):
    """Reproduce the character-swap image-node failure sequence and recover."""
    calls = []

    def scripted(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            return 1, "", (
                "minimum Engine runtime reservation requires 3575176704 bytes in addition to "
                "1073741824 bytes of automatic headroom, but only 4061181440 bytes are available after weights"
            )
        if len(calls) == 2:
            # If we had slammed to 8K this would fire; mid-context should usually succeed.
            # Force the exceed path once to prove expand works after a mid tighten.
            ctx = int(command[command.index("--max-context") + 1])
            if ctx <= runtime.IMAGE_SAFE_CONTEXT:
                return 1, "", "error: prepared prompt exceeds Engine context capacity"
            return 0, "done", ""
        return 0, "done", ""

    monkeypatch.setattr(runtime, "run_process", scripted)
    argv = command(replace(runtime.Settings(), context_length=32768), images=True)
    text, _thinking = runtime.generate(argv, 5, lambda: None, None)
    assert text == "done"
    assert len(calls) == 3
    assert calls[1][calls[1].index("--max-context") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
    # Expand ladder next above 16384 is 24576.
    assert calls[2][calls[2].index("--max-context") + 1] == "24576"
    assert calls[2][calls[2].index("--kv-capacity") + 1] == "24576"


def test_is_prompt_context_failure_detection():
    from Ninferencing import vram
    assert vram.is_prompt_context_failure("error: prepared prompt exceeds Engine context capacity")
    assert not vram.is_prompt_context_failure("runtime reservation requires more bytes")
    assert not vram.is_capacity_failure("error: prepared prompt exceeds Engine context capacity")


def test_settings_kv_capacity_options_include_mid_contexts():
    settings_inputs = {item.id: item for item in nodes.NinferencingSettings.define_schema().inputs}
    assert "12288" in settings_inputs["kv_capacity"].options
    assert "16384" in settings_inputs["kv_capacity"].options


def test_image_auto_at_mid_context_retries_explicit_same_size(monkeypatch):
    """auto at 16K should become explicit 16K (drop headroom), not shrink to 12K/8K."""
    calls = []

    def fail_then_ok(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            return 1, "", (
                "minimum Engine runtime reservation requires 3575176704 bytes in addition to "
                "1073741824 bytes of automatic headroom, but only 4061181440 bytes are available after weights"
            )
        return 0, "ok", ""

    monkeypatch.setattr(runtime, "run_process", fail_then_ok)
    argv = command(replace(runtime.Settings(), context_length=runtime.IMAGE_SAFE_CONTEXT), images=True)
    assert argv[argv.index("--kv-capacity") + 1] == "auto"
    assert runtime.generate(argv, 5, lambda: None, None) == ("ok", "")
    assert calls[1][calls[1].index("--kv-capacity") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
    assert calls[1][calls[1].index("--max-context") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
