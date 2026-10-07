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
--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8
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
    assert "--temperature" not in argv and "--vision" not in argv


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


def test_video_node_default_context_is_vram_safe():
    schema = nodes.NinferencingVideo.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["context_length"].default == runtime.VIDEO_SAFE_CONTEXT
    settings_inputs = {item.id: item for item in nodes.NinferencingSettings.define_schema().inputs}
    assert "kv_capacity" in settings_inputs
    assert settings_inputs["kv_capacity"].default == "auto"


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
    monkeypatch.setattr(vram.model_management, "unload_all_models", lambda: calls.append("unload"))
    monkeypatch.setattr(vram.model_management, "cleanup_models", lambda: calls.append("cleanup"))
    monkeypatch.setattr(vram.model_management, "cleanup_models_gc", lambda: calls.append("cleanup_gc"))
    monkeypatch.setattr(vram.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(vram.model_management, "soft_empty_cache", lambda force: calls.append(("empty", force)))
    monkeypatch.setattr(vram.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    report = nodes.prepare_memory(0, need_bytes=4 * 1024 ** 3, check_cancel=lambda: None)
    assert report["killed_orphans"] == [42]
    assert report["after_free"] == 8 * 1024 ** 3
    assert calls[0] == "reap"
    assert "unload" in calls and "cleanup" in calls and ("empty", True) in calls



class FakeVideo:
    """Minimal VIDEO stand-in for export_video_path call-arg checks."""

    def __init__(self, source=None, duration=10.0):
        self._source = source
        self._duration = duration
        self.as_trimmed_calls = []
        self.save_calls = []

    def get_stream_source(self):
        return self._source

    def get_duration(self):
        return self._duration

    def get_active_trim_window(self):
        return (0.0, 0.0)

    def as_trimmed(self, start_time=None, duration=None, strict_duration=True):
        self.as_trimmed_calls.append({"start_time": start_time, "duration": duration})
        # Mirror VideoFromComponents: None + float must not be reachable after the fix.
        if start_time is None or duration is None:
            raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'float'")
        if self._duration < start_time + duration and duration > 0.0:
            return None
        trimmed = FakeVideo(self._source, duration=duration or (self._duration - start_time))
        return trimmed

    def save_to(self, path, format=None, codec=None):
        self.save_calls.append(path)
        Path(path).write_bytes(b"exported")


def test_export_video_path_reuses_on_disk_when_untrimmed(tmp_path):
    source = tmp_path / "plate.mp4"
    source.write_bytes(b"on-disk")
    video = FakeVideo(str(source))
    assert nodes.export_video_path(video, tmp_path) == source
    assert video.as_trimmed_calls == []
    assert video.save_calls == []


def test_export_video_path_passes_float_start_when_only_duration_set(tmp_path):
    # Regression: start_at_timestamp=0 used to pass start_time=None while duration
    # was a float; as_trimmed then did start_time + duration and TypeErrored.
    video = FakeVideo(str(tmp_path / "missing.mp4"))
    path = nodes.export_video_path(video, tmp_path, start_seconds=0.0, duration_seconds=3.5)
    assert path == tmp_path / "input_video.mp4"
    assert video.as_trimmed_calls == [{"start_time": 0.0, "duration": 3.5}]
    assert path.is_file()


def test_export_video_path_passes_float_duration_when_only_start_set(tmp_path):
    video = FakeVideo(str(tmp_path / "missing.mp4"))
    path = nodes.export_video_path(video, tmp_path, start_seconds=1.25, duration_seconds=0.0)
    assert path == tmp_path / "input_video.mp4"
    assert video.as_trimmed_calls == [{"start_time": 1.25, "duration": 0.0}]


def test_export_video_path_clamps_duration_past_eof_to_full_clip(tmp_path):
    """Requesting longer than the source clamps to the full clip and skips trim."""
    source = tmp_path / "short.mp4"
    source.write_bytes(b"short-plate")
    video = FakeVideo(str(source), duration=2.0)
    assert nodes.export_video_path(video, tmp_path, start_seconds=0.0, duration_seconds=5.0) == source
    assert video.as_trimmed_calls == []
    assert video.save_calls == []


def test_export_video_path_past_eof_start_raises_with_diagnostics(tmp_path):
    video = FakeVideo(duration=2.0)
    with pytest.raises(ValueError, match=r"past end of clip.*source_duration=2.*start=5"):
        nodes.export_video_path(video, tmp_path, start_seconds=5.0, duration_seconds=3.0)
    assert video.as_trimmed_calls == []


def test_export_video_path_clamps_partial_window(tmp_path):
    """start in range + duration past EOF → clamp length, still call as_trimmed."""
    video = FakeVideo(duration=2.0)
    path = nodes.export_video_path(video, tmp_path, start_seconds=1.5, duration_seconds=5.0)
    assert path == tmp_path / "input_video.mp4"
    assert video.as_trimmed_calls == [{"start_time": 1.5, "duration": 0.5}]
    assert path.is_file()


def test_export_video_path_double_trim_safe_when_duration_matches_prep(tmp_path):
    """Prep already output an 8s (or slightly short) plate; Video asks for 8s again.

    After clamp the window equals the full input, so we skip as_trimmed and reuse
    the on-disk path (avoids VideoFromFile strict_duration → None).
    """
    source = tmp_path / "prep_plate.mp4"
    source.write_bytes(b"prep-8s")
    video = FakeVideo(str(source), duration=7.958333)  # 191 frames @ 24 fps
    assert nodes.export_video_path(video, tmp_path, start_seconds=0.0, duration_seconds=8.0) == source
    assert video.as_trimmed_calls == []



def test_export_video_path_does_not_reuse_file_with_pending_trim(tmp_path):
    """VideoFromFile-style: on-disk bytes are the full source; trim is metadata.

    After clamp the window equals the *presented* duration, but reusing the
    path would send the untrimmed file to ninfer. Must re-encode.
    """
    source = tmp_path / "full_source.mp4"
    source.write_bytes(b"longer-than-presented")
    video = FakeVideo(str(source), duration=8.0)
    video.get_active_trim_window = lambda: (0.0, 8.0)
    path = nodes.export_video_path(video, tmp_path, start_seconds=0.0, duration_seconds=8.0)
    assert path == tmp_path / "input_video.mp4"
    assert video.as_trimmed_calls == []
    assert video.save_calls == [str(path)]
    assert path.read_bytes() == b"exported"


def test_export_video_path_duration_zero_default_skips_trim(tmp_path):
    """Workflow default: duration_seconds=0 on already-Prep'd VIDEO → full input."""
    source = tmp_path / "prep_plate.mp4"
    source.write_bytes(b"prep-full")
    video = FakeVideo(str(source), duration=8.0)
    assert nodes.export_video_path(video, tmp_path, start_seconds=0.0, duration_seconds=0.0) == source
    assert video.as_trimmed_calls == []


def test_long_plate_is_sampled_across_its_length_to_fit_the_vision_budget(tmp_path):
    import av
    from fractions import Fraction

    from comfy_api.latest import VideoFromComponents, VideoComponents

    count = 40
    images = torch.zeros(count, 32, 48, 3)
    images[:, 0, 0, 0] = torch.arange(count)
    video = VideoFromComponents(VideoComponents(images=images, audio=None, frame_rate=Fraction(24, 1)))
    path = nodes.fit_video_for_vision(video, tmp_path, max_frames=15)
    container = av.open(str(path))
    try:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()
    assert len(decoded) == 15
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
    # H3J Video Frames style: one IMAGE batch. H3J Image Bin style: OUTPUT_IS_LIST.
    sources = {
        "image_list": [torch.zeros(3, 40, 30, 3), torch.ones(1, 20, 10, 3)],
        "image0": torch.full((1, 8, 8, 3), 0.5),
    }
    path, has_images = nodes.write_messages(tmp_path, "frames", "", sources, 0, 32768)
    content = json.loads(path.read_text(encoding="utf-8"))[0]["content"]
    assert has_images and len(content) == 6 and content[-1]["text"] == "frames"
    schema_inputs = {item.id: item for item in nodes.Ninferencing.define_schema().inputs}
    assert schema_inputs["image_list"].optional


def test_image_resize_is_only_opt_in(tmp_path):
    path, _ = nodes.write_messages(tmp_path, "", "", {"image0": torch.zeros(1, 300, 200, 1)}, 150, 32768)
    source = json.loads(path.read_text())[0]["content"][0]["image_url"]["url"]
    with Image.open(source) as image:
        assert image.size == (100, 150)


def test_schemas_are_valid_and_have_no_required_image():
    for node in (nodes.Ninferencing, nodes.NinferencingVideo, nodes.NinferencingSettings):
        schema = node.define_schema()
        schema.finalize()
        schema.validate()
    main_inputs = {item.id: item for item in nodes.Ninferencing.define_schema().inputs}
    video_inputs = {item.id: item for item in nodes.NinferencingVideo.define_schema().inputs}
    settings_inputs = {item.id: item for item in nodes.NinferencingSettings.define_schema().inputs}
    images = main_inputs["images"]
    assert images.optional and images.template.min == 0
    assert main_inputs["context_length"].default == 32768
    assert video_inputs["context_length"].default == runtime.VIDEO_SAFE_CONTEXT
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


def test_ninfer_all_adds_gdn_and_vision_overlay():
    argv = command(flags=FLAGS_ALL, help_text=HELP_ALL, images=True)
    assert "--gdn-state-fp16" in argv
    assert "--vision" in argv
    assert argv[argv.index("--vision-residency") + 1] == "overlay"
    assert argv[argv.index("--vision-max-merged") + 1] == "12288"
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
        "images.image0": ["image-node", 0],
        "image_list": ["video-frames", 0], "settings": ["settings", 0],
    }, nodes.Ninferencing, "generate")
    assert missing.keys() == {"settings", "images.image0", "image_list"}
    inputs["images.image0"] = [torch.zeros(2, 2, 3, 3), torch.zeros(1, 3, 4, 3)]
    # OUTPUT_IS_LIST of single-frame tensors (e.g. H3J Image Bin image_list)
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
        return "one answer for every image"

    monkeypatch.setattr(nodes, "generate", complete)
    results = asyncio.run(comfy_execution._async_map_node_over_list(
        "test", "generate", nodes.Ninferencing, inputs, "execute", v3_data=v3_data,
    ))
    assert results[0].result == ("one answer for every image",)
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
    text = runtime.generate(argv, 5, lambda: None, None)
    assert text == "rewritten prompt"
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
    text = runtime.generate(argv, 5, lambda: None, None)
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
    text = runtime.generate(argv, 5, lambda: None, None)
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
    assert runtime.generate(argv, 5, lambda: None, None) == "ok"
    assert calls[1][calls[1].index("--kv-capacity") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
    assert calls[1][calls[1].index("--max-context") + 1] == str(runtime.IMAGE_SAFE_CONTEXT)
