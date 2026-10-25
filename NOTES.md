# Notes

## Video FPS (status / possible phase-out)

- **Status (25 Oct 2026):** the current engine, ninfer-all `63428ac` (`/opt/ninfer-3090/current`,
  release `v0.12.0-sm86.63428ac`), has **no `--video-fps`**. Video is sampled at the artifact's fixed 2 fps.
- **ComfyUI-Ninferencing:** its `video_fps` option (2–6) needs the older `f009850` build
  (e7b7e66 + `patches/0001-video-fps.patch`, release `v0.12.0-sm86.f009850`). At 2 (the default)
  the node omits the flag, so it works on 63428ac. Values 3–6 make the node stop with
  "this ninfer build does not advertise --video-fps".
- **To get 3–6 fps back:** rebuild the video-fps variant (see "Building the video-fps engine variant" in the [cicalooo/ninfer-linux README](https://github.com/cicalooo/ninfer-linux#building-the-video-fps-engine-variant)),
  or roll back to the old build:
  `sudo ln -sfn /opt/ninfer-3090/releases/ninfer-all-sm86-f009850 /opt/ninfer-3090/current`
  (return with `sudo ln -sfn /opt/ninfer-3090/releases/ninfer-all-sm86-63428ac /opt/ninfer-3090/current`).
- **Undecided:** the `video_fps` option and the patch may be phased out later. Nothing is removed yet.
