# Notes

## Thinking output (stderr framing)

- ninfer streams the answer to stdout and the reasoning to stderr with no tags: the `</think>` close marker is consumed as the channel boundary and never published (verified against upstream `ninfer-all @ 7a89828`: `apps/cli/main.cpp` `StreamingSink`, `src/models/qwen3_5/frontend/output_session.cpp`).
- The node keeps the whole stderr (`tail_limit=None`), cuts the trailing summary block at the last exact `phase       detail                      elapsed/progress` header line, drops `trace:/debug:/warning:/error:/fatal:` / `media | ` / `engine | ` / `ninfer: ` / `ninfer logging failure: ` log lines everywhere and the closed set of unprefixed startup lines (`build …`, `starting engine`, `loading weights …`, `engine ready …`, …) from the leading block, and returns the rest as `thinking`.
- Startup progress lines come in two shapes: persistent non-TTY renders bar-less (`  loading weights 97.2% | …`), interactive TTY renders `[==>]` bars. A bar-less line once broke the leading-block strip and leaked the lines after it; regression-tested.
- Deliberately fail-open: an unknown future log line leaks into `thinking` rather than dropping trace text. A model line starting with e.g. `error: ` is indistinguishable from a log record and is dropped.

## Video FPS (removed)

- The current engine, ninfer-all `63428ac` (`/opt/ninfer-3090/current`,
  release `v0.12.0-sm86.63428ac`), has **no `--video-fps`**. Video is sampled at the artifact's fixed 2 fps.
- ComfyUI-Ninferencing no longer exposes a `video_fps` option: every native clip is sent whole and ninfer
  samples it at its baked 2 fps with true timestamps. The older `f009850` build
  (e7b7e66 + `patches/0001-video-fps.patch`) and its 3–6 fps range are unsupported; see
  [cicalooo/ninfer-linux](https://github.com/cicalooo/ninfer-linux#building-the-video-fps-engine-variant)
  if you still need that variant for other tooling.

## Video duration (single lever)

- The loader owns all trimming: every input is treated as start-0 full input. The node never trims.
- `video_duration` (FLOAT seconds, default `0` = probe the file) is the only duration input. Wire your
  loader's `duration` / `duration_float` straight in when container metadata
  is unreliable; it drives vision-token planning.

## Gemma 4 A4B

- Family detection: filename (`gemma4` / `gemma-4`) or a 256 KiB peek for `Gemma4ForCausalLM` /
  `gemma4_gguf` / `Gemma4ForConditionalGeneration`. Defaults stay Qwen otherwise.
- Auto KV picks `rk8v4` when the build advertises it (3090 recommendation in docs/gemma4.md);
  Qwen/Bonsai still get INT8.
- Speculation: Gemma ships MTP. `off` on the main node (and Settings) means no `--spec` flag;
  the engine may still use its built-in drafter default. Choosing `dflash` / `dflash2` for Gemma
  is remapped to `mtp` with a warning.
- Thinking budgets are ignored (no early-close control). Auto still sends `--no-thinking`.
- Video: Gemma samples up to 32 frames evenly (70 soft tokens/frame); stills use ~280 soft tokens
  each. The Qwen 2 fps temporal-pair plan is unchanged for Qwen artifacts.

