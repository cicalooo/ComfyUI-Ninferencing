# Ninferencing

Local NInfer text generation for ComfyUI, built around a short-lived native CLI process. Five nodes: **Ninferencing** (image-default rewrite), **Ninferencing Video** (native `video_url` rewrite), optional **Ninferencing Settings**, and two analysis views for a MiniMax H3 swap — **Ninferencing Image Analysis** and **Ninferencing Video Analysis**.

Text leaves through the node's output socket. There is no on-node preview widget.

Tuned for **RTX 3090 / 4090 (24 GB)**. Other GPUs: [ninfer-all](https://github.com/iamwavecut/ninfer-all), or [docs/rtx-6000.md](docs/rtx-6000.md) for RTX PRO 6000 Blackwell.

## Requirements

- ComfyUI with the V3 node API.
- A native NInfer CLI that matches the GPU.
- A compatible `.ninfer` artifact.
- `psutil` (already a ComfyUI dependency).

No extra Python packages.

## Install the CLI (Linux, RTX 30/40)

Prebuilt sm_86 binaries, no CUDA compile:

```bash
curl -fsSL https://github.com/cicalooo/ninfer-linux/releases/latest/download/install.sh | bash
```

That installs to `/opt/ninfer/ninfer`. RTX 40 series uses the same archive. Details, runtime libraries, and a user-prefix install: [cicalooo/ninfer-linux](https://github.com/cicalooo/ninfer-linux).

RTX 50 and RTX PRO 6000 are not in that archive. Build or pull [ninfer-all](https://github.com/iamwavecut/ninfer-all) and set `NINFER_EXECUTABLE`.

Windows: put `ninfer.exe` on `PATH`, or at `C:\ninfer\ninfer.exe`, or set `NINFER_EXECUTABLE`.

## Install this pack

Clone into ComfyUI's `custom_nodes` and restart:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cicalooo/ComfyUI-Ninferencing Ninferencing
```

The folder name `Ninferencing` is what the tests import. ComfyUI itself loads the pack by path, so a Manager checkout named `ComfyUI-Ninferencing` still runs.

## Model

Any `.ninfer` file your CLI can load. v3 artifacts (Ternary Bonsai 2) need a ninfer-all / ninfer-linux CLI, not an older v1/v2-only build.

Public cards:

- [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3)
- [emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer](https://huggingface.co/emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer)
- Official Qwen artifacts listed in the [ninfer-all README](https://github.com/iamwavecut/ninfer-all#artifacts)

Put files in `ComfyUI/models/ninfer` or `~/models/ninfer`. Refresh node definitions after adding one.

## Discovery

The CLI is found in this order:

1. `NINFER_EXECUTABLE` — executable path or installation directory;
2. `NINFER_HOME` — installation directory;
3. `/opt/ninfer/ninfer` (ninfer-linux default);
4. `/opt/ninfer/current/ninfer`;
5. `/opt/ninfer-3090/current/ninfer` (older layout);
6. `~/.local/ninfer/ninfer`;
7. `ninfer` on `PATH`;
8. `C:\ninfer\ninfer.exe` on Windows when present.

A configured `ninfer-serve` path resolves to its sibling `ninfer` CLI.

The **library** model picker includes:

- `ComfyUI/models/ninfer`;
- `~/models/ninfer`;
- the discovered runtime's `models` and `artifacts` directories;
- `NINFER_MODELS_DIR`;
- folders registered as `ninfer` in `extra_model_paths.yaml`.

Use **file path** to select any other existing `.ninfer` artifact.

## Everyday use

- **prompt**: the request sent to the model.
- **system_prompt**: instructions, under advanced inputs. The default improves image prompts; edit or clear it for chat and other tasks.
- **max_tokens**: output budget, default 4096, including thinking when enabled.
- **context_length**: prompt, image, and output capacity, default 32,768 tokens.
- **image_max_side**: down-only longest-edge cap for every still, list item, batch item, or video frame. `0` picks the largest side the frame count allows.
- **image_list**: optional single socket for an IMAGE batch or ComfyUI image list (`OUTPUT_IS_LIST`).
- **images**: optional expanding sockets. All connected images and `image_list` are combined into one request.

Connecting an image enables `--vision`. The artifact must contain vision support.

Vision cost is the square of each image's patch grid, summed over every image and frame. The node downscales to the largest side that still fits:

| frames | automatic side | patches | vision tokens |
|---:|---:|---:|---:|
| 1 | 1696 | 11,236 | 2,809 |
| 2 | 1408 | 7,744 | 3,872 |
| 4 | 1184 | 5,476 | 5,476 |
| 8 | 992 | 3,844 | 7,688 |
| 12 | 896 | 3,136 | 9,408 |
| 16 | 832 | 2,704 | 10,816 |

`image_max_side` is an upper bound on top of that. Vision tokens also count against the prompt budget, so a large system prompt plus user text shrinks the automatic side further. More than 16 images or frames in one request is rejected by NInfer and is not truncated.

On a 24 GB card the 16-frame case still fits: vision weights add about 0.28 GiB and a 32,768-token INT8 KV cache is 1.50 GiB. BF16 KV doubles the cache and is the wrong choice on 24 GB.

### Ninferencing Video

`LoadVideo` → **Ninferencing Video** sends the clip as a native `video_url` (`--vision`). Prefer this over a sparse frame grid.

- Default `system_prompt` is video-oriented. The image node keeps its own default.
- Default `context_length` is **8192**. On a 24 GB card with a 27B artifact, native `--vision` raises the engine floor to about 3.33 GiB after weights. `--kv-capacity auto` then adds a fixed 1024 MiB headroom and does not fit. Video therefore passes an explicit `--kv-capacity` matching `context_length`.
- Optional `start_at_timestamp` / `duration_seconds` trim before export. `duration_seconds=0` uses the full input. Over-long requests are clamped; a start past EOF raises with diagnostics.
- Optional stills share the same request as extra `image_url` parts.
- Outputs: `text`, and `enhanced_video` (the same string, for a downstream video-prompt field).

A clip longer than the 16-item vision budget is not rejected. The export keeps at most 16 frames **spread across the whole clip** and writes them at a rate that preserves wall-clock length, so an 8 second clip stays 8 seconds. Decoding is streamed — the clip is never pulled into a tensor batch just to be thinned. `video_max_side` (default `0` on this node) caps the longest edge while decoding.

### Analysis nodes

**Ninferencing Image Analysis** sees only the replacement still and returns appearance facts. It rejects a frame batch. Default context is 12288 with `image_max_side` 1024.

**Ninferencing Video Analysis** sees only the source clip and returns camera, action, timing, contacts, and lighting as `scene:` + `timeline:` sections. Leave `duration_seconds` at 0 when the clip is already windowed. Default `video_max_side` is **768**. `[MM:SS.mmm]` stamps only map if the clip it receives keeps its original length, which the sampled export does.

Neither is an output node. Wire `text` yourself. Both accept **Ninferencing Settings** and should use the same `.ninfer` file.

## Settings

Image **Ninferencing** defaults to 32,768 tokens. **Ninferencing Video** defaults to 8,192. When the CLI advertises the flags, both select:

- `--kv-capacity auto` for text and stills;
- `--kv-capacity <context_length>` for native video (`auto` on Settings is treated as match-context on the video path);
- INT8 KV cache;
- `--gdn-state-fp16` when advertised;
- CUDA graphs off;
- thinking off;
- no speculative decoding;
- `--vision-residency overlay` and `--vision-max-merged 12288` on multimodal requests when advertised.

`kv_capacity` is `auto`, `match_context`, or a fixed count (`4096` / `8192` / `12288` / `16384` / `32768`). On 24 GB, prefer `match_context` or `8192` for video, and **context_length 12288–16384** with `match_context` for image rewrites that include stills. Leaving image context at 32K with `auto` can shortfall, and slamming to 8K then fails with `prepared prompt exceeds Engine context capacity`. Prefer `int8` KV.

Seed defaults to `0` and stays fixed. `device=-1` follows ComfyUI's active device. Timeout covers load and generation together.

`extra_args` is a JSON argument array, not a shell command:

```json
["--prefill-chunk", "256", "--stop", "END"]
```

Reserved prompt, output, and seed arguments cannot be replaced. Unsupported settings name the missing CLI flag.

## 24 GB memory behavior

A 27B artifact asks for about 17 GB of weights before engine, KV, and vision. Start from an otherwise idle 24 GB card. The default 32K context, automatic KV sizing, disabled CUDA graphs, and disabled thinking are the conservative **text / still-image** profile. The same numbers apply to a 3090 and a 4090.

**Native video + `--vision`:** after weights, only a few GiB remain. The vision engine floor alone is about 3.33 GiB, so auto KV headroom does not fit. Video defaults to 8K context with explicit `--kv-capacity 8192`. On capacity failure the node reclaims VRAM and retries once with a tighter explicit capacity. It does not drop `--vision` or switch to a frame grid.

Before each request the node stops orphaned `ninfer` / `ninfer-serve` processes, unloads Comfy-owned models, clears supported caches, waits for free VRAM, runs one CLI process, and waits for it to exit.

It can release VRAM owned by the current ComfyUI process. It cannot release memory held by another application. If `ninfer` reports insufficient free memory, stop the other GPU workload and retry.

On a capacity failure, **image / still** requests step through mid contexts (`16384` → `12288` → `8192`) so a long rewrite plus vision tokens still fit. **Native video** stays on the 8K profile. If a too-tight context then reports prepared-prompt overflow, one expand retry raises context and KV up the same ladder. Temporary inputs are removed after execution.

Each child is isolated. Windows uses a kill-on-close Job Object. Linux uses a separate process group.

## Tests

From this directory, with ComfyUI's Python, and the folder named `Ninferencing`:

```console
python -m pytest tests -q
```

CPU tensors and fake runtime responses only. No model load, no CUDA. For a checkout outside ComfyUI, set `COMFYUI_ROOT` to the ComfyUI directory.

## License

Apache-2.0. The NInfer runtime is separately licensed: [ninfer-all](https://github.com/iamwavecut/ninfer-all/blob/main/LICENSE).
