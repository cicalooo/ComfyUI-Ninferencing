# Ninferencing

Local NInfer text generation for ComfyUI, built around a short-lived native CLI process. Two nodes: **Ninferencing** (text, stills, or an optional native video) and optional **Ninferencing Settings**.

Text leaves through the node's output sockets. There is no on-node preview widget. `thinking` carries the model's reasoning trace with engine logs and the trailing statistics block removed; it is empty when thinking is off.

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
- **system_prompt**: instructions, under advanced inputs. Empty by default (no system message). Nothing is added behind your back.
- **max_tokens**: output budget, default 4096, including thinking when enabled.
- **context_length**: prompt, media, and output capacity, default 32,768 tokens. Connecting video does not change this. On 24 GB, lower it to 8192–16384 when a clip is attached; `kv_capacity auto` then matches that context.
- **video**: optional. Sent whole as a native `video_url`. Your loader owns trimming — every input is start-0 full input. Leave unconnected for stills or text.
- **video_duration**: the only duration input (advanced, FLOAT seconds, default `0` = probe the file). Wire your loader's `duration` / `duration_float` here when container metadata is unreliable. It never cuts the clip; it drives token planning.
- **video_max_side**: advanced. Applies only when `video` is connected. `0` = 512 (about 2.2k vision tokens for 15 s at 16:9); shrunk automatically to fit context_length/3.
- **thinking** / **thinking_budget** / **reasoning_loop**: advanced, on the main node. Defaults `off` / `0` / `off`. Override Settings when set.
- **speculation**: advanced, on the main node. Default `off`. Opt in to `mtp` / `dflash` / `dflash2` (draft range 1..15) only if your artifact has the draft weights. Overrides Settings.
- **image_max_side**: down-only longest-edge cap for stills and `image_list`. `0` picks the largest side the frame count allows. Native video uses `video_max_side`.
- **refs**: optional CF Image Bin cable (`CF_REFS`). Bin order is kept; each still is written as its own image. Do not also wire the bin's `image_list`.
- **image_list** / **images**: backup stills after `refs`. Advanced.

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

### Optional video

Wire `LoadVideo` into **Ninferencing** `video`. The clip is a native `video_url`. Two outputs: `text`, `thinking`.

Known issue: with Gemma-family artifacts, engine startup lines (e.g. `loading weights …`, `engine ready …`) have been seen leaking into `thinking`; Qwen runs come out clean. The parser strips the known startup shapes — if you see log lines in `thinking`, paste the exact lines so the pattern set can be extended.

On a 24 GB card with a 27B artifact, native `--vision` raises the engine floor enough that `--kv-capacity auto`'s 1024 MiB headroom does not fit. A connected clip therefore passes `--kv-capacity` equal to `context_length`. The widget itself stays at 32768. Lower it to 8192–16384 for a clip.

The clip is always sent whole — never trimmed or capped. Stills are sized against what remains after the planned native-video tokens (about 2.2k for 15 s at side 512), so a plate plus stills can no longer silently overflow the context. If the plan still exceeds `context_length`, the node fails fast with the token breakdown instead of an opaque engine error. `video_max_side` caps the longest edge while exporting.

Wire your loader's duration into `video_duration` so planning matches the plate even when container metadata is unreliable.

## Settings

**Ninferencing** defaults to 32,768 tokens. When the CLI advertises the flags, it selects:

- Qwen3.8-27B non-thinking sampler: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5. Thinking stays off. Switch Settings sampling to **model defaults** for the artifact's thinking sampler;
- `--kv-capacity auto` for text and stills;
- `--kv-capacity <context_length>` for native video (`auto` on Settings is treated as match-context on the video path);
- INT8 KV cache (`fp8` / `k8v4` / `nvfp4` available when your build advertises them — smaller KV for large contexts);
- `--prefill-chunk 1024` (larger chunks speed up prefill; 2048+ pairs with `--prefill-cublas`);
- `--gdn-state-fp16` when advertised;
- CUDA graphs off;
- thinking off on the main node (cap with `thinking_budget`, or `reasoning-loop stop`/`conclude` when supported; Settings holds the same controls);
- no speculative decoding on the main node (`draft_tokens` range 1..15 when you opt in to `mtp`/`dflash`/`dflash2`);
- `--vision-residency overlay` and `--vision-max-merged 8192` on multimodal requests when advertised.

`kv_capacity` is `auto`, `match_context`, or a fixed count (`4096` / `8192` / `12288` / `16384` / `32768`). On 24 GB, prefer `match_context` or `8192` for video, and **context_length 12288–16384** with `match_context` for image rewrites that include stills. Leaving image context at 32K with `auto` can shortfall, and slamming to 8K then fails with `prepared prompt exceeds Engine context capacity`. Prefer `int8` KV. For the heavy case (15 s plate + 4–6 stills at 64K), combine `fp8`/`k8v4` KV with `["--lm-head-q6"]` / `["--embedding-q6"]` in `extra_args` (quality tradeoff — measure first) and `["--prefill-cublas", "--prefill-chunk", "2048"]` for prefill speed.

Seed defaults to `0` and stays fixed. `device=-1` follows ComfyUI's active device. Timeout covers load and generation together.

`extra_args` is a JSON argument array, not a shell command:

```json
["--prefill-chunk", "256", "--stop", "END"]
```

Reserved prompt, output, and seed arguments cannot be replaced. Unsupported settings name the missing CLI flag.

## 24 GB memory behavior

A 27B artifact asks for about 17 GB of weights before engine, KV, and vision. Start from an otherwise idle 24 GB card. The default 32K context, automatic KV sizing, disabled CUDA graphs, and disabled thinking are the conservative **text / still-image** profile. The same numbers apply to a 3090 and a 4090.

**Native video + `--vision`:** after weights, only a few GiB remain. The vision engine floor alone is about 3.33 GiB, so auto KV headroom does not fit. A connected clip therefore passes `--kv-capacity` equal to `context_length` (the widget stays at 32768; lower it to 8192–16384 for a clip). On capacity failure the node reclaims VRAM and retries with a tighter explicit capacity. It does not drop `--vision` or switch to a frame grid. The intermediate export uses a fast x264 preset — it is a carrier ninfer re-samples, not the deliverable.

Before each request the node stops orphaned `ninfer` / `ninfer-serve` processes, unloads Comfy-owned models, clears supported caches, waits for free VRAM, runs one CLI process, and waits for it to exit.

It can release VRAM owned by the current ComfyUI process. It cannot release memory held by another application. If `ninfer` reports insufficient free memory, stop the other GPU workload and retry.

On a capacity failure, **image / still** requests step through mid contexts (`16384` → `12288` → `8192`) so a long rewrite plus vision tokens still fit. **Native video** requests tighten the same way. If a too-tight context then reports prepared-prompt overflow, one expand retry raises context and KV up the same ladder. Temporary inputs are removed after execution.

Each child is isolated. Windows uses a kill-on-close Job Object. Linux uses a separate process group.

## Tests

From this directory, with ComfyUI's Python, and the folder named `Ninferencing`:

```console
python -m pytest tests -q
```

CPU tensors and fake runtime responses only. No model load, no CUDA. For a checkout outside ComfyUI, set `COMFYUI_ROOT` to the ComfyUI directory.

## License

Apache-2.0. The NInfer runtime is separately licensed: [ninfer-all](https://github.com/iamwavecut/ninfer-all/blob/main/LICENSE).
