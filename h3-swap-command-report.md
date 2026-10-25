# H3 swap prompting

Freeform path: you write a plain request; Ninferencing sees it together with the media and rewrites it into one full-reference prompt for MiniMax H3. There is no command language and no command-builder node.

## System dialect

Two system dialects ship **inside the pack** (`swap_dialects.py`). Use one per Ninferencing call. Do not concatenate them.

| Job | Ninferencing H3 Swap System `dialect` |
| --- | --- |
| Character, object, or background swap; plate otherwise unchanged | `locked` |
| Same swap, plus changes you name (motion, look, speech) | `delta` |

No `models/markdown` files are needed. Wire **Ninferencing H3 Swap System** into **Ninferencing** `system_prompt`.

## User prompt

The `prompt` is the attachment map followed by your own words:

```text
ATTACHMENTS:
<Picture 1> = replacement still 1 of 1
<Video 1> = source plate (native video)
Frames sampled from <Video 1> are observations of <Video 1>, not Pictures. No other media is attached.
Swap the woman in the video for the person in the picture, face only. Keep the original sound.
```

**Ninferencing H3 Attachment Map** builds the `ATTACHMENTS:` block from the still count and whether the plate is connected (both are required). Join it with a plain text node using stock Concatenate Text (newline delimiter), map first.

Wire the clip into `video` (`<Video 1>`). Stills are `<Picture 1>` onward in the order Ninferencing sends them. Sampled frames are observations of `<Video 1>`, not extra Pictures.

What the dialects understand from plain words:

- Default transfer is face, hair, body and visible clothing; say "face only" to keep plate clothing.
- Several stills of one person are views of the same Subject.
- Object or background swaps change only that element.
- Audio: ask to keep the original sound to have H3 copy the plate track; otherwise no source-audio language is written (use this when sound is muxed later).
- `delta` only: name one motion, look, or speech change per axis; unmentioned axes stay with the plate. New words replace the plate speech; old dialogue survives only if you quote it to keep.

These lock the instruction. They do not make H3 pixel-perfect. Turns, occlusion, and small text can still drift.

## Validation helper (pure, no GPU)

`replace_brief.lint_swap_output(text, picture_count, has_video, *, audio_muxed_later=False)` checks a rewrite: `<Picture N>` above the attached count, any Video other than `<Video 1>`, and `+AUDIO` / `fully_copy` when audio is muxed later.
