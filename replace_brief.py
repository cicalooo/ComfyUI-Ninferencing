"""Analysis prompts for the two oneshot H3 swap views.

Image analysis and video analysis stay separate: the still node sees only the
replacement appearance, the video node sees only the source plate. Nothing
here joins or rewrites their text, and nothing here calls a model.
"""
import re


IMAGE_ANALYSIS_SYSTEM = (
    "You describe one attached still for a MiniMax H3 replacement. "
    "The still is the replacement appearance. You do not see the source video.\n\n"
    "Reply with appearance facts only. No preamble, no title, no timeline, no camera, "
    "no background, no pose, and no motion.\n"
    "Cover the complete visible appearance: identity, body or object structure, hair, "
    "wardrobe, accessories, carried objects, colors, materials, and distinctive marks.\n"
    "Do not invent traits that are not visible. Do not mention Picture, Video, Subject, or Audio labels."
)
IMAGE_ANALYSIS_USER = (
    "Describe the complete visible appearance of the replacement subject in the attached still."
)
VIDEO_ANALYSIS_SYSTEM = (
    "You observe one attached source video for a MiniMax H3 replacement. "
    "You do not see the replacement still. Do not invent a replacement appearance, "
    "wardrobe, face, or object look.\n\n"
    "The user names the target to replace. Identify that target only well enough to track it. "
    "Then describe what must stay: camera, action, timing, contacts, occlusions, scene layout, "
    "and lighting. Do not restage the action.\n\n"
    "Reply with only these two sections and no preamble:\n\n"
    "scene:\n"
    "<one or two sentences of camera, scene, and lighting>\n\n"
    "timeline:\n"
    "[MM:SS.mmm-MM:SS.mmm]:\n"
    "[VISUAL]: <what the target and the camera do in that span>\n\n"
    "Cover the full clip from the start to the end. Use operational sentences: who or what moves, "
    "along what path, what they contact, and the physical result. "
    "Do not add Picture, Video, Subject, or Audio labels."
)

_LABEL = re.compile(r"<(Picture|Video|Subject|Audio)\s+(\d+)>", re.IGNORECASE)


def video_analysis_user_prompt(target: str) -> str:
    return f"Target to replace: {sanitize_analysis(target) or 'the named target'}"


def sanitize_analysis(text) -> str:
    """Drop media labels from model text so analysis cannot invent extra Pictures."""
    if text is None:
        return ""

    def repl(match):
        kind = match.group(1).lower()
        if kind == "picture":
            return "the attached still"
        if kind == "video":
            return "the source plate"
        if kind == "audio":
            return "the source soundtrack"
        return "the replacement"

    return _LABEL.sub(repl, str(text)).strip()


# --- H3 swap attachment map + output lint (Wave 1) -------------------------
# Improves on freeform "here are my files" prose: the map is derived from what
# Ninferencing actually sends, so stills and plate observations never blur.

ATTACHMENT_MAP_HEADER = "ATTACHMENTS:"


MAX_VIDEOS = 3


def _video_flags(has_video, video_count):
    """Normalize (has_video, video_count) to a list of per-video soundtrack flags."""
    if isinstance(has_video, (list, tuple)):
        flags = [bool(x) for x in has_video]
    else:
        n = int(video_count) if video_count is not None else (1 if has_video else 0)
        flags = [False] * n
    if len(flags) > MAX_VIDEOS:
        raise ValueError(f"H3 allows at most {MAX_VIDEOS} reference videos (<Video 1..{MAX_VIDEOS}>).")
    return flags


def build_attachment_map(picture_count, has_video, *, duration_s=None, note="",
                         video_count=None, video_soundtracks=None, audio_count=0) -> str:
    """Deterministic attachment brief for the H3 swap system dialects.

    Numbering mirrors CF Ref To Video / stock MiniMax H3 Reference to Video:
    pictures 1..N; for each video k in order, its soundtrack (if any) takes the
    next <Audio j> just before <Video k>; standalone audio continues the <Audio>
    numbering afterwards. Defaults (one plate, no audio) keep the old output.
    """
    picture_count = int(picture_count)
    if picture_count < 1:
        raise ValueError("H3 swap needs at least one replacement still (<Picture 1>).")
    flags = list(video_soundtracks) if video_soundtracks is not None else _video_flags(has_video, video_count)
    flags = _video_flags(flags, None)
    if not flags:
        raise ValueError("H3 swap needs the source plate connected as <Video 1>.")
    lines = [ATTACHMENT_MAP_HEADER]
    for n in range(1, picture_count + 1):
        lines.append(f"<Picture {n}> = replacement still {n} of {picture_count}")
    audio_j = 0
    for k, has_track in enumerate(flags, start=1):
        if has_track:
            audio_j += 1
            lines.append(f"<Audio {audio_j}> = soundtrack of <Video {k}>")
        plate = f"<Video {k}> = source plate (native video)" if k == 1 else f"<Video {k}> = reference video {k} (native video)"
        if k == 1 and duration_s is not None and float(duration_s) > 0:
            plate += f", {float(duration_s):g} s"
        lines.append(plate)
    for _ in range(int(audio_count or 0)):
        audio_j += 1
        lines.append(f"<Audio {audio_j}> = standalone reference audio")
    vids = ", ".join(f"<Video {k}>" for k in range(1, len(flags) + 1))
    lines.append(
        f"Frames sampled from {vids} are observations of that video, not Pictures. "
        "No other media is attached."
        if len(flags) > 1 else
        "Frames sampled from <Video 1> are observations of <Video 1>, not Pictures. "
        "No other media is attached."
    )
    note = " ".join(sanitize_analysis(note).split())
    if note:
        lines.append(f"NOTE: {note}")
    return "\n".join(lines)


_PICTURE = re.compile(r"<\s*Picture\s+(\d+)\s*>", re.IGNORECASE)
_VIDEO = re.compile(r"<?\bVideo\s+(\d+)\b>?", re.IGNORECASE)
_AUDIO = re.compile(r"<\s*Audio\s+(\d+)\s*>", re.IGNORECASE)
_FULLY_COPY = re.compile(r"fully[_ ]copy", re.IGNORECASE)


def lint_swap_output(text, picture_count, has_video, *, audio_muxed_later=False,
                     video_count=None, audio_total=None) -> list:
    """Check a rewritten swap prompt against what was attached. Returns issue strings.

    Pure text check; no command parsing. audio_muxed_later=True means sound is
    attached after generation, so the rewrite must not keep or copy plate audio.
    """
    text = str(text or "")
    issues = []
    picture_count = int(picture_count)
    for n in sorted({int(m) for m in _PICTURE.findall(text)}):
        if n < 1 or n > picture_count:
            issues.append(f"<Picture {n}> referenced but only {picture_count} still(s) attached.")
    video_total = int(video_count) if video_count is not None else (1 if has_video else 0)
    for n in sorted({int(m) for m in _VIDEO.findall(text)}):
        if video_total < 1:
            issues.append(f"Video {n} referenced but no video is attached.")
        elif n < 1 or n > min(video_total, MAX_VIDEOS):
            issues.append(f"Video {n} referenced; only <Video 1..{video_total}> attached.")
    if audio_total is not None:
        for n in sorted({int(m) for m in _AUDIO.findall(text)}):
            if n < 1 or n > int(audio_total):
                issues.append(f"<Audio {n}> referenced but only {int(audio_total)} audio label(s) exist.")
    if audio_muxed_later:
        if "+AUDIO" in text.upper():
            issues.append("+AUDIO present but audio is muxed later.")
        if _FULLY_COPY.search(text):
            issues.append("fully_copy audio language present but audio is muxed later.")
    return issues
