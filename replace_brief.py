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
