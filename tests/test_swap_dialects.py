import re

import pytest

from Ninferencing.swap_dialects import DIALECTS, swap_system_prompt
from Ninferencing.swap_nodes import NinferencingH3SwapSystem

HOOKS = ["ROLE", "INPUT SHAPE", "ATTACHMENTS", "READING THE REQUEST", "IDENTITY RULES",
         "AUDIO RULES", "HONEST LIMITS", "OUTPUT", "observations of <Video 1>", "outgoing performer's face"]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_dialect_contracts(dialect):
    text = swap_system_prompt(dialect)
    for hook in HOOKS:
        assert hook in text, hook
    assert f"MODE: {dialect.upper()}" in text
    lowered = text.lower()
    assert "pixel-perfect" not in lowered and "pixel perfect" not in lowered
    assert not re.search(r"<Video\s+[2-9]>", text)
    # Plain user prompt: no command language assumed.
    for token in ("REPLACE ", "+AUDIO", "::", "MOTION:", "LOOK:", "SAY:", "KEEP SAY"):
        assert token not in text, token


def test_delta_documents_change_axes_locked_does_not():
    delta = swap_system_prompt("delta")
    for axis in ("- Motion:", "- Look:", "- Speech:"):
        assert axis in delta
        assert axis not in swap_system_prompt("locked")


def test_unknown_dialect_and_node():
    with pytest.raises(ValueError):
        swap_system_prompt("merge")
    schema = NinferencingH3SwapSystem.define_schema()
    schema.finalize()
    schema.validate()
    assert NinferencingH3SwapSystem.execute("delta")[0] == swap_system_prompt("delta")
