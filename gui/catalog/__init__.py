"""Step catalog: the design-time translation of the user workflow.

Hand-derived from the workflow formalization; each step mirrors its entry
in scripts/commands.toml.
"""
from gui.catalog import (
    workspace_family, video_input, extraction, faceset_care, xseg, training,
    merging, video_output, export,
)

# The GUI launcher itself is a command in scripts/commands.toml but not a
# workflow step: it never appears in the catalog.
META_COMMANDS = {"DeepFaceLab GUI"}

_FAMILIES = (
    workspace_family, video_input, extraction, faceset_care, xseg, training,
    merging, video_output, export,
)


def all_steps():
    return tuple(step for module in _FAMILIES for step in module.STEPS)


def step_by_name(name):
    for step in all_steps():
        if step.name == name:
            return step
    raise KeyError(name)
