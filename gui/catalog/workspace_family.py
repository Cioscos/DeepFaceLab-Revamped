"""Family workspace: the two menu-only steps, neither a main.py invocation."""
from gui.catalog.model import (
    KIND_CLEAR, KIND_EBSYNTH, PROCESS_BATCH, StepDef,
)

STEPS = (
    StepDef(
        name="1) clear workspace",
        family="workspace",
        kind=KIND_CLEAR,
        process=PROCESS_BATCH,
        modifies=(
            "frame_src", "frame_dst", "faceset_src", "faceset_dst",
            "debug_dst", "modello", "merged", "merged_mask",
        ),
        optional=True,
    ),
    StepDef(
        name="10.misc) start EBSynth",
        family="workspace",
        kind=KIND_EBSYNTH,
        process=PROCESS_BATCH,
        optional=True,
    ),
)
