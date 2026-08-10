"""Family workspace: the two menu-only steps, neither a main.py invocation."""
from gui.catalog.model import (
    KIND_CLEAR, KIND_EBSYNTH, PROCESS_BATCH, StepDef,
)

STEPS = (
    StepDef(
        name="1) clear workspace",
        summary="Wipes data_src, data_dst and model back to empty folders. Keeps the source videos and any result.",
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
        summary="Opens the EBSynth application with the bundled sample project. Windows only.",
        family="workspace",
        kind=KIND_EBSYNTH,
        process=PROCESS_BATCH,
        optional=True,
    ),
)
