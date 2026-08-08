"""Family esportazione: the three "6) export * as dfm" steps.

All three resolve to `main.py exportdfm`, constructing the model with
`cpu_only=True` hardwired (`mainscripts/ExportDFM.py:22`) -- unlike train and
merge, `exportdfm` never asks a device-choice prompt, because
`ModelBase.__init__` only calls `nn.ask_choose_device_idxs` on the
`not cpu_only` branch. The "choose a model name" prompt of
`models/ModelBase.py` (shared with "6) train *"/"7) merge *", modeled there)
is skippable here: `main.py exportdfm --force-model-name`
reaches `ExportDFM.main`, which forwards it to the model constructor same as
train/merge -- which is what `needs_model_name=True` stands for: the GUI
supplies it as `--force-model-name` rather than as a form field.

The rest of `ModelBase`'s "first run or override" option battery
(autobackup, preview history, target iteration, the two random-flip
switches, and for SAEHDX its own `cudnn_benchmark`/`cuda_graph`/
`torch_compile`) is skipped for the ordinary case this step is meant for --
exporting an already-trained model, `iter != 0` -- because `exportdfm` never
passes `is_training=True`, so both `ask_override()` and `is_first_run()`
short-circuit false (the same structural argument the fusione family's
scheda already spells out for merge). None of that battery is modeled as
fields here; only "Export quantized?", the one prompt specific to exporting,
gated solely on `self.is_exporting` and therefore always shown regardless of
override/first-run state.
"""
from gui.catalog.model import FIELD_BOOL, KIND_MAIN, PROCESS_PROMPT, FieldDef, Invocation, StepDef

_EXPORT_QUANTIZED = FieldDef(
    key="export-quantized",
    label="Export quantized?",
    kind=FIELD_BOOL,
    default=False,
    help="Makes the exported model faster. If you have problems, disable this option.",
)

STEPS = (
    StepDef(
        name="6) export AMP as dfm",
        family="esportazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("exportdfm",), args=(
                "--model-dir", "{WORKSPACE}/model",
                "--model", "AMP",
            )),
        ),
        fields=(_EXPORT_QUANTIZED,),
        consumes=("modello",),
        modifies=("modello",),
        optional=True,
        needs_model_name=True,
    ),
    StepDef(
        name="6) export SAEHD as dfm",
        family="esportazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("exportdfm",), args=(
                "--model-dir", "{WORKSPACE}/model",
                "--model", "SAEHD",
            )),
        ),
        fields=(_EXPORT_QUANTIZED,),
        consumes=("modello",),
        modifies=("modello",),
        optional=True,
        needs_model_name=True,
    ),
    StepDef(
        name="6) export SAEHDX as dfm",
        family="esportazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("exportdfm",), args=(
                "--model-dir", "{WORKSPACE}/model",
                "--model", "SAEHDX",
            )),
        ),
        fields=(_EXPORT_QUANTIZED,),
        consumes=("modello",),
        modifies=("modello",),
        optional=True,
        needs_model_name=True,
    ),
)
