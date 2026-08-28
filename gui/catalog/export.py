"""Family esportazione: the five "6) export * as dfm" steps.

All five resolve to `main.py exportdfm`, constructing the model with
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

**H1 inherits this unmodified from SAEHD** (through SAEHDX): same prompt,
same fields. **H2 overrides `on_initialize` wholesale and never reaches
SAEHD's "Export quantized?" prompt**; its own `export_dfm`
(`models/Model_H2/Model.py`) writes opset 12 with a second input
`morph_value:0`. So "6) export H2 as dfm" has **no fields**.
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
        summary="Exports the trained AMP model as a .dfm file for DeepFaceLive, with a morph_value input the others lack.",
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
        name="6) export H1 as dfm",
        summary="Exports the trained H1 model as a .dfm file, through the same unmodified export SAEHD uses.",
        family="esportazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("exportdfm",), args=(
                "--model-dir", "{WORKSPACE}/model",
                "--model", "H1",
            )),
        ),
        fields=(_EXPORT_QUANTIZED,),
        consumes=("modello",),
        modifies=("modello",),
        optional=True,
        needs_model_name=True,
    ),
    StepDef(
        name="6) export H2 as dfm",
        summary="Exports the trained H2 model as a .dfm file with a morph_value input, for DeepFaceLive model_type 2.",
        family="esportazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("exportdfm",), args=(
                "--model-dir", "{WORKSPACE}/model",
                "--model", "H2",
            )),
        ),
        fields=(),
        consumes=("modello",),
        modifies=("modello",),
        optional=True,
        needs_model_name=True,
    ),
    StepDef(
        name="6) export SAEHD as dfm",
        summary="Exports the trained SAEHD model as a .dfm file, the format DeepFaceLive loads for real-time swapping.",
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
        summary="Exports the trained SAEHDX model as a .dfm file, through the same unmodified export SAEHD uses.",
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
