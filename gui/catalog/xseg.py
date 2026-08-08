"""Family xseg: the thirteen "5.XSeg*)" mask-labeling and mask-training steps.

Two are pure PyQt5 sessions (the mask editor, `kind = "sessione"`), ten
resolve to `main.py xseg apply/fetch/remove/remove_labels`, and one
("5.XSeg) train") is a training session identical in shape to the
addestramento family but always training the fixed model name "XSeg"
(`XSegModel.__init__` passes `force_model_class_name='XSeg'`, so the "choose
a model name" prompt of `models/ModelBase.py` never fires here).

**"XSeg model face type" is left out of the two "Generic ... - apply" steps'
fields, but kept for the two "trained mask - apply" steps.** Both apply the
same prompt (`XSegUtil.apply_xseg`, `mainscripts/XSegUtil.py:25-37`) under the
same filesystem-state condition -- it only fires if
`<model-dir>/XSeg_data.dat` does not yet exist or lacks a saved
`options['face_type']`. For the Generic steps `<model-dir>` is the shipped
`{INTERNAL}/model_generic_xseg` asset, and reading it
(`pickle.load(...)['options']` -> `{'face_type': 'wf', 'pretrain': False,
'batch_size': 16}`, recorded in the workflow scheda) shows the key is always
already there, so the prompt provably never fires for those two steps -- it
is dropped, the same treatment extraction.py gives a prompt that "can no
longer fire" on a given command line. For the two "trained mask - apply"
steps `<model-dir>` is `{WORKSPACE}/model`, whose `XSeg_data.dat` only gets a
`face_type` once "5.XSeg) train" has run at least once -- genuinely
state-dependent the way extraction.py's re-inferred face type is, so the
field stays, without an `enabled_if` (the condition is file state, not
another field).

**The two "Press enter to continue." confirmations before
`remove`/`remove_labels` are modeled as fields, unlike the superficially
identical "Press enter to continue and overwrite." on faceset pack
(excluded in faceset_care.py).** The difference is the call itself:
`PackedFaceset.pack` uses raw `io.input(...)`, never routed through
DFL_ANSWERS_FILE; `XSegUtil.remove_xseg`/`remove_xseg_labels` use
`io.input_str('Press enter to continue.')`, which -- like every other
`input_str`/`input_bool`/`input_int` call -- IS looked up by `prompt_key`
against a preset answers file (`core/interact/interact.py::_preset`), even
though the caller discards the return value. Routing, not what the caller
does with the result, is what a form field needs.

**"Press enter in 2 seconds to override model settings." (`ModelBase.ask_override`,
only reachable on "5.XSeg) train" when resuming a training already in
progress) is the one prompt left out for a third, different reason: it is
not addressable at all.** `io.input_in_time` never calls `prompt_key` or
looks up its own text in the answers file -- it only checks whether *some*
answers file is configured at all (`self._preset_answers() is not None`) and,
if so, unconditionally returns `True`. There is no per-key value a field
could hold; the GUI, which always drives a step through an answers file,
would always resolve it to "yes, override" on a resumed run. "Restart
training?", "Face type", "Batch_size" and "Enable pretraining mode" are kept
as fields even though their conditions chain through this unaddressable
prompt (see the workflow scheda) -- the same filesystem/session-state
treatment as the face-type field above, not encoded as `enabled_if`.
"""
from gui.catalog.model import (
    FIELD_BOOL, FIELD_CHOICE, FIELD_INT, FIELD_TEXT, KIND_MAIN,
    PROCESS_PROMPT, PROCESS_SESSION, FieldDef, Invocation, StepDef,
)

_XSEG_GPU_INDEX = FieldDef(
    key="which-gpu-index-to-choose",
    label="GPU index",
    kind=FIELD_TEXT,
    default=None,
    help="Index of the device to use, or 'cpu'. Computed at runtime as the single best-scoring detected device.",
)

_XSEG_MODEL_FACE_TYPE = FieldDef(
    key="xseg-model-face-type",
    label="XSeg model face type",
    kind=FIELD_CHOICE,
    default="same",
    choices=("h", "mf", "f", "wf", "head", "same"),
    help="Specify face type of trained XSeg model. For example if XSeg model trained as WF, but faceset is HEAD, specify WF to apply xseg only on WF part of HEAD. Default is 'same'. Shown only if the model's XSeg_data.dat does not yet have a saved face type -- e.g. before any XSeg training has run.",
)

_FETCH_DELETE_ORIGINAL = FieldDef(
    key="delete-original-files",
    label="Delete original files?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown after the faces containing labeled XSeg polygons have been copied to the aligned_xseg sibling folder. If yes, the copied faces are moved rather than duplicated.",
)

_CONFIRM_REMOVE_LABELS = FieldDef(
    key="press-enter-to-continue",
    label="Press enter to continue.",
    kind=FIELD_TEXT,
    default=None,
    help="Confirms the warning 'LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES'. Removes only the hand-drawn polygons from the editor, not an applied trained mask.",
)

_CONFIRM_REMOVE_MASK = FieldDef(
    key="press-enter-to-continue",
    label="Press enter to continue.",
    kind=FIELD_TEXT,
    default=None,
    help="Confirms the warning 'APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES'. Removes only a mask applied by a trained XSeg model, not the hand-drawn editor polygons.",
)

_TRAIN_GPU_INDEXES = FieldDef(
    key="which-gpu-indexes-to-choose",
    label="GPU indexes",
    kind=FIELD_TEXT,
    default=None,
    help="Comma-separated device indexes, or 'cpu'. Computed at runtime as the devices matching the best detected one by name (suggest_best_multi_gpu) -- unlike the plural GPU field in extraction/faceset_care, this does not suggest every detected device: on heterogeneous cards only those sharing the best one's name are proposed together.",
)

_TRAIN_RESTART = FieldDef(
    key="restart-training",
    label="Restart training?",
    kind=FIELD_BOOL,
    default=False,
    help="Reset model weights and start training from scratch. Shown only when resuming a training already in progress.",
)

_TRAIN_FACE_TYPE = FieldDef(
    key="face-type",
    label="Face type",
    kind=FIELD_CHOICE,
    default="wf",
    choices=("h", "mf", "f", "wf", "head"),
    help="Half / mid face / full face / whole face / head. Choose the same as your deepfake model. Shown only on the first run of this model.",
)

_TRAIN_BATCH_SIZE = FieldDef(
    key="batch_size",
    label="Batch size",
    kind=FIELD_INT,
    default=4,
    valid_range=(2, 16),
    help="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually. Shown on the first run, or when overriding settings on a resumed run.",
)

_TRAIN_PRETRAIN = FieldDef(
    key="enable-pretraining-mode",
    label="Enable pretraining mode?",
    kind=FIELD_BOOL,
    default=False,
    help="Shown under the same condition as Batch size.",
)

STEPS = (
    StepDef(
        name="5.XSeg Generic) data_dst whole_face mask - apply",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "apply"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{INTERNAL}/model_generic_xseg",
            )),
        ),
        fields=(_XSEG_GPU_INDEX,),
        consumes=("xseg_generico",),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg Generic) data_src whole_face mask - apply",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "apply"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--model-dir", "{INTERNAL}/model_generic_xseg",
            )),
        ),
        fields=(_XSEG_GPU_INDEX,),
        consumes=("xseg_generico",),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_dst mask - edit",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("xseg", "editor"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
            )),
        ),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_dst mask - fetch",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "fetch"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
            )),
        ),
        fields=(_FETCH_DELETE_ORIGINAL,),
        consumes=("faceset_dst",),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_dst mask - remove",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "remove_labels"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
            )),
        ),
        fields=(_CONFIRM_REMOVE_LABELS,),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_dst trained mask - apply",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "apply"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{WORKSPACE}/model",
            )),
        ),
        fields=(_XSEG_MODEL_FACE_TYPE, _XSEG_GPU_INDEX),
        consumes=("modello",),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_dst trained mask - remove",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "remove"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
            )),
        ),
        fields=(_CONFIRM_REMOVE_MASK,),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_src mask - edit",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("xseg", "editor"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_src mask - fetch",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "fetch"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        fields=(_FETCH_DELETE_ORIGINAL,),
        consumes=("faceset_src",),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_src mask - remove",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "remove_labels"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        fields=(_CONFIRM_REMOVE_LABELS,),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_src trained mask - apply",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "apply"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--model-dir", "{WORKSPACE}/model",
            )),
        ),
        fields=(_XSEG_MODEL_FACE_TYPE, _XSEG_GPU_INDEX),
        consumes=("modello",),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) data_src trained mask - remove",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("xseg", "remove"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        fields=(_CONFIRM_REMOVE_MASK,),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.XSeg) train",
        family="xseg",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_dst/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "XSeg",
            )),
        ),
        fields=(
            _TRAIN_GPU_INDEXES, _TRAIN_RESTART, _TRAIN_FACE_TYPE,
            _TRAIN_BATCH_SIZE, _TRAIN_PRETRAIN,
        ),
        consumes=("faceset_src", "faceset_dst", "pretrain"),
        produces=("modello",),
        optional=True,
    ),
)
