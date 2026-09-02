"""Family fusione: the five "7) merge *" steps (AMP, SAEHD, SAEHDX, H1, H2).

All five are `needs_model_name=True`, same reasoning as `training.py`: the
"choose one of saved models" block of `models/ModelBase.py` is shared,
unmodified code, always active (none of the five command lines passes
`--force-model-name`), and the GUI supplies the name as `--force-model-name`
instead. The GPU-index prompt right after it stays a field.

**None of `ModelBase`'s "first run or override" option battery is modeled
here -- the same exclusion `export.py` documents for `exportdfm`, and for
the identical structural reason.** `mainscripts/Merger.py:49` constructs the
model with `is_training=False`; `ModelBase.ask_override` is `self.
is_training and self.iter != 0 and io.input_in_time(...)` (`ModelBase.py:
299-300`), so its first term is always false here and the prompt never
fires, short-circuit, not merely a false outcome. `is_first_run()` is also
false for the normal case (an already-trained model). Every option gated on
"first run or override" -- SAEHD/AMP's resolution, archi, dims, GAN, etc.,
and SAEHDX's own `cudnn_benchmark`/`cuda_graph`/`torch_compile` -- therefore
does not appear in an ordinary merge session, and none of it is modeled as a
field. The one edge case where it structurally could (merging a model that
has never been trained, `iter == 0`) is documented in the workflow scheda
and treated the same way filesystem/runtime state is treated everywhere
else in this catalog: not encoded. `Model_SAEHDX` also skips its own fast
path in this mode (`Model_SAEHDX/Model.py:226`, guarded on
`self.is_training`) and falls back to the `Model_SAEHD.get_MergerConfig` it
inherits unmodified -- so "7) merge SAEHDX" has the exact same fields as
"7) merge SAEHD". The guard is what makes that true: without it, the fast
path would reach for a training-only attribute that merge never sets and
crash right after the GPU choice.

**Three prompts inside `MergerConfigMasked.ask_settings()` -- the mode
selector, the mask-mode selector and the (inherited) sharpen-mode selector
-- used to be called with an empty prompt string, and no longer are
(fixed 2026-08-08, `merger/MergerConfig.py:39,196,211`).** Each now carries
the text of the menu title already logged above it ("Choose sharpen mode",
"Choose mode", "Choose mask mode" -- see the workflow scheda's 2026-08-08
note for the exact before/after), so `prompt_key` gives each one its own
stable key (`choose-sharpen-mode`, `choose-mode`, `choose-mask-mode`)
instead of the three colliding on `prompt_key("")`. That collision was a
real limitation of the prompt protocol, not something this catalog could
route around: an answers file -- and the `values` dict a generated form
will evaluate `enabled_if` against -- can only hold one value per key. With
distinct keys, the sub-conditions internal to `ask_settings` that depend on
*which* of these three was chosen are now encoded exactly, no
approximation needed, because the field each one depends on is itself a
fixed, fully-enumerable `FIELD_CHOICE`: `'raw' not in self.mode` (gating
hist-match/erode/blur/motion-blur/color-transfer/denoise/bicubic/
color-degrade) becomes the AND of two exact `!=` conditions, since exactly
two of the seven mode values contain "raw"; `mode == 'hist-match'` (gating
masked-hist-match) and `sharpen_mode != 0` (gating blur/sharpen amount, `0`
being the choice labeled "none") are direct `=`/`!=` matches. All fifteen
fields of the battery still carry `enabled_if=("use-interactive-merger=n",)`
first -- unambiguous, exactly one field with that key in this form.

**"Number of workers?" passes `valid_range=[1, multiprocessing.cpu_count()]`
directly to `io.input_int`** (`Merger.py:77-78`) -- unlike most numeric
fields in this catalog, whose ranges come from a `np.clip` after an
unbounded read -- but the upper bound is the machine's own CPU count,
computed when the step runs, not a value this design-time catalog can carry
in a static tuple. Left as `valid_range=()`, same treatment as the "computed
at runtime" GPU-index fields elsewhere; the constraint is stated in the
field's help text instead.

**"Use interactive merger?" has a Colab-only alternate path** (`Merger.py:
72`: `False` unconditionally, no prompt, when `io.is_colab()`) -- not
modeled, same reasoning as the Colab branch of "Write preview history" in
`training.py`: this GUI never runs inside Colab.

**"Use saved session?" depends on this form's own "Use interactive merger?"
field (`=y`) *and* on filesystem state** (a `<ModelName>_merger_session.dat`
left by a previous interactive session that exited with Esc,
`InteractiveMergerSubprocessor.__init__`) -- the field part is encoded, the
filesystem part is not, the same split already used for "Choose image for
the preview history" in `training.py`.

AMP **and H2** ask "Morph factor" (`models/Model_AMP/Model.py:1249-1250`,
`models/Model_H2/Model.py`: same text, same 1.0 default, not persisted)
between the GPU-index prompt and "Use interactive merger?": a local blend
factor for this merge session only, never written to `self.options`, always
defaulting to 1.0 regardless of what was chosen during training.
`Model_SAEHD.get_MergerConfig` (`Model_SAEHD/Model.py:1264-1266`) and the
inherited-unmodified `Model_SAEHDX` never ask it -- and neither does H1,
which inherits the same unmodified method through `Model_SAEHDX`.

**H1's fields are `Model_SAEHDX`'s, unchanged**: `H1Model` never overrides
`get_MergerConfig`, so a merge session sees exactly what "7) merge SAEHDX"
does, and the seven supervisor prompts of training never appear here --
`ModelBase.ask_override` is false for an already-trained model constructed
with `is_training=False`, the same structural argument already made above
for SAEHD/SAEHDX/AMP's own first-run batteries.
"""
from gui.catalog.model import (
    FIELD_BOOL, FIELD_CHOICE, FIELD_FLOAT, FIELD_INT, FIELD_TEXT, KIND_MAIN,
    PROCESS_SESSION, FieldDef, Invocation, StepDef,
)

_GPU_INDEXES = FieldDef(
    key="which-gpu-indexes-to-choose",
    label="GPU indexes",
    kind=FIELD_TEXT,
    default=None,
    help="Comma-separated device indexes, or 'cpu'. Computed at runtime as the devices matching the best detected one by name (suggest_best_multi_gpu) -- unlike the plural GPU field in extraction/faceset_care, this does not suggest every detected device: on heterogeneous cards only those sharing the best one's name are proposed together.",
)

_MORPH_FACTOR_MERGE = FieldDef(
    key="morph-factor",
    label="Morph factor",
    kind=FIELD_FLOAT,
    default=1.0,
    valid_range=(0.0, 1.0),
    help="The src/dst blend used for this merge session only -- distinct from the persisted 'Morph factor.' asked during AMP training (range 0.1..0.5, default 0.5). Not written to disk. Shown always, before the interactive-merger choice.",
)

_USE_INTERACTIVE_MERGER = FieldDef(
    key="use-interactive-merger",
    label="Use interactive merger?",
    kind=FIELD_BOOL,
    default=True,
    help="If disabled, the whole non-interactive settings battery below is asked instead and no cv2 session window opens.",
)

_MERGE_MODE = FieldDef(
    key="choose-mode",
    label="Mode",
    kind=FIELD_CHOICE,
    default="overlay",
    choices=("original", "overlay", "hist-match", "seamless",
             "seamless-hist-match", "raw-rgb", "raw-predict"),
    choice_values=(0, 1, 2, 3, 4, 5, 6),
    choice_help=(
        "Passes the destination frame through untouched, no face swapped in. Useful to confirm the pipeline runs before tuning anything.",
        "Pastes the predicted face over the destination using the chosen mask. The default, and the one to start from.",
        "Like overlay, plus a histogram match to bring the predicted face's tones closer to the destination's.",
        "Like overlay, but blends the seam with OpenCV's seamless (Poisson) cloning instead of a plain mask edge.",
        "Seamless cloning plus the histogram match -- the heaviest combination of the two blending styles.",
        "Pastes the predicted face at full frame position without any mask blending, for inspecting the raw output or compositing it yourself downstream.",
        "Outputs only the predicted face crop itself, not warped back into the frame -- what the model actually produced, nothing else.",
    ),
    help="Non-interactive default; the interactive session's first-frame default is also 'overlay'.",
    enabled_if=("use-interactive-merger=n",),
)

_MASKED_HIST_MATCH = FieldDef(
    key="masked-hist-match",
    label="Masked hist match?",
    kind=FIELD_BOOL,
    default=True,
    help="Only asked by the console when Mode is 'hist-match'.",
    enabled_if=("use-interactive-merger=n", "choose-mode=hist-match"),
)

_HIST_MATCH_THRESHOLD = FieldDef(
    key="hist-match-threshold",
    label="Hist match threshold",
    kind=FIELD_INT,
    default=255,
    valid_range=(0, 255),
    help="Only asked by the console when Mode is 'hist-match' or 'seamless-hist-match'. Differs from the interactive session's first-frame default of 238.",
    enabled_if=("use-interactive-merger=n", "choose-mode=hist-match|seamless-hist-match"),
)

_MERGE_MASK_MODE = FieldDef(
    key="choose-mask-mode",
    label="Mask mode",
    kind=FIELD_CHOICE,
    default="dst",
    choices=("full", "dst", "learned-prd", "learned-dst",
             "learned-prd*learned-dst", "learned-prd+learned-dst",
             "XSeg-prd", "XSeg-dst", "XSeg-prd*XSeg-dst",
             "learned-prd*learned-dst*XSeg-prd*XSeg-dst"),
    choice_values=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    choice_help=(
        "No masking at all: everything inside the predicted face square is pasted, borders included.",
        "The destination frame's own landmark mask. Follows the real face outline, but ignores anything the model learned about occlusion.",
        "The mask the model learned to predict for the swapped face itself. The usual choice when the model was trained with masked training on.",
        "The mask the model learned to predict for the destination face. Tends to hug the real outline more closely than the swapped-face mask above.",
        "The two masks above kept where they agree: the most conservative combination of what the model learned.",
        "The two masks above joined: covers more than either alone, at the risk of including area neither predicted confidently.",
        "An XSeg mask computed on the predicted, swapped face using the trained XSeg model.",
        "An XSeg mask computed on the destination face using the trained XSeg model. Use it when you labelled the destination with the XSeg editor.",
        "The two XSeg masks above kept where they agree: the most conservative XSeg combination, and the one that survives hands and hair best.",
        "All four masks above -- both learned and both XSeg -- kept only where every one of them agrees. The tightest and safest of the ten, at the risk of cutting too much.",
    ),
    help="Not gated by Mode -- always asked when the battery runs. Differs from the interactive session's first-frame default of 'learned-prd*learned-dst'.",
    enabled_if=("use-interactive-merger=n",),
)

_ERODE_MASK_MODIFIER = FieldDef(
    key="choose-erode-mask-modifier",
    label="Choose erode mask modifier",
    kind=FIELD_INT,
    default=0,
    valid_range=(-400, 400),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_BLUR_MASK_MODIFIER = FieldDef(
    key="choose-blur-mask-modifier",
    label="Choose blur mask modifier",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 400),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_MOTION_BLUR_POWER = FieldDef(
    key="choose-motion-blur-power",
    label="Choose motion blur power",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 100),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'. In a real session this scales motion vectors computed from neighboring frames; here it stays 0, no vectors are computed for a single non-interactive answer.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_OUTPUT_FACE_SCALE = FieldDef(
    key="choose-output-face-scale-modifier",
    label="Choose output face scale modifier",
    kind=FIELD_INT,
    default=0,
    valid_range=(-50, 50),
    help="Not gated by Mode -- always asked when the battery runs.",
    enabled_if=("use-interactive-merger=n",),
)

_COLOR_TRANSFER_MODE = FieldDef(
    key="color-transfer-to-predicted-face",
    label="Color transfer to predicted face",
    kind=FIELD_CHOICE,
    default=None,
    choices=("rct", "lct", "mkl", "mkl-m", "idt", "idt-m", "sot-m", "mix-m"),
    choice_help=(
        "Reinhard colour transfer: fast, matches the average colour and contrast of the destination. The safe first try.",
        "Linear colour transfer: gentler than rct, keeps more of the predicted face's own tonality.",
        "Monge-Kantorovich transfer: matches the full colour distribution over the whole crop, better on a strong colour cast, slower than rct/lct.",
        "As mkl, but computed only from the pixels inside the face mask, ignoring the background around the crop.",
        "Iterative distribution transfer over the whole crop: a closer colour match than mkl, and slower.",
        "As idt, restricted to the face mask -- the closer match without letting the background skew it.",
        "Sliced optimal transport restricted to the face mask: the most thorough match here, and the slowest of the eight -- expect it to cost real time per frame.",
        "A blend: linear transfer for lightness, sliced optimal transport for colour, both restricted to the face mask. A compromise when no single mode looks right.",
    ),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'. The console default is an empty answer, which resolves to no color transfer ('None') -- a value this field's choices cannot express, since only the eight named modes are valid answers; leaving this field unanswered has the same effect.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_SHARPEN_MODE = FieldDef(
    key="choose-sharpen-mode",
    label="Sharpen mode",
    kind=FIELD_CHOICE,
    default="none",
    choices=("none", "box", "gaussian"),
    choice_values=(0, 1, 2),
    choice_help=(
        "No sharpening applied.",
        "Box-filter sharpening: cheap, can look a bit harsh on edges.",
        "Gaussian sharpening: smoother than box, usually the better default when sharpening is wanted.",
    ),
    help="Enhance details by applying sharpen filter.",
    enabled_if=("use-interactive-merger=n",),
)

_BLURSHARPEN_AMOUNT = FieldDef(
    key="choose-blursharpen-amount",
    label="Choose blur/sharpen amount",
    kind=FIELD_INT,
    default=0,
    valid_range=(-100, 100),
    help="Only asked by the console when Sharpen mode is not 'none'.",
    enabled_if=("use-interactive-merger=n", "choose-sharpen-mode!=none"),
)

_SUPER_RESOLUTION_POWER = FieldDef(
    key="choose-super-resolution-power",
    label="Choose super resolution power",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 100),
    help="Enhance details by applying superresolution network. Not gated by Mode -- always asked when the battery runs.",
    enabled_if=("use-interactive-merger=n",),
)

_IMAGE_DENOISE_POWER = FieldDef(
    key="choose-image-degrade-by-denoise-power",
    label="Choose image degrade by denoise power",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 500),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_BICUBIC_DEGRADE_POWER = FieldDef(
    key="choose-image-degrade-by-bicubic-rescale-power",
    label="Choose image degrade by bicubic rescale power",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 100),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_COLOR_DEGRADE_POWER = FieldDef(
    key="degrade-color-power-of-final-image",
    label="Degrade color power of final image",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 100),
    help="Only asked by the console when Mode is not 'raw-rgb'/'raw-predict'.",
    enabled_if=("use-interactive-merger=n", "choose-mode!=raw-rgb", "choose-mode!=raw-predict"),
)

_NUMBER_OF_WORKERS = FieldDef(
    key="number-of-workers",
    label="Number of workers?",
    kind=FIELD_INT,
    default=None,
    help="Specify the number of threads to process. A low value may affect performance. A high value may result in memory error. The value may not be greater than CPU cores. Default and upper bound are both computed at runtime as max(8, cpu_count()) and cpu_count() -- not a fixed range this catalog can state.",
)

_USE_SAVED_SESSION = FieldDef(
    key="use-saved-session",
    label="Use saved session?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown only if interactive and a previous interactive session for this model was left with Esc (a <ModelName>_merger_session.dat file exists) -- the filesystem part of the condition is not encoded, only the dependency on Use interactive merger? is.",
    enabled_if=("use-interactive-merger=y",),
)

_SETTINGS_BATTERY = (
    _MERGE_MODE, _MASKED_HIST_MATCH, _HIST_MATCH_THRESHOLD, _MERGE_MASK_MODE,
    _ERODE_MASK_MODIFIER, _BLUR_MASK_MODIFIER, _MOTION_BLUR_POWER,
    _OUTPUT_FACE_SCALE, _COLOR_TRANSFER_MODE, _SHARPEN_MODE,
    _BLURSHARPEN_AMOUNT, _SUPER_RESOLUTION_POWER, _IMAGE_DENOISE_POWER,
    _BICUBIC_DEGRADE_POWER, _COLOR_DEGRADE_POWER,
)

_AMP_MERGE_FIELDS = (
    (_GPU_INDEXES, _MORPH_FACTOR_MERGE, _USE_INTERACTIVE_MERGER)
    + _SETTINGS_BATTERY
    + (_NUMBER_OF_WORKERS, _USE_SAVED_SESSION)
)

_SAEHD_MERGE_FIELDS = (
    (_GPU_INDEXES, _USE_INTERACTIVE_MERGER)
    + _SETTINGS_BATTERY
    + (_NUMBER_OF_WORKERS, _USE_SAVED_SESSION)
)

# ---- Form sections ----------------------------------------------------
#
# Grouped by what a user is looking for: the general session controls
# first (which device, which mode, how many workers, whether to resume a
# saved session), then the three parts of the settings battery in the
# order they visually stack on the composited frame -- the mask that
# selects the area, the color that matches it to the plate, and the
# sharpening/denoising/super-resolution pass applied last. AMP's own
# `morph-factor` prompt joins the general controls, since it is asked
# before the mode is even chosen.

_SEZIONI_MERGE_AMP = (
    ("Output", ("which-gpu-indexes-to-choose", "morph-factor",
               "use-interactive-merger", "choose-mode",
               "choose-output-face-scale-modifier", "number-of-workers",
               "use-saved-session")),
    ("Mask", ("masked-hist-match", "hist-match-threshold", "choose-mask-mode",
             "choose-erode-mask-modifier", "choose-blur-mask-modifier",
             "choose-motion-blur-power")),
    ("Color", ("color-transfer-to-predicted-face",
              "choose-image-degrade-by-denoise-power",
              "choose-image-degrade-by-bicubic-rescale-power",
              "degrade-color-power-of-final-image")),
    ("Sharpening", ("choose-sharpen-mode", "choose-blursharpen-amount",
                    "choose-super-resolution-power")),
)

_SEZIONI_MERGE_SAEHD = (
    ("Output", ("which-gpu-indexes-to-choose", "use-interactive-merger",
               "choose-mode", "choose-output-face-scale-modifier",
               "number-of-workers", "use-saved-session")),
    ("Mask", ("masked-hist-match", "hist-match-threshold", "choose-mask-mode",
             "choose-erode-mask-modifier", "choose-blur-mask-modifier",
             "choose-motion-blur-power")),
    ("Color", ("color-transfer-to-predicted-face",
              "choose-image-degrade-by-denoise-power",
              "choose-image-degrade-by-bicubic-rescale-power",
              "degrade-color-power-of-final-image")),
    ("Sharpening", ("choose-sharpen-mode", "choose-blursharpen-amount",
                    "choose-super-resolution-power")),
)

STEPS = (
    StepDef(
        name="7) merge AMP",
        summary="Pastes the trained AMP face back onto every destination frame, with its own morph factor prompt.",
        family="fusione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("merge",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/merged",
                "--output-mask-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--aligned-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "AMP",
            )),
        ),
        fields=_AMP_MERGE_FIELDS,
        sections=_SEZIONI_MERGE_AMP,
        consumes=("frame_dst", "faceset_dst", "modello"),
        produces=("merged", "merged_mask"),
        needs_model_name=True,
    ),
    StepDef(
        name="7) merge H1",
        summary="Pastes the trained H1 face back onto every frame -- merges through SAEHD's path, the supervisors play no part here.",
        family="fusione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("merge",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/merged",
                "--output-mask-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--aligned-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "H1",
            )),
        ),
        fields=_SAEHD_MERGE_FIELDS,
        sections=_SEZIONI_MERGE_SAEHD,
        consumes=("frame_dst", "faceset_dst", "modello"),
        produces=("merged", "merged_mask"),
        needs_model_name=True,
    ),
    StepDef(
        name="7) merge H2",
        summary="Pastes the trained H2 face back onto every frame, with its own morph factor prompt: 1 = the src identity vector.",
        family="fusione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("merge",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/merged",
                "--output-mask-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--aligned-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "H2",
            )),
        ),
        fields=_AMP_MERGE_FIELDS,
        sections=_SEZIONI_MERGE_AMP,
        consumes=("frame_dst", "faceset_dst", "modello"),
        produces=("merged", "merged_mask"),
        needs_model_name=True,
    ),
    StepDef(
        name="7) merge SAEHD",
        summary="Pastes the trained face back onto every destination frame, one interactive session.",
        family="fusione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("merge",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/merged",
                "--output-mask-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--aligned-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "SAEHD",
            )),
        ),
        fields=_SAEHD_MERGE_FIELDS,
        sections=_SEZIONI_MERGE_SAEHD,
        consumes=("frame_dst", "faceset_dst", "modello"),
        produces=("merged", "merged_mask"),
        needs_model_name=True,
    ),
    StepDef(
        name="7) merge SAEHDX",
        summary="Pastes the trained face back onto every frame -- merges through the same path as SAEHD, none of the training speedup.",
        family="fusione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("merge",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/merged",
                "--output-mask-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--aligned-dir", "{WORKSPACE}/data_dst/aligned",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "SAEHDX",
            )),
        ),
        fields=_SAEHD_MERGE_FIELDS,
        sections=_SEZIONI_MERGE_SAEHD,
        consumes=("frame_dst", "faceset_dst", "modello"),
        produces=("merged", "merged_mask"),
        needs_model_name=True,
    ),
)
