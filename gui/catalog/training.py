"""Family addestramento: the six "6) train *" steps (SAEHD, SAEHDX, AMP, AMP SRC-SRC, H1, H2).

All six are `needs_model_name=True`: `models.ModelBase.__init__`'s "choose one
of saved models, or enter a name" block (prompt key `""`, and its
"No saved models found..." sibling) is always active for these steps -- none
of the six command lines passes `--force-model-name` -- but the GUI supplies
the name as `--force-model-name` instead of driving that console block, the
same treatment `export.py` gives the identical prompt shared with "6) export
* as dfm". The GPU-index prompt right after it is a different story: it is
not skipped by anything the GUI does, so it stays a field, same key and help
text as the training step in `xseg.py` (`nn.ask_choose_device_idxs`,
`suggest_best_multi_gpu=True`).

`ModelBase.ask_override` ("Press enter in 2 seconds to override model
settings.") is left out for the same reason it is in `xseg.py`: `io.
input_in_time` never calls `prompt_key`, so there is no key an answers file
could drive. Every option gated on "first run or override" is still modeled
as a field -- always enabled, per the rule that runtime state (resuming vs.
first run) is not encoded as `enabled_if`, only dependencies on *other fields
of the same form* are. Two prompts inside `ask_write_preview_history`
fork on the runtime backend: the Windows chooser ("Choose image for the
preview history") is kept as a field, gated on the sibling
`write-preview-history` field being on -- the Windows-only part of its
condition is state, not a field, and is left out the same way; the Colab
variant ("Randomly choose new image for preview history", a different prompt
text entirely) is dropped, since this GUI never runs inside Colab.

**"'True face' power." is gated by a substring test, not an exact value.**
The real condition is `'df' in self.options['archi']`
(`models/Model_SAEHD/Model.py:755`), true for `df`, `df-u`, `df-ud`, `df-d`,
`df-t`, `df-c` and every other suffix combination the validation loop
accepts (`Model.py:662-693`) -- `archi` is free text, not a fixed enum, so no
finite pipe-separated list of exact values would cover it. Written as
`ae-architecture~=df` -- the `~=` ("contains") operator of `FieldDef.
enabled_if` (`gui/catalog/model.py`), added for exactly this case.

`Model_SAEHDX.on_initialize_options` calls `super().on_initialize_options()`
first (`models/Model_SAEHDX/Model.py:142`), so "6) train SAEHDX" carries every
SAEHD field unchanged plus its own three, all first-run-only booleans with no
cross-field condition. `Model_AMP` does not subclass `Model_SAEHD`
(`models/Model_AMP/Model.py:756`) and has a materially smaller, partly
differently-worded option set -- the full field-by-field diff is transcribed
below. AMP never asks "Enable pretraining mode": `self.pretrain` is never
assigned in that model (pre-existing behavior, inherited unchanged from
before this port, not something the port introduced), so neither AMP step
declares `pretrain` as a consumed artifact and neither has that field.
"6) train AMP SRC-SRC"
reuses `Model_AMP` unchanged; only the invocation's `--training-data-dst-dir`
differs (pointed at `data_src/aligned` -- the fork's manual substitute for a
pretraining mode this model does not have).

`` `use_fp16` `` is not modeled: both the option's default-value line and the
prompt that would use it are commented out in `Model_SAEHD/Model.py`
(lines 612 and 651) -- the option has never been ported and the prompt never
fires.

**H1 is SAEHDX plus seven supervisor prompts, nothing removed.**
`H1Model.on_initialize_options` calls `super().on_initialize_options()` first
(`models/Model_H1/Model.py`), so "6) train H1" carries every SAEHDX field
unchanged, then its own seven, all first-run-only. At zero power on all
seven, H1 *is* SAEHDX -- same weight files, same `.dfm`.

**H2 does not call `super().on_initialize_options()`: its prompt sequence is
its own**, built from scratch rather than shared. No "AE architecture" (it is
always liae-udt), no GAN/style/pretrain fields. The first two prompts are the
graft: "Graft from model" is a free-text field, not a fixed-choice one --
the list of valid sources is computed at runtime from the `*_data.dat` files
already in the chosen model dir, which a design-time catalog cannot
enumerate. A name that resolves to nothing stops the run with the list of
what is actually there (`models/Model_H2/Model.py`), rather than silently
falling back to "train from scratch" the way an answers-file `valid_list`
would. With a graft source given, the six size fields (resolution, face
type, the four dimension counts) are fixed by the source's own weights and
never asked -- `enabled_if=("graft-from-model=",)` models exactly that:
enabled only while the field above it is empty. The seven supervisor prompts
are shared with H1's, textually, but not by default: H1 zeroes them, H2
defaults to a recipe measured on a real run -- two distinct `FieldDef`s per
prompt, since a catalog `FieldDef` carries one default.
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
    # Never saved as a model option: `device_config` is built from the
    # constructor's `force_gpu_idxs` and never round-trips through
    # `load_or_def_option`. No saved value to show.
    help="Comma-separated device indexes, or 'cpu'. Computed at runtime as the devices matching the best detected one by name (suggest_best_multi_gpu) -- unlike the plural GPU field in extraction/faceset_care, this does not suggest every detected device: on heterogeneous cards only those sharing the best one's name are proposed together.",
)

_AUTOBACKUP_HOUR = FieldDef(
    key="autobackup-every-n-hour",
    option="autobackup_hour",   # models/ModelBase.py:303
    label="Autobackup every N hour",
    kind=FIELD_INT,
    default=0,
    help="Autobackup model files with preview every N hour. Latest backup located in model/<>_autobackups/01. Shown on the first run, or when overriding settings on a resumed run.",
)

_WRITE_PREVIEW_HISTORY = FieldDef(
    key="write-preview-history",
    option="write_preview_history",   # models/ModelBase.py:307
    label="Write preview history",
    kind=FIELD_BOOL,
    default=False,
    help="Preview history will be writed to <ModelName>_history folder. Shown under the same condition as Autobackup every N hour.",
)

_CHOOSE_PREVIEW_IMAGE = FieldDef(
    key="choose-image-for-the-preview-history",
    label="Choose image for the preview history",
    kind=FIELD_BOOL,
    default=False,
    # Never saved as a model option: `choose_preview_history` is a plain
    # instance attribute, not part of `self.options`. No saved value to show.
    help="Shown only if Write preview history is enabled. On this backend the interactive chooser only exists on Windows; elsewhere (and inside Colab, which asks a differently-worded prompt not modeled here) it is not offered, and random preview samples are used instead.",
    enabled_if=("write-preview-history=y",),
)

_TARGET_ITERATION = FieldDef(
    key="target-iteration",
    option="target_iter",   # models/ModelBase.py:317
    label="Target iteration",
    kind=FIELD_INT,
    default=0,
    help="Shown under the same condition as Autobackup every N hour. 0 - no target, train indefinitely.",
)

_FLIP_SRC = FieldDef(
    key="flip-src-faces-randomly",
    option="random_src_flip",   # models/ModelBase.py:325
    label="Flip SRC faces randomly",
    kind=FIELD_BOOL,
    default=False,
    help="Random horizontal flip SRC faceset. Covers more angles, but the face may look less naturally. Shown under the same condition as Autobackup every N hour.",
)

_FLIP_DST = FieldDef(
    key="flip-dst-faces-randomly",
    option="random_dst_flip",   # models/ModelBase.py:329 -- the key of
                                 # the prompt and the option's name do
                                 # not even resemble each other
    label="Flip DST faces randomly",
    kind=FIELD_BOOL,
    default=True,
    help="Random horizontal flip DST faceset. Makes generalization of src->dst better, if src random flip is not enabled. Shown under the same condition as Autobackup every N hour.",
)

_BATCH_SIZE_SAEHD = FieldDef(
    key="batch_size",
    option="batch_size",        # models/ModelBase.py:185
    label="Batch size",
    kind=FIELD_INT,
    default=8,
    help="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually. No range is enforced. Runtime-suggested default: 8 if the worst detected device has >=4 GB VRAM, else 4. Shown under the same condition as Autobackup every N hour.",
)

_AE_ARCHITECTURE = FieldDef(
    key="ae-architecture",
    option="archi",   # models/Model_SAEHD/Model.py:617
    label="AE architecture",
    kind=FIELD_TEXT,
    default="liae-ud",
    help="'df' keeps more identity-preserved face. 'liae' can fix overly different face shapes. '-u' increased likeness of the face. '-d' (experimental) doubling the resolution using the same computation cost. Examples: df, liae, df-d, df-ud, liae-ud, ... Validated in a loop: type must be 'df' or 'liae', optional suffix letters from u/d/t/c. Shown only on the first run.",
)

_AE_DIMS = FieldDef(
    key="autoencoder-dimensions",
    option="ae_dims",   # models/Model_SAEHD/Model.py:619, models/Model_AMP/Model.py:764
    label="AutoEncoder dimensions",
    kind=FIELD_INT,
    default=256,
    valid_range=(32, 1024),
    help="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU. Shown only on the first run.",
)

_ENCODER_DIMS = FieldDef(
    key="encoder-dimensions",
    option="e_dims",   # models/Model_SAEHD/Model.py:620, models/Model_AMP/Model.py:767
    label="Encoder dimensions",
    kind=FIELD_INT,
    default=64,
    valid_range=(16, 256),
    help="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU. Rounded to an even number. Shown only on the first run.",
)

_DECODER_DIMS = FieldDef(
    key="decoder-dimensions",
    option="d_dims",   # models/Model_SAEHD/Model.py:696, models/Model_AMP/Model.py:794
    label="Decoder dimensions",
    kind=FIELD_INT,
    default=64,
    valid_range=(16, 256),
    help="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU. Rounded to an even number. Shown only on the first run.",
)

_DECODER_MASK_DIMS = FieldDef(
    key="decoder-mask-dimensions",
    option="d_mask_dims",   # models/Model_SAEHD/Model.py:700, models/Model_AMP/Model.py:798
    label="Decoder mask dimensions",
    kind=FIELD_INT,
    default=22,
    valid_range=(16, 256),
    help="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality. Rounded to an even number. Default is Decoder dimensions / 3, rounded even. Shown only on the first run.",
)

_UNIFORM_YAW = FieldDef(
    key="uniform-yaw-distribution-of-samples",
    option="uniform_yaw",   # models/Model_SAEHD/Model.py:625, models/Model_AMP/Model.py:771
    label="Uniform yaw distribution of samples",
    kind=FIELD_BOOL,
    default=False,
    help="Helps to fix blurry side faces due to small amount of them in the faceset. Shown under the same condition as Autobackup every N hour.",
)

_BLUR_OUT_MASK = FieldDef(
    key="blur-out-mask",
    option="blur_out_mask",   # models/Model_SAEHD/Model.py:626, models/Model_AMP/Model.py:772
    label="Blur out mask",
    kind=FIELD_BOOL,
    default=False,
    help="Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required. Shown under the same condition as Autobackup every N hour.",
)

_LR_DROPOUT = FieldDef(
    key="use-learning-rate-dropout",
    option="lr_dropout",   # models/Model_SAEHD/Model.py:630, models/Model_AMP/Model.py:773
    label="Use learning rate dropout",
    kind=FIELD_CHOICE,
    default="n",
    choices=("n", "y", "cpu"),
    choice_help=(
        "Off. The plain choice, and the right one while the model is still improving quickly.",
        "On, on the GPU. Enable it only once the loss has stopped falling: it makes the model generalize instead of memorizing, at a measured cost of about 13% more time per iteration.",
        "On, on the CPU. Same effect, no extra VRAM, noticeably slower -- for when the card has nothing left.",
    ),
    help="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. n - disabled. y - enabled. cpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration. Shown under the same condition as Autobackup every N hour.",
)

_RANDOM_WARP = FieldDef(
    key="enable-random-warp-of-samples",
    option="random_warp",   # models/Model_SAEHD/Model.py:634, models/Model_AMP/Model.py:774
    label="Enable random warp of samples",
    kind=FIELD_BOOL,
    default=True,
    help="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations. Shown under the same condition as Autobackup every N hour.",
)

_GAN_PATCH_SIZE = FieldDef(
    key="gan-patch-size",
    option="gan_patch_size",   # models/Model_SAEHD/Model.py:723, models/Model_AMP/Model.py:822
    label="GAN patch size",
    kind=FIELD_INT,
    default=16,
    valid_range=(3, 640),
    help="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8. Default is resolution / 8. Shown only if GAN power is not 0.",
    enabled_if=("gan-power!=0.0",),
)

_GAN_DIMS = FieldDef(
    key="gan-dimensions",
    option="gan_dims",   # models/Model_SAEHD/Model.py:724, models/Model_AMP/Model.py:823
    label="GAN dimensions",
    kind=FIELD_INT,
    default=16,
    valid_range=(4, 512),
    help="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16. Shown only if GAN power is not 0.",
    enabled_if=("gan-power!=0.0",),
)

_CLIPGRAD = FieldDef(
    key="enable-gradient-clipping",
    option="clipgrad",   # models/Model_SAEHD/Model.py:640, models/Model_AMP/Model.py:776
    label="Enable gradient clipping",
    kind=FIELD_BOOL,
    default=False,
    help="Gradient clipping reduces chance of model collapse, sacrificing speed of training. Shown under the same condition as Autobackup every N hour.",
)

# ---- SAEHD/SAEHDX-only fields -------------------------------------------

_FACE_TYPE_SAEHD = FieldDef(
    key="face-type",
    option="face_type",   # models/Model_SAEHD/Model.py:614
    label="Face type",
    kind=FIELD_CHOICE,
    default="f",
    choices=("h", "mf", "f", "wf", "head"),
    choice_help=(
        "Half face: eyes, nose and mouth only. The fastest and the least natural at the borders.",
        "Middle half face: as half face, extended downwards over the chin.",
        "Full face: the face up to the eyebrows and the chin. The classic choice.",
        "Whole face: forehead and jaw included. The most natural merge, and the heaviest.",
        "Head: the whole head, hair included. Needs a faceset extracted as head, and far more iterations.",
    ),
    help="Half / mid face / full face / whole face / head. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset. Shown only on the first run.",
)

_RESOLUTION_SAEHD = FieldDef(
    key="resolution",
    option="resolution",   # models/Model_SAEHD/Model.py:613
    label="Resolution",
    kind=FIELD_INT,
    default=128,
    valid_range=(64, 640),
    help="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi. Shown only on the first run.",
)

_MASKED_TRAINING = FieldDef(
    key="masked-training",
    option="masked_training",   # models/Model_SAEHD/Model.py:623
    label="Masked training",
    kind=FIELD_BOOL,
    default=True,
    help="This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly. Shown only if Face type is 'wf' or 'head', on the first run or override.",
    enabled_if=("face-type=wf|head",),
)

_EYES_MOUTH_PRIO = FieldDef(
    key="eyes-and-mouth-priority",
    option="eyes_mouth_prio",   # models/Model_SAEHD/Model.py:624
    label="Eyes and mouth priority",
    kind=FIELD_BOOL,
    default=False,
    help="Helps to fix eye problems during training like \"alien eyes\" and wrong eyes direction. Also makes the detail of the teeth higher. Shown under the same condition as Autobackup every N hour.",
)

_MODELS_OPT_ON_GPU_SAEHD = FieldDef(
    key="place-models-and-optimizer-on-gpu",
    option="models_opt_on_gpu",   # models/Model_SAEHD/Model.py:615, models/Model_AMP/Model.py:762
    label="Place models and optimizer on GPU",
    kind=FIELD_BOOL,
    default=True,
    help="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. Answering n moves the whole model to the CPU: in this PyTorch port the weights and the computation cannot be split, so training then runs entirely on the CPU and is far slower. It frees all the VRAM, but it is not a way to fit bigger dimensions on the GPU. Outside training (merge and export) this option is ignored: use --cpu-only there. Shown under the same condition as Autobackup every N hour.",
)

_ADABELIEF = FieldDef(
    key="use-adabelief-optimizer",
    option="adabelief",   # models/Model_SAEHD/Model.py:628
    label="Use AdaBelief optimizer?",
    kind=FIELD_BOOL,
    default=True,
    help="Use AdaBelief optimizer. It requires more VRAM, but the accuracy and the generalization of the model is higher. Shown under the same condition as Autobackup every N hour.",
)

_RANDOM_HSV_POWER = FieldDef(
    key="random-huesaturationlight-intensity",
    option="random_hsv_power",   # models/Model_SAEHD/Model.py:635
    label="Random hue/saturation/light intensity",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 0.3),
    help="Random hue/saturation/light intensity applied to the src face set only at the input of the neural network. Stabilizes color perturbations during face swapping. Reduces the quality of the color transfer by selecting the closest one in the src faceset. Thus the src faceset must be diverse enough. Typical fine value is 0.05. Shown under the same condition as Autobackup every N hour.",
)

_GAN_POWER_SAEHD = FieldDef(
    key="gan-power",
    option="gan_power",   # models/Model_SAEHD/Model.py:722, models/Model_AMP/Model.py:821
    label="GAN power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 5.0),
    help="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with lr_dropout(on) and random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1. Shown under the same condition as Autobackup every N hour.",
)

_TRUE_FACE_POWER = FieldDef(
    key="true-face-power",
    option="true_face_power",   # models/Model_SAEHD/Model.py:636
    label="'True face' power.",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 1.0),
    help="Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01. Shown only if AE architecture is 'df' (any suffix), on the first run or override -- if it is 'liae', stays 0.0 with no prompt.",
    enabled_if=("ae-architecture~=df",),
)

_FACE_STYLE_POWER = FieldDef(
    key="face-style-power",
    option="face_style_power",   # models/Model_SAEHD/Model.py:637
    label="Face style power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 100.0),
    help="Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse. Shown under the same condition as Autobackup every N hour.",
)

_BG_STYLE_POWER = FieldDef(
    key="background-style-power",
    option="bg_style_power",   # models/Model_SAEHD/Model.py:638
    label="Background style power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 100.0),
    help="Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0. Shown under the same condition as Autobackup every N hour.",
)

_CT_MODE_SAEHD = FieldDef(
    key="color-transfer-for-src-faceset",
    option="ct_mode",   # models/Model_SAEHD/Model.py:639, models/Model_AMP/Model.py:775
    label="Color transfer for src faceset",
    kind=FIELD_CHOICE,
    default="none",
    choices=("none", "rct", "lct", "mkl", "idt", "sot"),
    choice_help=(
        "No color transfer: src samples keep their own colors as extracted.",
        "Reinhard color transfer: fast, matches the average color and contrast of dst. The usual first try.",
        "Linear color transfer: gentler than rct, keeps more of src's own tonality.",
        "Monge-Kantorovich transfer: matches the full color distribution of dst, better on a strong color cast, slower than rct/lct.",
        "Iterative distribution transfer: the closest color match of these, and the slowest.",
        "Sliced optimal transport: the most thorough match of local color, and the heaviest of the six.",
    ),
    help="Change color distribution of src samples close to dst samples. Try all modes to find the best. Shown under the same condition as Autobackup every N hour.",
)

_PRETRAIN = FieldDef(
    key="enable-pretraining-mode",
    option="pretrain",   # models/Model_SAEHD/Model.py:641
    label="Enable pretraining mode",
    kind=FIELD_BOOL,
    default=False,
    help="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly. Forces random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y. Shown under the same condition as Autobackup every N hour. Requires --pretraining-data-dir, always passed by this step's invocation.",
)

_CUDNN_BENCHMARK = FieldDef(
    key="enable-cudnnbenchmark",
    option="cudnn_benchmark",   # models/Model_SAEHDX/Model.py:150
    label="Enable cudnn.benchmark",
    kind=FIELD_BOOL,
    default=False,
    help="Lets cuDNN autotune the fastest convolution algorithm for the fixed shapes of this run. The extra speed on top of the other SAEHDX optimizations was too small to tell apart from measurement noise, while the extra VRAM was consistent and large. Off by default: on recent, VRAM-constrained cards the trade is usually not worth it. Shown only on the first run.",
)

_CUDA_GRAPH = FieldDef(
    key="enable-cuda-graph-capture",
    option="cuda_graph",   # models/Model_SAEHDX/Model.py:164
    label="Enable CUDA graph capture",
    kind=FIELD_BOOL,
    default=False,
    help="Records the training step once as a CUDA graph and replays it, instead of re-issuing every kernel from Python each iteration. It needs fixed input buffers, so it turns the batch prefetch off, and it only applies to a plain src_dst step: with GAN, true face power, learning rate dropout or clipgrad the capture is skipped and training goes on unchanged. Off by default. Shown only on the first run.",
)

_TORCH_COMPILE = FieldDef(
    key="enable-torchcompile",
    option="torch_compile",   # models/Model_SAEHDX/Model.py:180
    label="Enable torch.compile",
    kind=FIELD_BOOL,
    default=False,
    help="Compiles the training step ahead of time instead of interpreting it kernel by kernel every iteration. Verified over a 5000-iteration run: convergence indistinguishable from the uncompiled model, and an iteration takes about a quarter less. It buys speed with memory, mostly host RAM (peak resident memory nearly doubled here, +7.1 GiB held for the whole run) and 15% more allocated VRAM. Needs a working Triton backend and, on Windows, the MSVC compiler on PATH: where either is missing the option turns itself off and training goes on unchanged. The first iteration also costs the compilation, around half a minute. Not combined with the CUDA graph. Off by default. Shown only on the first run.",
)

_SAEHD_FIELDS = (
    _GPU_INDEXES, _AUTOBACKUP_HOUR, _WRITE_PREVIEW_HISTORY,
    _CHOOSE_PREVIEW_IMAGE, _TARGET_ITERATION, _FLIP_SRC, _FLIP_DST,
    _BATCH_SIZE_SAEHD, _RESOLUTION_SAEHD, _FACE_TYPE_SAEHD, _AE_ARCHITECTURE,
    _AE_DIMS, _ENCODER_DIMS, _DECODER_DIMS, _DECODER_MASK_DIMS,
    _MASKED_TRAINING, _EYES_MOUTH_PRIO, _UNIFORM_YAW, _BLUR_OUT_MASK,
    _MODELS_OPT_ON_GPU_SAEHD, _ADABELIEF, _LR_DROPOUT, _RANDOM_WARP,
    _RANDOM_HSV_POWER, _GAN_POWER_SAEHD, _GAN_PATCH_SIZE, _GAN_DIMS,
    _TRUE_FACE_POWER, _FACE_STYLE_POWER, _BG_STYLE_POWER, _CT_MODE_SAEHD,
    _CLIPGRAD, _PRETRAIN,
)

_SAEHDX_FIELDS = _SAEHD_FIELDS + (_CUDNN_BENCHMARK, _CUDA_GRAPH, _TORCH_COMPILE)

# ---- AMP-only fields ------------------------------------------------------

_RESOLUTION_AMP = FieldDef(
    key="resolution",
    option="resolution",   # models/Model_AMP/Model.py:760
    label="Resolution",
    kind=FIELD_INT,
    default=224,
    valid_range=(64, 640),
    help="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 32. Shown only on the first run.",
)

_FACE_TYPE_AMP = FieldDef(
    key="face-type",
    option="face_type",   # models/Model_AMP/Model.py:761
    label="Face type",
    kind=FIELD_CHOICE,
    default="wf",
    choices=("f", "wf", "head"),
    choice_help=(
        "Full face: the face up to the eyebrows and the chin. Lighter to train, less area covered.",
        "Whole face: forehead and jaw included, the more natural merge. The default for AMP.",
        "Head: the whole head, hair included. Needs a faceset extracted as head, and far more iterations.",
    ),
    help="whole face / head. Shown only on the first run.",
)

_INTER_DIMS = FieldDef(
    key="inter-dimensions",
    option="inter_dims",   # models/Model_AMP/Model.py:765
    label="Inter dimensions",
    kind=FIELD_INT,
    default=1024,
    valid_range=(32, 2048),
    help="Should be equal or more than AutoEncoder dimensions. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU. Shown only on the first run.",
)

_MORPH_FACTOR_TRAIN = FieldDef(
    key="morph-factor",
    # Saved as self.options['morph_factor'] (models/Model_AMP/Model.py:768),
    # but read back with self.options.get(...), not load_or_def_option --
    # the guard that checks a declared name really appears in a
    # load_or_def_option(...) call would reject it, so left undeclared
    # rather than pass a name the guard cannot confirm.
    label="Morph factor.",
    kind=FIELD_FLOAT,
    default=0.5,
    valid_range=(0.1, 0.5),
    help="Typical fine value is 0.5. Shown only on the first run. Unlike the other options above, a new model reusing an existing name does not inherit this value from the class-wide _default_options.dat file.",
)

_BATCH_SIZE_AMP = FieldDef(
    key="batch_size",
    option="batch_size",   # models/ModelBase.py:185
    label="Batch size",
    kind=FIELD_INT,
    default=8,
    help="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually. No range is enforced. Suggested default is fixed at 8 for this model, not computed from VRAM. Shown under the same condition as Autobackup every N hour.",
)

_MODELS_OPT_ON_GPU_AMP = FieldDef(
    key="place-models-and-optimizer-on-gpu",
    option="models_opt_on_gpu",   # models/Model_AMP/Model.py:762
    label="Place models and optimizer on GPU",
    kind=FIELD_BOOL,
    default=True,
    help="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions. This help text is still the TensorFlow-era wording, not corrected for this port the way SAEHD's equivalent was: here too, answering n moves the whole model to the CPU. Shown under the same condition as Autobackup every N hour.",
)

_GAN_POWER_AMP = FieldDef(
    key="gan-power",
    option="gan_power",   # models/Model_AMP/Model.py:821
    label="GAN power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 5.0),
    help="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1. Shown under the same condition as Autobackup every N hour.",
)

_CT_MODE_AMP = FieldDef(
    key="color-transfer-for-src-faceset",
    option="ct_mode",   # models/Model_AMP/Model.py:775
    label="Color transfer for src faceset",
    kind=FIELD_CHOICE,
    default="none",
    choices=("none", "rct", "lct", "mkl", "idt", "sot"),
    choice_help=(
        "No color transfer: src samples keep their own colors as extracted.",
        "Reinhard color transfer: fast, matches the average color and contrast of dst. The usual first try.",
        "Linear color transfer: gentler than rct, keeps more of src's own tonality -- fine in most cases if the src faceset is diverse enough.",
        "Monge-Kantorovich transfer: matches the full color distribution of dst, better on a strong color cast, slower than rct/lct.",
        "Iterative distribution transfer: the closest color match of these, and the slowest.",
        "Sliced optimal transport: the most thorough match of local color, and the heaviest of the six.",
    ),
    help="Change color distribution of src samples close to dst samples. If src faceset is deverse enough, then lct mode is fine in most cases. Shown under the same condition as Autobackup every N hour.",
)

_AMP_FIELDS = (
    _GPU_INDEXES, _AUTOBACKUP_HOUR, _WRITE_PREVIEW_HISTORY,
    _CHOOSE_PREVIEW_IMAGE, _TARGET_ITERATION, _FLIP_SRC, _FLIP_DST,
    _BATCH_SIZE_AMP, _RESOLUTION_AMP, _FACE_TYPE_AMP, _AE_DIMS, _INTER_DIMS,
    _ENCODER_DIMS, _DECODER_DIMS, _DECODER_MASK_DIMS, _MORPH_FACTOR_TRAIN,
    _UNIFORM_YAW, _BLUR_OUT_MASK, _LR_DROPOUT, _MODELS_OPT_ON_GPU_AMP,
    _RANDOM_WARP, _GAN_POWER_AMP, _GAN_PATCH_SIZE, _GAN_DIMS, _CT_MODE_AMP,
    _CLIPGRAD,
)

# ---- Form sections, for the four steps long enough to need them -----------
#
# Grouped by what a user is looking for, not by declaration order: first
# what to train and for how long, then the network's shape, then how the
# data is perturbed, then the two adversarial/style levers, then the
# performance switches. SAEHD/SAEHDX share everything through GAN; SAEHDX
# repeats the six sections with its three extra levers folded into
# Performance. AMP has a smaller, differently-shaped option set (its own
# `inter-dimensions`/`morph-factor`, no AdaBelief/HSV/pretrain, and none of
# the three style-power fields) and gets five sections, not six -- an empty
# "Style" section is not declared for it.

_SEZIONI_SAEHD = (
    ("Model", ("which-gpu-indexes-to-choose", "autobackup-every-n-hour",
               "write-preview-history", "choose-image-for-the-preview-history",
               "target-iteration", "batch_size", "resolution")),
    ("Architecture", ("face-type", "ae-architecture", "autoencoder-dimensions",
                      "encoder-dimensions", "decoder-dimensions",
                      "decoder-mask-dimensions", "masked-training",
                      "eyes-and-mouth-priority")),
    ("Augmentation", ("flip-src-faces-randomly", "flip-dst-faces-randomly",
                      "uniform-yaw-distribution-of-samples", "blur-out-mask",
                      "enable-random-warp-of-samples",
                      "random-huesaturationlight-intensity",
                      "color-transfer-for-src-faceset")),
    ("GAN", ("gan-power", "gan-patch-size", "gan-dimensions")),
    ("Style", ("true-face-power", "face-style-power", "background-style-power")),
    ("Performance", ("place-models-and-optimizer-on-gpu", "use-adabelief-optimizer",
                     "use-learning-rate-dropout", "enable-gradient-clipping",
                     "enable-pretraining-mode")),
)

_SEZIONI_SAEHDX = _SEZIONI_SAEHD[:-1] + (
    ("Performance", _SEZIONI_SAEHD[-1][1] + (
        "enable-cudnnbenchmark", "enable-cuda-graph-capture",
        "enable-torchcompile",
    )),
)

_SEZIONI_AMP = (
    ("Model", ("which-gpu-indexes-to-choose", "autobackup-every-n-hour",
               "write-preview-history", "choose-image-for-the-preview-history",
               "target-iteration", "batch_size", "resolution")),
    ("Architecture", ("face-type", "autoencoder-dimensions", "inter-dimensions",
                      "encoder-dimensions", "decoder-dimensions",
                      "decoder-mask-dimensions", "morph-factor")),
    ("Augmentation", ("flip-src-faces-randomly", "flip-dst-faces-randomly",
                      "uniform-yaw-distribution-of-samples", "blur-out-mask",
                      "enable-random-warp-of-samples",
                      "color-transfer-for-src-faceset")),
    ("GAN", ("gan-power", "gan-patch-size", "gan-dimensions")),
    ("Performance", ("use-learning-rate-dropout", "place-models-and-optimizer-on-gpu",
                     "enable-gradient-clipping")),
)

# ---- H1/H2: the supervisors ----------------------------------------------
#
# The same seven prompts in models/Model_H1/Model.py and models/Model_H2/
# Model.py, with different defaults: H1 at zero *is* SAEHDX (its
# definition); H2 defaults to a recipe measured on a real run. Distinct
# FieldDefs where the default differs, shared ones where it does not.

_HELP_ID_POWER = "Cosine loss between the AdaFace embedding of the swapped face and the mean embedding of the src faceset. 0 disables it. Turning any supervisor on downloads its weights on first use and switches CUDA graph capture off. Shown only on the first run."
_ID_POWER_H1 = FieldDef(key="identity-power", option="id_power", label="Identity power", kind=FIELD_FLOAT,
                        default=0.0, valid_range=(0.0, 10.0), help=_HELP_ID_POWER)
_ID_POWER_H2 = FieldDef(key="identity-power", option="id_power", label="Identity power", kind=FIELD_FLOAT,
                        default=2.0, valid_range=(0.0, 10.0),
                        help=_HELP_ID_POWER + " 2.0 is the measured recipe for H2.")
_HELP_IFSR = "L1 between AdaFace intermediate features of the swapped face and of the dst face: keeps pose, lighting and occlusions of dst. Alone it collapses the swap onto dst -- it is the companion of Identity power, not a lever of its own. 0 disables it. Shown only on the first run."
_IFSR_POWER_H1 = FieldDef(key="ifsr-power", option="ifsr_power", label="IFSR power", kind=FIELD_FLOAT,
                          default=0.0, valid_range=(0.0, 10.0), help=_HELP_IFSR)
_IFSR_POWER_H2 = FieldDef(key="ifsr-power", option="ifsr_power", label="IFSR power", kind=FIELD_FLOAT,
                          default=0.08, valid_range=(0.0, 10.0), help=_HELP_IFSR + " 0.08 is the measured recipe for H2.")
_HELP_BLEED = "Penalizes the swap when its AdaFace embedding drifts toward the mean embedding of the dst faceset beyond a fixed cosine margin. 0 disables it. Shown only on the first run."
_BLEED_POWER_H1 = FieldDef(key="bleed-power", option="bleed_power", label="Bleed power", kind=FIELD_FLOAT,
                           default=0.0, valid_range=(0.0, 10.0), help=_HELP_BLEED)
_BLEED_POWER_H2 = FieldDef(key="bleed-power", option="bleed_power", label="Bleed power", kind=FIELD_FLOAT,
                           default=1.0, valid_range=(0.0, 10.0), help=_HELP_BLEED + " 1.0 is the measured recipe for H2.")
_BLEED_PER_SAMPLE = FieldDef(key="bleed-per-sample", option="bleed_campione", label="Bleed per sample", kind=FIELD_BOOL,
                             default=False,
                             help="Bleed repels the swap from its own sample's dst embedding instead of the dst faceset mean. Only effective when Bleed power is not 0. Shown only on the first run.",
                             enabled_if=("bleed-power!=0.0",))
_DINO_POWER = FieldDef(key="dinov2-perceptual-power", option="dino_power", label="DINOv2 perceptual power", kind=FIELD_FLOAT,
                       default=0.0, valid_range=(0.0, 10.0),
                       help="L1 between DINOv2-S tokens of the masked reconstructions and their targets. 0 disables it. The most expensive supervisor per iteration. Shown only on the first run.")
_HELP_DINO_STRIDE = "Applies the DINOv2 term every N iterations instead of every one, scaled by N so the average gradient is unchanged (lazy regularization). 1 disables the stride. Shown only if DINOv2 perceptual power is not 0, on the first run."
_DINO_STRIDE_H1 = FieldDef(key="dinov2-stride", option="dino_ogni", label="DINOv2 stride", kind=FIELD_INT,
                           default=1, valid_range=(1, 100), help=_HELP_DINO_STRIDE,
                           enabled_if=("dinov2-perceptual-power!=0.0",))
_DINO_STRIDE_H2 = FieldDef(key="dinov2-stride", option="dino_ogni", label="DINOv2 stride", kind=FIELD_INT,
                           default=4, valid_range=(1, 100), help=_HELP_DINO_STRIDE + " 4 is the measured setting for H2.",
                           enabled_if=("dinov2-perceptual-power!=0.0",))
_FFL_POWER = FieldDef(key="focal-frequency-power", option="ffl_power", label="Focal frequency power", kind=FIELD_FLOAT,
                      default=0.0, valid_range=(0.0, 10.0),
                      help="Focal Frequency Loss on the masked reconstructions. 0 disables it. Shown only on the first run.")

_SUPERVISORS_H1 = (_ID_POWER_H1, _IFSR_POWER_H1, _BLEED_POWER_H1, _BLEED_PER_SAMPLE, _DINO_POWER, _DINO_STRIDE_H1, _FFL_POWER)
_SUPERVISORS_H2 = (_ID_POWER_H2, _IFSR_POWER_H2, _BLEED_POWER_H2, _BLEED_PER_SAMPLE, _DINO_POWER, _DINO_STRIDE_H2, _FFL_POWER)
_CHIAVI_SUPERVISORS = ("identity-power", "ifsr-power", "bleed-power", "bleed-per-sample",
                       "dinov2-perceptual-power", "dinov2-stride", "focal-frequency-power")

# ---- H2-only fields --------------------------------------------------------

_GRAFT = FieldDef(
    key="graft-from-model",
    option="innesto",   # models/Model_H2/Model.py, load_or_def_option('innesto', '')
    label="Graft from model",
    kind=FIELD_TEXT,
    default="",
    help="The name of a trained liae-udt SAEHD or SAEHDX model in this model dir (a pretrained RTM works), exactly as it appears in the Model name list: its encoder and decoder weights are copied at the first start, the identity vectors and the optimizer start from zero. The six size fields below are then fixed by those weights and not asked. Empty = train from scratch, never measured. A name that matches no such model stops the run and lists what is actually there. Shown only on the first run.",
)

_GRAFT_INTER = FieldDef(
    key="copy-the-inter-from-the-source",
    option="innesto_inter",
    label="Copy the inter from the source",
    kind=FIELD_BOOL,
    default=False,
    help="Warm inter: faster reconstruction at the start. Without supervisors it stops the identity vectors from separating (measured); with the supervisors on it is safe. Shown only with a graft source, on the first run.",
    enabled_if=("graft-from-model!=",),
)

_IDENTITY_VECTORS = FieldDef(
    key="identity-vectors",
    option="identita",
    label="Identity vectors",
    kind=FIELD_CHOICE,
    default="learned",
    choices=("learned", "adaface"),
    choice_help=(
        "Two free vectors trained with the rest of the net. The measured choice: they carry identity on their own (morph margin 0.73).",
        "The mean AdaFace embedding of each faceset, computed once at the first start and frozen. Measured: alone, the identity stays in the code, not in the vector.",
    ),
    help="Where the two identity vectors of the modulated decoder come from. Shown only on the first run.",
)

_MASK_TRUNK = FieldDef(
    key="mask-reads-the-identity-modulated-trunk",
    option="maschera_tronco",
    label="Mask reads the identity-modulated trunk",
    kind=FIELD_BOOL,
    default=False,
    help="Adds a zero-initialized 1x1 bridge from the identity-modulated trunk into the mask branch, so the mask can depend on the identity vector. Off: the mask depends on the code only, as in SAEHD. Fixed at the first start.",
)

_FISSATE_DAI_PESI = ("graft-from-model=",)     # enabled only while no graft source is written

_RESOLUTION_H2 = FieldDef(
    key="resolution", option="resolution", label="Resolution", kind=FIELD_INT,
    default=224, valid_range=(64, 640),
    help="More resolution requires more VRAM and time to train. H2 always uses the -t encoder and the -d output: the value is rounded to a multiple of 32. Fixed by the graft source when one is given. Shown only on the first run.",
    enabled_if=_FISSATE_DAI_PESI,
)
_FACE_TYPE_H2 = FieldDef(
    key="face-type", option="face_type", label="Face type", kind=FIELD_CHOICE,
    default="wf", choices=_FACE_TYPE_SAEHD.choices, choice_help=_FACE_TYPE_SAEHD.choice_help,
    help="Half / mid face / full face / whole face / head. Fixed by the graft source when one is given. Shown only on the first run.",
    enabled_if=_FISSATE_DAI_PESI,
)
_AE_DIMS_H2 = FieldDef(
    key="autoencoder-dimensions", option="ae_dims", label="AutoEncoder dimensions", kind=FIELD_INT,
    default=512, valid_range=(32, 1024),
    help="Size of the code. The decoder receives twice this many channels, as liae does. Fixed by the graft source when one is given. Shown only on the first run.",
    enabled_if=_FISSATE_DAI_PESI,
)
_ENCODER_DIMS_H2 = FieldDef(key="encoder-dimensions", option="e_dims", label="Encoder dimensions", kind=FIELD_INT,
                            default=64, valid_range=(16, 256), help=_ENCODER_DIMS.help + " Fixed by the graft source when one is given.",
                            enabled_if=_FISSATE_DAI_PESI)
_DECODER_DIMS_H2 = FieldDef(key="decoder-dimensions", option="d_dims", label="Decoder dimensions", kind=FIELD_INT,
                            default=64, valid_range=(16, 256), help=_DECODER_DIMS.help + " Fixed by the graft source when one is given.",
                            enabled_if=_FISSATE_DAI_PESI)
_DECODER_MASK_DIMS_H2 = FieldDef(key="decoder-mask-dimensions", option="d_mask_dims", label="Decoder mask dimensions", kind=FIELD_INT,
                                 default=22, valid_range=(16, 256), help=_DECODER_MASK_DIMS.help + " Fixed by the graft source when one is given.",
                                 enabled_if=_FISSATE_DAI_PESI)

_H1_FIELDS = _SAEHDX_FIELDS + _SUPERVISORS_H1

_H2_FIELDS = (
    _GPU_INDEXES, _AUTOBACKUP_HOUR, _WRITE_PREVIEW_HISTORY, _CHOOSE_PREVIEW_IMAGE,
    _TARGET_ITERATION, _FLIP_SRC, _FLIP_DST, _BATCH_SIZE_SAEHD,
    _GRAFT, _GRAFT_INTER,
    _RESOLUTION_H2, _FACE_TYPE_H2, _AE_DIMS_H2, _ENCODER_DIMS_H2, _DECODER_DIMS_H2, _DECODER_MASK_DIMS_H2,
    _MASK_TRUNK, _IDENTITY_VECTORS,
) + _SUPERVISORS_H2 + (
    _MASKED_TRAINING, _EYES_MOUTH_PRIO, _UNIFORM_YAW, _BLUR_OUT_MASK,
    _MODELS_OPT_ON_GPU_SAEHD, _ADABELIEF, _LR_DROPOUT, _RANDOM_WARP,
    _RANDOM_HSV_POWER, _CT_MODE_SAEHD, _CLIPGRAD,
    _CUDNN_BENCHMARK, _CUDA_GRAPH, _TORCH_COMPILE,
)

_SEZIONI_H1 = _SEZIONI_SAEHDX + (("Supervisors", _CHIAVI_SUPERVISORS),)

_SEZIONI_H2 = (
    ("Model", ("which-gpu-indexes-to-choose", "autobackup-every-n-hour",
               "write-preview-history", "choose-image-for-the-preview-history",
               "target-iteration", "batch_size",
               "resolution", "face-type", "autoencoder-dimensions", "encoder-dimensions",
               "decoder-dimensions", "decoder-mask-dimensions")),
    ("Graft", ("graft-from-model", "copy-the-inter-from-the-source")),
    ("Identity", ("identity-vectors",)),
    ("Supervisors", _CHIAVI_SUPERVISORS),
    ("Mask", ("masked-training", "blur-out-mask", "mask-reads-the-identity-modulated-trunk")),
    ("Augmentation", ("flip-src-faces-randomly", "flip-dst-faces-randomly",
                      "uniform-yaw-distribution-of-samples", "eyes-and-mouth-priority",
                      "enable-random-warp-of-samples", "random-huesaturationlight-intensity",
                      "color-transfer-for-src-faceset")),
    ("Performance", ("place-models-and-optimizer-on-gpu", "use-adabelief-optimizer",
                     "use-learning-rate-dropout", "enable-gradient-clipping",
                     "enable-cudnnbenchmark", "enable-cuda-graph-capture", "enable-torchcompile")),
)

STEPS = (
    StepDef(
        name="6) train AMP SRC-SRC",
        summary="Trains AMP on src alone, both sides fed from data_src -- the fork's stand-in for AMP's missing pretraining.",
        family="addestramento",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_src/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "AMP",
            )),
        ),
        fields=_AMP_FIELDS,
        sections=_SEZIONI_AMP,
        consumes=("faceset_src",),
        produces=("modello",),
        optional=True,
        needs_model_name=True,
    ),
    StepDef(
        name="6) train AMP",
        summary="Trains the AMP model on both facesets: a morphable architecture with a single adjustable blend factor.",
        family="addestramento",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_dst/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "AMP",
            )),
        ),
        fields=_AMP_FIELDS,
        sections=_SEZIONI_AMP,
        consumes=("faceset_src", "faceset_dst"),
        produces=("modello",),
        needs_model_name=True,
    ),
    StepDef(
        name="6) train H1",
        summary="SAEHDX plus frozen supervisors in the loss (identity, IFSR, DINOv2, focal frequency, bleed). All at zero, it is SAEHDX.",
        family="addestramento",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_dst/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "H1",
            )),
        ),
        fields=_H1_FIELDS,
        sections=_SEZIONI_H1,
        consumes=("faceset_src", "faceset_dst", "pretrain"),
        produces=("modello",),
        needs_model_name=True,
    ),
    StepDef(
        name="6) train H2",
        summary="Identity-modulated decoder grafted from a trained liae-udt model in this model dir; defaults are the measured recipe.",
        family="addestramento",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_dst/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "H2",
            )),
        ),
        fields=_H2_FIELDS,
        sections=_SEZIONI_H2,
        consumes=("faceset_src", "faceset_dst"),
        produces=("modello",),
        needs_model_name=True,
    ),
    StepDef(
        name="6) train SAEHD",
        summary="Trains the SAEHD model on both facesets: the fullest set of tunable options of the four.",
        family="addestramento",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_dst/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "SAEHD",
            )),
        ),
        fields=_SAEHD_FIELDS,
        sections=_SEZIONI_SAEHD,
        consumes=("faceset_src", "faceset_dst", "pretrain"),
        produces=("modello",),
        needs_model_name=True,
    ),
    StepDef(
        name="6) train SAEHDX",
        summary="SAEHD in mixed precision: about 29% faster per iteration and 18% less VRAM, same result.",
        family="addestramento",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("train",), args=(
                "--training-data-src-dir", "{WORKSPACE}/data_src/aligned",
                "--training-data-dst-dir", "{WORKSPACE}/data_dst/aligned",
                "--pretraining-data-dir", "{INTERNAL}/pretrain_faces",
                "--model-dir", "{WORKSPACE}/model",
                "--model", "SAEHDX",
            )),
        ),
        fields=_SAEHDX_FIELDS,
        sections=_SEZIONI_SAEHDX,
        consumes=("faceset_src", "faceset_dst", "pretrain"),
        produces=("modello",),
        needs_model_name=True,
    ),
)
