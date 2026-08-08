"""Family addestramento: the four "6) train *" steps (SAEHD, SAEHDX, AMP, AMP SRC-SRC).

All four are `needs_model_name=True`: `models.ModelBase.__init__`'s "choose one
of saved models, or enter a name" block (prompt key `""`, and its
"No saved models found..." sibling) is always active for these steps -- none
of the four command lines passes `--force-model-name` -- but the GUI supplies
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

_AUTOBACKUP_HOUR = FieldDef(
    key="autobackup-every-n-hour",
    label="Autobackup every N hour",
    kind=FIELD_INT,
    default=0,
    help="Autobackup model files with preview every N hour. Latest backup located in model/<>_autobackups/01. Shown on the first run, or when overriding settings on a resumed run.",
)

_WRITE_PREVIEW_HISTORY = FieldDef(
    key="write-preview-history",
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
    help="Shown only if Write preview history is enabled. On this backend the interactive chooser only exists on Windows; elsewhere (and inside Colab, which asks a differently-worded prompt not modeled here) it is not offered, and random preview samples are used instead.",
    enabled_if=("write-preview-history=y",),
)

_TARGET_ITERATION = FieldDef(
    key="target-iteration",
    label="Target iteration",
    kind=FIELD_INT,
    default=0,
    help="Shown under the same condition as Autobackup every N hour. 0 - no target, train indefinitely.",
)

_FLIP_SRC = FieldDef(
    key="flip-src-faces-randomly",
    label="Flip SRC faces randomly",
    kind=FIELD_BOOL,
    default=False,
    help="Random horizontal flip SRC faceset. Covers more angles, but the face may look less naturally. Shown under the same condition as Autobackup every N hour.",
)

_FLIP_DST = FieldDef(
    key="flip-dst-faces-randomly",
    label="Flip DST faces randomly",
    kind=FIELD_BOOL,
    default=True,
    help="Random horizontal flip DST faceset. Makes generalization of src->dst better, if src random flip is not enabled. Shown under the same condition as Autobackup every N hour.",
)

_BATCH_SIZE_SAEHD = FieldDef(
    key="batch_size",
    label="Batch size",
    kind=FIELD_INT,
    default=8,
    help="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually. No range is enforced. Runtime-suggested default: 8 if the worst detected device has >=4 GB VRAM, else 4. Shown under the same condition as Autobackup every N hour.",
)

_AE_ARCHITECTURE = FieldDef(
    key="ae-architecture",
    label="AE architecture",
    kind=FIELD_TEXT,
    default="liae-ud",
    help="'df' keeps more identity-preserved face. 'liae' can fix overly different face shapes. '-u' increased likeness of the face. '-d' (experimental) doubling the resolution using the same computation cost. Examples: df, liae, df-d, df-ud, liae-ud, ... Validated in a loop: type must be 'df' or 'liae', optional suffix letters from u/d/t/c. Shown only on the first run.",
)

_AE_DIMS = FieldDef(
    key="autoencoder-dimensions",
    label="AutoEncoder dimensions",
    kind=FIELD_INT,
    default=256,
    valid_range=(32, 1024),
    help="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU. Shown only on the first run.",
)

_ENCODER_DIMS = FieldDef(
    key="encoder-dimensions",
    label="Encoder dimensions",
    kind=FIELD_INT,
    default=64,
    valid_range=(16, 256),
    help="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU. Rounded to an even number. Shown only on the first run.",
)

_DECODER_DIMS = FieldDef(
    key="decoder-dimensions",
    label="Decoder dimensions",
    kind=FIELD_INT,
    default=64,
    valid_range=(16, 256),
    help="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU. Rounded to an even number. Shown only on the first run.",
)

_DECODER_MASK_DIMS = FieldDef(
    key="decoder-mask-dimensions",
    label="Decoder mask dimensions",
    kind=FIELD_INT,
    default=22,
    valid_range=(16, 256),
    help="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality. Rounded to an even number. Default is Decoder dimensions / 3, rounded even. Shown only on the first run.",
)

_UNIFORM_YAW = FieldDef(
    key="uniform-yaw-distribution-of-samples",
    label="Uniform yaw distribution of samples",
    kind=FIELD_BOOL,
    default=False,
    help="Helps to fix blurry side faces due to small amount of them in the faceset. Shown under the same condition as Autobackup every N hour.",
)

_BLUR_OUT_MASK = FieldDef(
    key="blur-out-mask",
    label="Blur out mask",
    kind=FIELD_BOOL,
    default=False,
    help="Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required. Shown under the same condition as Autobackup every N hour.",
)

_LR_DROPOUT = FieldDef(
    key="use-learning-rate-dropout",
    label="Use learning rate dropout",
    kind=FIELD_CHOICE,
    default="n",
    choices=("n", "y", "cpu"),
    help="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. n - disabled. y - enabled. cpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration. Shown under the same condition as Autobackup every N hour.",
)

_RANDOM_WARP = FieldDef(
    key="enable-random-warp-of-samples",
    label="Enable random warp of samples",
    kind=FIELD_BOOL,
    default=True,
    help="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations. Shown under the same condition as Autobackup every N hour.",
)

_GAN_PATCH_SIZE = FieldDef(
    key="gan-patch-size",
    label="GAN patch size",
    kind=FIELD_INT,
    default=16,
    valid_range=(3, 640),
    help="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8. Default is resolution / 8. Shown only if GAN power is not 0.",
    enabled_if=("gan-power!=0.0",),
)

_GAN_DIMS = FieldDef(
    key="gan-dimensions",
    label="GAN dimensions",
    kind=FIELD_INT,
    default=16,
    valid_range=(4, 512),
    help="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16. Shown only if GAN power is not 0.",
    enabled_if=("gan-power!=0.0",),
)

_CLIPGRAD = FieldDef(
    key="enable-gradient-clipping",
    label="Enable gradient clipping",
    kind=FIELD_BOOL,
    default=False,
    help="Gradient clipping reduces chance of model collapse, sacrificing speed of training. Shown under the same condition as Autobackup every N hour.",
)

# ---- SAEHD/SAEHDX-only fields -------------------------------------------

_FACE_TYPE_SAEHD = FieldDef(
    key="face-type",
    label="Face type",
    kind=FIELD_CHOICE,
    default="f",
    choices=("h", "mf", "f", "wf", "head"),
    help="Half / mid face / full face / whole face / head. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset. Shown only on the first run.",
)

_RESOLUTION_SAEHD = FieldDef(
    key="resolution",
    label="Resolution",
    kind=FIELD_INT,
    default=128,
    valid_range=(64, 640),
    help="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi. Shown only on the first run.",
)

_MASKED_TRAINING = FieldDef(
    key="masked-training",
    label="Masked training",
    kind=FIELD_BOOL,
    default=True,
    help="This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly. Shown only if Face type is 'wf' or 'head', on the first run or override.",
    enabled_if=("face-type=wf|head",),
)

_EYES_MOUTH_PRIO = FieldDef(
    key="eyes-and-mouth-priority",
    label="Eyes and mouth priority",
    kind=FIELD_BOOL,
    default=False,
    help="Helps to fix eye problems during training like \"alien eyes\" and wrong eyes direction. Also makes the detail of the teeth higher. Shown under the same condition as Autobackup every N hour.",
)

_MODELS_OPT_ON_GPU_SAEHD = FieldDef(
    key="place-models-and-optimizer-on-gpu",
    label="Place models and optimizer on GPU",
    kind=FIELD_BOOL,
    default=True,
    help="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. Answering n moves the whole model to the CPU: in this PyTorch port the weights and the computation cannot be split, so training then runs entirely on the CPU and is far slower. It frees all the VRAM, but it is not a way to fit bigger dimensions on the GPU. Outside training (merge and export) this option is ignored: use --cpu-only there. Shown under the same condition as Autobackup every N hour.",
)

_ADABELIEF = FieldDef(
    key="use-adabelief-optimizer",
    label="Use AdaBelief optimizer?",
    kind=FIELD_BOOL,
    default=True,
    help="Use AdaBelief optimizer. It requires more VRAM, but the accuracy and the generalization of the model is higher. Shown under the same condition as Autobackup every N hour.",
)

_RANDOM_HSV_POWER = FieldDef(
    key="random-huesaturationlight-intensity",
    label="Random hue/saturation/light intensity",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 0.3),
    help="Random hue/saturation/light intensity applied to the src face set only at the input of the neural network. Stabilizes color perturbations during face swapping. Reduces the quality of the color transfer by selecting the closest one in the src faceset. Thus the src faceset must be diverse enough. Typical fine value is 0.05. Shown under the same condition as Autobackup every N hour.",
)

_GAN_POWER_SAEHD = FieldDef(
    key="gan-power",
    label="GAN power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 5.0),
    help="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with lr_dropout(on) and random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1. Shown under the same condition as Autobackup every N hour.",
)

_TRUE_FACE_POWER = FieldDef(
    key="true-face-power",
    label="'True face' power.",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 1.0),
    help="Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01. Shown only if AE architecture is 'df' (any suffix), on the first run or override -- if it is 'liae', stays 0.0 with no prompt.",
    enabled_if=("ae-architecture~=df",),
)

_FACE_STYLE_POWER = FieldDef(
    key="face-style-power",
    label="Face style power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 100.0),
    help="Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse. Shown under the same condition as Autobackup every N hour.",
)

_BG_STYLE_POWER = FieldDef(
    key="background-style-power",
    label="Background style power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 100.0),
    help="Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0. Shown under the same condition as Autobackup every N hour.",
)

_CT_MODE_SAEHD = FieldDef(
    key="color-transfer-for-src-faceset",
    label="Color transfer for src faceset",
    kind=FIELD_CHOICE,
    default="none",
    choices=("none", "rct", "lct", "mkl", "idt", "sot"),
    help="Change color distribution of src samples close to dst samples. Try all modes to find the best. Shown under the same condition as Autobackup every N hour.",
)

_PRETRAIN = FieldDef(
    key="enable-pretraining-mode",
    label="Enable pretraining mode",
    kind=FIELD_BOOL,
    default=False,
    help="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly. Forces random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y. Shown under the same condition as Autobackup every N hour. Requires --pretraining-data-dir, always passed by this step's invocation.",
)

_CUDNN_BENCHMARK = FieldDef(
    key="enable-cudnnbenchmark",
    label="Enable cudnn.benchmark",
    kind=FIELD_BOOL,
    default=False,
    help="Lets cuDNN autotune the fastest convolution algorithm for the fixed shapes of this run. The extra speed on top of the other SAEHDX optimizations was too small to tell apart from measurement noise, while the extra VRAM was consistent and large. Off by default: on recent, VRAM-constrained cards the trade is usually not worth it. Shown only on the first run.",
)

_CUDA_GRAPH = FieldDef(
    key="enable-cuda-graph-capture",
    label="Enable CUDA graph capture",
    kind=FIELD_BOOL,
    default=False,
    help="Records the training step once as a CUDA graph and replays it, instead of re-issuing every kernel from Python each iteration. It needs fixed input buffers, so it turns the batch prefetch off, and it only applies to a plain src_dst step: with GAN, true face power, learning rate dropout or clipgrad the capture is skipped and training goes on unchanged. Off by default. Shown only on the first run.",
)

_TORCH_COMPILE = FieldDef(
    key="enable-torchcompile",
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
    label="Resolution",
    kind=FIELD_INT,
    default=224,
    valid_range=(64, 640),
    help="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 32. Shown only on the first run.",
)

_FACE_TYPE_AMP = FieldDef(
    key="face-type",
    label="Face type",
    kind=FIELD_CHOICE,
    default="wf",
    choices=("f", "wf", "head"),
    help="whole face / head. Shown only on the first run.",
)

_INTER_DIMS = FieldDef(
    key="inter-dimensions",
    label="Inter dimensions",
    kind=FIELD_INT,
    default=1024,
    valid_range=(32, 2048),
    help="Should be equal or more than AutoEncoder dimensions. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU. Shown only on the first run.",
)

_MORPH_FACTOR_TRAIN = FieldDef(
    key="morph-factor",
    label="Morph factor.",
    kind=FIELD_FLOAT,
    default=0.5,
    valid_range=(0.1, 0.5),
    help="Typical fine value is 0.5. Shown only on the first run. Unlike the other options above, a new model reusing an existing name does not inherit this value from the class-wide _default_options.dat file.",
)

_BATCH_SIZE_AMP = FieldDef(
    key="batch_size",
    label="Batch size",
    kind=FIELD_INT,
    default=8,
    help="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually. No range is enforced. Suggested default is fixed at 8 for this model, not computed from VRAM. Shown under the same condition as Autobackup every N hour.",
)

_MODELS_OPT_ON_GPU_AMP = FieldDef(
    key="place-models-and-optimizer-on-gpu",
    label="Place models and optimizer on GPU",
    kind=FIELD_BOOL,
    default=True,
    help="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions. This help text is still the TensorFlow-era wording, not corrected for this port the way SAEHD's equivalent was: here too, answering n moves the whole model to the CPU. Shown under the same condition as Autobackup every N hour.",
)

_GAN_POWER_AMP = FieldDef(
    key="gan-power",
    label="GAN power",
    kind=FIELD_FLOAT,
    default=0.0,
    valid_range=(0.0, 5.0),
    help="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1. Shown under the same condition as Autobackup every N hour.",
)

_CT_MODE_AMP = FieldDef(
    key="color-transfer-for-src-faceset",
    label="Color transfer for src faceset",
    kind=FIELD_CHOICE,
    default="none",
    choices=("none", "rct", "lct", "mkl", "idt", "sot"),
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

STEPS = (
    StepDef(
        name="6) train AMP SRC-SRC",
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
        consumes=("faceset_src",),
        produces=("modello",),
        optional=True,
        needs_model_name=True,
    ),
    StepDef(
        name="6) train AMP",
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
        consumes=("faceset_src", "faceset_dst"),
        produces=("modello",),
        needs_model_name=True,
    ),
    StepDef(
        name="6) train SAEHD",
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
        consumes=("faceset_src", "faceset_dst", "pretrain"),
        produces=("modello",),
        needs_model_name=True,
    ),
    StepDef(
        name="6) train SAEHDX",
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
        consumes=("faceset_src", "faceset_dst", "pretrain"),
        produces=("modello",),
        needs_model_name=True,
    ),
)
