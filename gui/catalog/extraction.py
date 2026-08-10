"""Family estrazione: the six "4)"/"5)" faceset-extraction steps.

All six resolve to the `extract` subcommand of `main.py`. Each command line
in scripts/commands.toml fixes some of Extractor.main's options directly
(--detector always; --max-faces-from-image/--output-debug on the four
data_dst variants), which suppresses the matching console prompt — only the
prompts that can still fire on a given command are modelled below, in the
order they appear.

Two prompts are deliberately left out of every step's fields even though
they are documented in the workflow schede: the "Continue extraction?"
warning block and "Press enter to continue" confirmation are raw
`io.input()` calls whose return value is discarded (see
core/interact/interact.py -- `input()` is not routed through the
DFL_ANSWERS_FILE lookup that backs every other prompt, unlike
`input_bool`/`input_int`/`input_str`), so there is no value for a form field
to hold and no way for a driven answer to reach them.
"""
from gui.catalog.model import (
    FIELD_BOOL, FIELD_CHOICE, FIELD_INT, FIELD_TEXT, KIND_MAIN,
    PROCESS_PROMPT, PROCESS_SESSION, FieldDef, Invocation, StepDef,
)

_CONTINUE_EXTRACTION = FieldDef(
    key="continue-extraction",
    label="Continue extraction?",
    kind=FIELD_BOOL,
    default=True,
    help="Extraction can be continued, but you must specify the same options again.",
)

_GPU_INDEX = FieldDef(
    key="which-gpu-index-to-choose",
    label="GPU index",
    kind=FIELD_TEXT,
    default=None,
    help="Index of the device to use, or 'cpu'. Computed at runtime as the single best-scoring detected device.",
)

_GPU_INDEXES = FieldDef(
    key="which-gpu-indexes-to-choose",
    label="GPU indexes",
    kind=FIELD_TEXT,
    default=None,
    help="Comma-separated device indexes, or 'cpu'. Computed at runtime as all detected devices.",
)

_FACE_TYPE = FieldDef(
    key="face-type",
    label="Face type",
    kind=FIELD_CHOICE,
    default="wf",
    choices=("f", "wf", "head"),
    choice_help=(
        "Full face: crops down to the eyebrows and chin. Pick it only if every model trained on this faceset will use 'f' too.",
        "Whole face: extends the crop to include the forehead. The default, and the safe choice if you have not settled on a training face type yet.",
        "Head: keeps the entire head, hair included. Needed only for a 'head' model, and both facesets will then need an XSeg mask.",
    ),
    help="Full face / whole face / head. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.",
)

_MAX_FACES = FieldDef(
    key="max-number-of-faces-from-image",
    label="Max number of faces from image",
    kind=FIELD_INT,
    default=0,
    help="If you extract a src faceset that has frames with a large number of faces, it is advisable to set max faces to 3 to speed up extraction. 0 - unlimited",
)

_IMAGE_SIZE = FieldDef(
    key="image-size",
    label="Image size",
    kind=FIELD_INT,
    default=512,
    valid_range=(256, 2048),
    help="Output image size. The higher image size, the worse face-enhancer works. Use higher than 512 value only if the source image is sharp enough and the face does not need to be enhanced. Defaults to 768 when Face type is head.",
)

_JPEG_QUALITY = FieldDef(
    key="jpeg-quality",
    label="Jpeg quality",
    kind=FIELD_INT,
    default=90,
    valid_range=(1, 100),
    help="The higher jpeg quality the larger the output file size.",
)

_OUTPUT_DEBUG = FieldDef(
    key="write-debug-images-to-aligned_debug",
    label="Write debug images to aligned_debug?",
    kind=FIELD_BOOL,
    default=False,
    help="Saves a copy of every processed frame with the detected face rectangle and landmarks drawn on it, in the aligned_debug folder next to aligned. Useful to spot misdetections before training; costs extra disk space and write time.",
)

STEPS = (
    StepDef(
        name="4) data_src faceset extract MANUAL",
        summary="Detects and aligns the source faces by hand, frame by frame, in the manual extractor window.",
        family="estrazione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("extract",), args=(
                "--input-dir", "{WORKSPACE}/data_src",
                "--output-dir", "{WORKSPACE}/data_src/aligned",
                "--detector", "manual",
            )),
        ),
        fields=(
            _CONTINUE_EXTRACTION, _GPU_INDEX, _FACE_TYPE, _MAX_FACES,
            _IMAGE_SIZE, _JPEG_QUALITY, _OUTPUT_DEBUG,
        ),
        consumes=("frame_src",),
        produces=("faceset_src",),
    ),
    StepDef(
        name="4) data_src faceset extract",
        summary="Finds and aligns the faces in the source frames automatically, writing them to data_src/aligned.",
        family="estrazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("extract",), args=(
                "--input-dir", "{WORKSPACE}/data_src",
                "--output-dir", "{WORKSPACE}/data_src/aligned",
                "--detector", "s3fd",
            )),
        ),
        fields=(
            _CONTINUE_EXTRACTION, _GPU_INDEXES, _FACE_TYPE, _MAX_FACES,
            _IMAGE_SIZE, _JPEG_QUALITY, _OUTPUT_DEBUG,
        ),
        consumes=("frame_src",),
        produces=("faceset_src",),
    ),
    StepDef(
        name="5) data_dst faceset MANUAL RE-EXTRACT DELETED ALIGNED_DEBUG",
        summary="Re-extracts by hand only the destination frames whose debug image was deleted, to fix misdetections.",
        family="estrazione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("extract",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/aligned",
                "--detector", "manual",
                "--max-faces-from-image", "0",
                "--output-debug",
                "--manual-output-debug-fix",
            )),
        ),
        # Face type is only asked if the silent inference from an existing
        # aligned_debug/aligned DFLJPG fails (see the workflow scheda); that
        # condition is filesystem state, not another field, so it is not
        # encoded as enabled_if here.
        fields=(_GPU_INDEX, _FACE_TYPE, _IMAGE_SIZE, _JPEG_QUALITY),
        consumes=("frame_dst", "debug_dst"),
        modifies=("faceset_dst", "debug_dst"),
    ),
    StepDef(
        name="5) data_dst faceset extract + manual fix",
        summary="Extracts destination faces automatically, then opens the manual window only for the frames it missed.",
        family="estrazione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("extract",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/aligned",
                "--output-debug",
                "--detector", "s3fd",
                "--max-faces-from-image", "0",
                "--manual-fix",
            )),
        ),
        fields=(
            _CONTINUE_EXTRACTION, _GPU_INDEXES, _FACE_TYPE, _IMAGE_SIZE,
            _JPEG_QUALITY,
        ),
        consumes=("frame_dst",),
        produces=("faceset_dst", "debug_dst"),
    ),
    StepDef(
        name="5) data_dst faceset extract MANUAL",
        summary="Detects and aligns every destination face by hand, frame by frame, in the manual extractor window.",
        family="estrazione",
        kind=KIND_MAIN,
        process=PROCESS_SESSION,
        invocations=(
            Invocation(verb=("extract",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/aligned",
                "--detector", "manual",
                "--max-faces-from-image", "0",
                "--output-debug",
            )),
        ),
        fields=(
            _CONTINUE_EXTRACTION, _GPU_INDEX, _FACE_TYPE, _IMAGE_SIZE,
            _JPEG_QUALITY,
        ),
        consumes=("frame_dst",),
        produces=("faceset_dst", "debug_dst"),
    ),
    StepDef(
        name="5) data_dst faceset extract",
        summary="Finds and aligns the faces in the destination frames, writing them to data_dst/aligned.",
        family="estrazione",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("extract",), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
                "--output-dir", "{WORKSPACE}/data_dst/aligned",
                "--detector", "s3fd",
                "--max-faces-from-image", "0",
                "--output-debug",
            )),
        ),
        fields=(
            _CONTINUE_EXTRACTION, _GPU_INDEXES, _FACE_TYPE, _IMAGE_SIZE,
            _JPEG_QUALITY,
        ),
        consumes=("frame_dst",),
        produces=("faceset_dst", "debug_dst"),
    ),
)
