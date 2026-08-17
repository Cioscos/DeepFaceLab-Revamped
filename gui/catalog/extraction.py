"""Family estrazione: the six "4)"/"5)" faceset-extraction steps.

All six resolve to the `extract` subcommand of `main.py`. Each command line
in scripts/commands.toml fixes some of Extractor.main's options directly
(--max-faces-from-image/--output-debug on the four data_dst variants; the
three MANUAL variants also fix --detector manual), which suppresses the
matching console prompt -- only the prompts that can still fire on a given
command are modelled below, in the order they appear.

**The three non-manual steps ("4) data_src faceset extract",
"5) data_dst faceset extract", "5) data_dst faceset extract + manual fix")
no longer fix --detector.** `manual` stays out of the CLI choice for these
three (it is a different branch of Extractor.main, not a detector -- see
MotoriCatalog's docstring), so removing the fixed value re-opens exactly the
"Choose detector type."/"Choose landmark model." console prompts, and
_DETECTOR/_LANDMARKER model them: choices and defaults come straight from
MotoriCatalog, never duplicated here. They are placed right after
_JPEG_QUALITY and before _OUTPUT_DEBUG where present -- the real order in
Extractor.main (detector, then landmarker, come after jpeg_quality and
before the debug-output prompt, not right after the GPU-index prompt).

**The two prompts now answer to "Detector"/"Landmarker" (`Extractor.py`,
the `if detector is None:`/`if landmarker is None ...:` blocks), not to the
empty string.** They used to share `io.input_int("", 0, ...)`: with
`prompt_key("")` being `""` for both, DFL_ANSWERS_FILE could not address
one without the other, and the type mismatch (an index vs. the catalog's
string choices) would still have defeated it even with distinct keys --
`io.input_str` reads the engine key directly, the same shape `_DETECTOR`/
`_LANDMARKER` send. `_DETECTOR.key`/`_LANDMARKER.key` (`"detector"`/
`"landmarker"`) are exactly `prompt_key("Detector")`/`prompt_key("Landmarker")`,
so no catalog key changed. The precedent fix for the identical class of
collision is `merger/MergerConfig.py:39,196,211` (see
`gui/catalog/merging.py`'s docstring).

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
from mainscripts import MotoriCatalog as _MC

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

# I due selettori del motore. Nel form seguono _JPEG_QUALITY e precedono
# _OUTPUT_DEBUG (dove presente): e' l'ordine reale dei prompt dentro
# Extractor.main -- il detector e' chiesto DOPO jpeg_quality, non subito
# dopo l'indice GPU, e va letto li' e non presunto.
#
# La tendina mostra `Motore.label` ("S3FD, full resolution") e manda
# `Motore.key` ("s3fd-alta-risoluzione"): e' esattamente cio' per cui
# esiste `choice_values` (stesso uso in faceset_care.py e merging.py). Con
# le sole chiavi l'utente leggeva lo slug, in un'interfaccia il cui testo e'
# per convenzione inglese, e la label del catalogo -- che il suo docstring
# dichiara essere "cio' che il selettore mostra" -- non arrivava a nessuno.
def _etichetta(motori, chiave):
    return next(m.label for m in motori if m.key == chiave)


_DETECTOR = FieldDef(
    key="detector",
    label="Detector",
    kind=FIELD_CHOICE,
    default=_etichetta(_MC.RILEVATORI, _MC.DEFAULT_RILEVATORE),
    choices=tuple(m.label for m in _MC.RILEVATORI),
    choice_values=_MC.CHIAVI_RILEVATORI,
    choice_help=tuple(m.help for m in _MC.RILEVATORI),
    help="Which model finds the faces in the frame.",
)

_LANDMARKER = FieldDef(
    key="landmarker",
    label="Landmark model",
    kind=FIELD_CHOICE,
    default=_etichetta(_MC.ALLINEATORI, _MC.DEFAULT_ALLINEATORE),
    choices=tuple(m.label for m in _MC.ALLINEATORI),
    choice_values=_MC.CHIAVI_ALLINEATORI,
    choice_help=tuple(m.help for m in _MC.ALLINEATORI),
    help="Which model places the 68 landmarks once a face has been found.",
)

# Il default NON e' scritto qui: e' MotoriCatalog.LATO_MIN_PREDEFINITO, la
# stessa costante che S3FDExtractor usa nella propria firma e che il prompt
# "Minimum face size" di Extractor.main mostra. Un 40 scritto a mano in
# questo file sarebbe la terza copia, e la prima a scostarsi.
_MIN_FACE_SIZE = FieldDef(
    key="minimum-face-size",
    label="Minimum face size",
    kind=FIELD_INT,
    default=_MC.LATO_MIN_PREDEFINITO,
    valid_range=(1, 1024),
    help="Detections whose shorter side is below this many pixels of the source frame are discarded. Lower it to keep small or distant faces the default throws away; it costs nothing in speed.",
)

_OUTPUT_DEBUG = FieldDef(
    key="write-debug-images-to-aligned_debug",
    label="Write debug images to aligned_debug?",
    kind=FIELD_BOOL,
    default=False,
    help="Saves a copy of every processed frame with the detected face rectangle and landmarks drawn on it, in the aligned_debug folder next to aligned. Useful to spot misdetections before training; costs extra disk space and write time.",
)

# Adding _DETECTOR/_LANDMARKER pushes "4) data_src faceset extract" from 7
# to 9 fields, past the >=8 threshold that test_catalog_aiuti.py sections at
# (see test_i_passi_da_sezionare_sono_otto). Grouped by theme, same
# granularity as training.py/merging.py's sections, not by console prompt
# order (sections group by theme there too, e.g. _SEZIONI_MERGE_AMP's
# "Output" mixes fields that fire in a different order at runtime).
_SEZIONI_ESTRAZIONE = (
    ("Session", ("continue-extraction", "which-gpu-indexes-to-choose",
                "face-type")),
    ("Detection", ("max-number-of-faces-from-image", "detector",
                   "landmarker", "minimum-face-size")),
    ("Output", ("image-size", "jpeg-quality",
               "write-debug-images-to-aligned_debug")),
)


def _sezioni(campi):
    """Le stesse tre sezioni tematiche, ristrette ai campi che il passo ha
    davvero.

    Coi tre campi del motore i due passi "5) data_dst faceset extract" e
    "+ manual fix" passano da 7 a 8 campi, cioe' oltre la soglia di
    sezionamento, ma non hanno ne' --max-faces-from-image ne' l'output di
    debug fra i loro prompt (li fissa gia' la riga di comando). Tre tuple
    scritte a mano divergerebbero al primo campo nuovo: la ripartizione per
    tema resta una sola, in _SEZIONI_ESTRAZIONE sopra."""
    presenti = {c.key for c in campi}
    return tuple((titolo, tuple(k for k in chiavi if k in presenti))
                 for titolo, chiavi in _SEZIONI_ESTRAZIONE
                 if any(k in presenti for k in chiavi))


# I due soli insiemi di campi che i passi automatici usano: "4) data_src"
# chiede anche il numero massimo di volti e l'output di debug, i due
# "5) data_dst" no -- li fissa gia' la riga di comando di commands.toml.
_CAMPI_SRC = (
    _CONTINUE_EXTRACTION, _GPU_INDEXES, _FACE_TYPE, _MAX_FACES,
    _IMAGE_SIZE, _JPEG_QUALITY, _DETECTOR, _LANDMARKER, _MIN_FACE_SIZE,
    _OUTPUT_DEBUG,
)

_CAMPI_DST = (
    _CONTINUE_EXTRACTION, _GPU_INDEXES, _FACE_TYPE, _IMAGE_SIZE,
    _JPEG_QUALITY, _DETECTOR, _LANDMARKER, _MIN_FACE_SIZE,
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
            )),
        ),
        fields=_CAMPI_SRC,
        sections=_sezioni(_CAMPI_SRC),
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
                "--max-faces-from-image", "0",
                "--manual-fix",
            )),
        ),
        fields=_CAMPI_DST,
        sections=_sezioni(_CAMPI_DST),
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
                "--max-faces-from-image", "0",
                "--output-debug",
            )),
        ),
        fields=_CAMPI_DST,
        sections=_sezioni(_CAMPI_DST),
        consumes=("frame_dst",),
        produces=("faceset_dst", "debug_dst"),
    ),
)
