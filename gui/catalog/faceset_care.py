"""Family cura-faceset: the seventeen "4.x)"/"5.x)" faceset-curation steps.

Three steps are pure viewers (`kind = "viewer"`): they never touch main.py,
the generated script just opens the target folder with the OS-associated
application, and they carry no fields. The other fourteen resolve to the
`sort`, `util` and `facesettool` subcommands.

Two prompts are left out of every step's fields for the same reason given in
extraction.py: they are raw `io.input()` calls whose return value is
discarded and are never routed through the DFL_ANSWERS_FILE lookup, so
there is no value for a form field to hold -- "Press enter to continue and
overwrite." on faceset pack (PackedFaceset.pack, only when a `faceset.pak`
already exists) and the equivalent "Continue extraction?" family is not
present in this family at all.

The sorting-method prompt (`Sorter.main`) is called with an empty prompt
string -- the menu itself is printed beforehand with `io.log_info`, and only
the trailing `io.input_int("", 5, ...)` is the actual answer-file lookup
site -- so its FieldDef key is the empty string, matching prompt_key("").
"""
from gui.catalog.model import (
    FIELD_BOOL, FIELD_CHOICE, FIELD_INT, FIELD_PATH, FIELD_TEXT, KIND_MAIN,
    KIND_VIEWER, PROCESS_BATCH, PROCESS_PROMPT, FieldDef, Invocation, StepDef,
)

# Duplicati dei descrittori dei metodi di ordinamento: l'interfaccia non
# importa l'applicazione a runtime. La duplicazione non puo' derivare -- una
# guardia della suite leggera confronta etichette, indici, spiegazioni e campi
# condizionali con la loro sorgente. L'ordine e' quello del menu' e non si
# tocca: la memoria di progetto ricorda l'indice, non l'etichetta.
_SORT_METHODS = (
    "blur", "motion blur", "face yaw direction", "face pitch direction",
    "face rect size in source image", "histogram similarity",
    "histogram dissimilarity", "brightness", "hue",
    "amount of black pixels", "original filename", "one face in image",
    "absolute pixel difference", "best faces", "best faces faster",
    "duplicate faces", "most varied faces", "upscale factor",
    "match the other faceset's poses",
    "absolute pixel difference on 32x32 thumbnails, 5x faster",
)

# The call-site (Sorter.py: io.input_int("", 5, valid_list=[*range(20)]))
# takes the menu index, not the description string, so choice_values maps
# each _SORT_METHODS entry to its real index in Sorter.sort_func_methods.
_SORT_METHOD = FieldDef(
    key="",
    label="Sorting method",
    kind=FIELD_CHOICE,
    default="histogram similarity",
    choices=_SORT_METHODS,
    choice_values=tuple(range(len(_SORT_METHODS))),
    choice_help=(
        "Sharpest faces first, blurriest last. The usual default and a good general-purpose order to look through. Faces it cannot read at all are moved to a sibling _trash folder, never deleted.",
        "Like blur, but measures motion smear specifically instead of general softness -- better at catching faces blurred by fast head movement. Faces it cannot read at all are moved to a sibling _trash folder, never deleted.",
        "Orders by left/right head turn. Useful to check the faceset covers enough angles, or to isolate the extreme ones. Files without the alignment data it needs are moved to a sibling _trash folder, never deleted.",
        "Orders by up/down head tilt. Same use as face yaw direction, on the other axis. Files without the alignment data it needs are moved to a sibling _trash folder, never deleted.",
        "Orders by how large the detected face was in its source frame -- largest (closest to camera) first. Files without the alignment data it needs are moved to a sibling _trash folder, never deleted.",
        "Groups visually similar faces next to each other, walking the whole folder as one chain. Uses the face mask when the faces carry landmarks, so the background does not decide the order.",
        "Like histogram similarity, but pushes the most different-looking faces to the front instead of grouping similar ones. Same histogram measurement, ranked by total difference from every other face instead of chained pair by pair.",
        "Brightest faces first. Handy for spotting over/under-exposed frames.",
        "Orders by hue (dominant color tone), not by lightness -- catches frames with an odd color cast.",
        "Fewest black pixels first -- pushes frames with letterboxing, occlusion or failed alignment toward the end.",
        "Groups faces by the source frame they came from, in the order the frames were extracted. Undoes any other sort. Files that do not carry a source frame name are moved to a sibling _trash folder, never deleted.",
        "Keeps only frames where exactly one face was detected, trashing every frame that produced more than one -- crowd shots, mirrors, posters.",
        "Orders by pairwise visual difference between every face in the set. The exact order, and by far the slowest method here. Every face must be at the same resolution, or it stops and points at the thumbnail variant.",
        "Keeps a target number of faces spread evenly across head angles instead of just the sharpest ones, trashing the rest. The general-purpose way to cut a faceset down to size.",
        "Like best faces, but ranks by source-rect size instead of blur -- much faster, somewhat less discriminating. The faces that do not make the cut go to a sibling _trash folder, not deleted.",
        "Finds faces that are duplicates of another one -- consecutive video frames, mostly -- keeps the sharpest of each group and moves the rest to a sibling _trash folder. At the default threshold of 0 only perceptually identical faces are grouped; raising the threshold catches near-copies too, and grows the groups very fast, so check the _trash folder before training.",
        "Keeps a target number of faces chosen to be as different from each other as possible in head angle and lighting, instead of the sharpest or the most frequent. Not a uniform sample: a small target favors the extremes and can leave gaps in between. The rest go to a sibling _trash folder.",
        "Orders by how much each face had to be enlarged from its source frame -- least enlarged first. Faces cut from a handful of pixels and blown up are noise for training, and no other method tells them apart from good ones. Reorders only, nothing is moved.",
        "Orders data_src by how well each head pose is represented in data_dst (or the other way round), best covered first. The tail is the faces at angles the other faceset never uses. Without a reference faceset the order is left unchanged. Reorders only, nothing is moved.",
        "Compares a 32x32 thumbnail of each face instead of every pixel: about five times faster than absolute pixel difference, and it copes with a folder whose faces are not all the same size. A different order, not an approximation of the exact one -- the thumbnail averages away the fine detail the exact method compares.",
    ),
    help="How the faces in the folder are reordered on disk. Some methods also drop faces that do not make the cut -- they are moved to a sibling _trash folder, never deleted outright, so it is always safe to run again with a different method.",
)

_SORT_BY_SIMILAR = FieldDef(
    key="sort-by-similar",
    label="Sort by similar?",
    kind=FIELD_BOOL,
    default=True,
    help="Otherwise sort by dissimilar.",
    enabled_if=("=absolute pixel difference|absolute pixel difference on 32x32 thumbnails, 5x faster",),
)

_SORT_TARGET_COUNT = FieldDef(
    key="target-number-of-faces",
    label="Target number of faces",
    kind=FIELD_INT,
    default=2000,
    enabled_if=("=best faces|best faces faster|most varied faces",),
    help="How many faces to keep. Which ones is up to the method chosen above -- spread across head angles for the two 'best faces', as different from each other as possible for 'most varied faces'. The rest are moved to a sibling _trash folder, not deleted.",
)

_SORT_REF_DIR = FieldDef(
    key="reference-faceset-directory",
    label="Reference faceset",
    kind=FIELD_PATH,
    default=None,
    enabled_if=("=match the other faceset's poses",),
    help="The other faceset, whose head poses this one is compared against. Left empty it is derived from the sibling folder: data_src/aligned pairs with data_dst/aligned and the other way round.",
)

_SORT_THRESHOLD = FieldDef(
    key="duplicate-distance-threshold",
    label="Duplicate distance threshold",
    kind=FIELD_INT,
    default=0,
    valid_range=(0, 64),
    enabled_if=("=duplicate faces",),
    help="How many bits of difference between two faces still counts as a duplicate. The default 0 groups only perceptually identical faces. Raising it catches near-copies, but groups are chains -- if A matches B and B matches C, all three are trashed together even when A and C look nothing alike. On a faceset extracted from video that chain closes fast: measured on 1619 consecutive frames, 0 trashes 441 of them, while 2 trashes more than half the faceset in one group of 1011. Faces dropped here go to a sibling _trash folder.",
)

_SORT_FIELDS = (_SORT_METHOD, _SORT_BY_SIMILAR, _SORT_TARGET_COUNT,
                _SORT_REF_DIR, _SORT_THRESHOLD)

_ENHANCE_GPU_INDEXES = FieldDef(
    key="which-gpu-indexes-to-choose",
    label="GPU indexes",
    kind=FIELD_TEXT,
    default=None,
    help="Comma-separated device indexes, or 'cpu'. Computed at runtime as all detected devices.",
)

_MERGE_ENHANCED_SRC = FieldDef(
    key="merge-data_srcaligned_enhanced-to-data_srcaligned",
    label="Merge aligned_enhanced back to aligned?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown after the enhanced faces are written to the aligned_enhanced sibling folder.",
)

_PACK_PERSON_FACESET = FieldDef(
    key="process-as-person-faceset",
    label="Process as person faceset?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown only when the input directory contains subdirectories. The subdirectory count is logged separately, not embedded in the prompt.",
)

_PACK_DELETE_ORIGINAL = FieldDef(
    key="delete-original-files",
    label="Delete original files?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown after faceset.pak has been written.",
)

_PACK_FIELDS = (_PACK_PERSON_FACESET, _PACK_DELETE_ORIGINAL)

_RESIZE_NEW_SIZE = FieldDef(
    key="new-image-size",
    label="New image size",
    kind=FIELD_INT,
    default=512,
    valid_range=(128, 2048),
    help="Side, in pixels, of the resized faces. The originals are untouched -- the result goes to a sibling aligned_resized folder, merged back only if you say yes to the next question.",
)

_RESIZE_FACE_TYPE = FieldDef(
    key="change-face-type",
    label="Change face type",
    kind=FIELD_CHOICE,
    default="same",
    choices=("h", "mf", "f", "wf", "head", "same"),
    choice_help=(
        "Half face: crops down to eyes, nose and mouth only -- the tightest of the six.",
        "Mid-full face: as half face, extended a little further down over the chin.",
        "Full face: the face up to the eyebrows and chin. The most common training crop.",
        "Whole face: forehead and jaw included. Widening to this needs alignment data the original crop may not have covered.",
        "Head: the whole head, hair included. The widest of the six, and the one most likely to run out of alignment margin from the original extraction.",
        "Keeps the crop the faceset already has -- resizes the image without changing what area of the head it shows.",
    ),
    help="Crops each face to a different area of the head instead of just resizing it -- 'same' keeps the crop used when the faceset was extracted. Widening it (e.g. full to whole face) needs alignment data the original crop may not have covered.",
)

_MERGE_RESIZED_SRC = FieldDef(
    key="merge-data_srcaligned_resized-to-data_srcaligned",
    label="Merge aligned_resized back to aligned?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown after the resized faces are written to the aligned_resized sibling folder.",
)

_MERGE_RESIZED_DST = FieldDef(
    key="merge-data_dstaligned_resized-to-data_dstaligned",
    label="Merge aligned_resized back to aligned?",
    kind=FIELD_BOOL,
    default=True,
    help="Shown after the resized faces are written to the aligned_resized sibling folder.",
)

STEPS = (
    StepDef(
        name="4.1) data_src view aligned result",
        summary="Opens the data_src/aligned folder with the OS's default app, to look through the extracted faces.",
        family="cura-faceset",
        kind=KIND_VIEWER,
        process=PROCESS_BATCH,
        consumes=("faceset_src",),
        optional=True,
        target="{WORKSPACE}/data_src/aligned",
    ),
    StepDef(
        name="4.2) data_src sort",
        summary="Reorders the source faces on disk by the chosen method, moving the ones it drops to a _trash folder.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("sort",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        fields=_SORT_FIELDS,
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util add landmarks debug images",
        summary="Writes a landmarks-annotated *_debug.jpg next to each aligned source face, to spot alignment errors.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--add-landmarks-debug-images",
            )),
        ),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util faceset enhance",
        summary="Runs the face enhancer network over the source faces, then offers to merge the result back in.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("facesettool", "enhance"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        fields=(_ENHANCE_GPU_INDEXES, _MERGE_ENHANCED_SRC),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util faceset metadata restore",
        summary="Reads back the meta.dat saved earlier, restoring each face's alignment data after external editing.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--restore-faceset-metadata",
            )),
        ),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util faceset metadata save",
        summary="Saves each face's alignment data to meta.dat, so the images can be edited externally and restored later.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--save-faceset-metadata",
            )),
        ),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util faceset pack",
        summary="Packs every source face into one faceset.pak file, and by default deletes the loose originals.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--pack-faceset",
            )),
        ),
        fields=_PACK_FIELDS,
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util faceset resize",
        summary="Resizes every source face, or crops it to a different face type, into a sibling aligned_resized folder.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("facesettool", "resize"), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
            )),
        ),
        fields=(_RESIZE_NEW_SIZE, _RESIZE_FACE_TYPE, _MERGE_RESIZED_SRC),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util faceset unpack",
        summary="Expands faceset.pak back into individual face images in the folder.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--unpack-faceset",
            )),
        ),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="4.2) data_src util recover original filename",
        summary="Renames the aligned source faces back to the filename of the frame each one came from.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_src/aligned",
                "--recover-original-aligned-filename",
            )),
        ),
        modifies=("faceset_src",),
        optional=True,
    ),
    StepDef(
        name="5.1) data_dst view aligned results",
        summary="Opens the data_dst/aligned folder with the OS's default app, to look through the extracted faces.",
        family="cura-faceset",
        kind=KIND_VIEWER,
        process=PROCESS_BATCH,
        consumes=("faceset_dst",),
        optional=True,
        target="{WORKSPACE}/data_dst/aligned",
    ),
    StepDef(
        name="5.1) data_dst view aligned_debug results",
        summary="Opens the data_dst/aligned_debug folder, showing detected face boxes and landmarks over each frame.",
        family="cura-faceset",
        kind=KIND_VIEWER,
        process=PROCESS_BATCH,
        consumes=("debug_dst",),
        optional=True,
        target="{WORKSPACE}/data_dst/aligned_debug",
    ),
    StepDef(
        name="5.2) data_dst sort",
        summary="Reorders the destination faces on disk by the chosen method, moving the ones it drops to a _trash folder.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("sort",), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
            )),
        ),
        fields=_SORT_FIELDS,
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.2) data_dst util faceset pack",
        summary="Packs every destination face into one faceset.pak file, and by default deletes the loose originals.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
                "--pack-faceset",
            )),
        ),
        fields=_PACK_FIELDS,
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.2) data_dst util faceset resize",
        summary="Resizes every destination face, or crops it to a different face type, into a sibling aligned_resized folder.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("facesettool", "resize"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
            )),
        ),
        fields=(_RESIZE_NEW_SIZE, _RESIZE_FACE_TYPE, _MERGE_RESIZED_DST),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.2) data_dst util faceset unpack",
        summary="Expands faceset.pak back into individual face images in the folder.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
                "--unpack-faceset",
            )),
        ),
        modifies=("faceset_dst",),
        optional=True,
    ),
    StepDef(
        name="5.2) data_dst util recover original filename",
        summary="Renames the aligned destination faces back to the filename of the frame each one came from.",
        family="cura-faceset",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("util",), args=(
                "--input-dir", "{WORKSPACE}/data_dst/aligned",
                "--recover-original-aligned-filename",
            )),
        ),
        modifies=("faceset_dst",),
        optional=True,
    ),
)
