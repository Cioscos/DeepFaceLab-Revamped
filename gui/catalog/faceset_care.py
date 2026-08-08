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
    FIELD_BOOL, FIELD_CHOICE, FIELD_INT, FIELD_TEXT, KIND_MAIN, KIND_VIEWER,
    PROCESS_BATCH, PROCESS_PROMPT, FieldDef, Invocation, StepDef,
)

_SORT_METHODS = (
    "blur", "motion_blur", "face yaw direction", "face pitch direction",
    "face rect size in source image", "histogram similarity",
    "histogram dissimilarity", "brightness", "hue",
    "amount of black pixels", "original filename", "one face in image",
    "absolute pixel difference", "best faces", "best faces faster",
)

# The call-site (Sorter.py: io.input_int("", 5, valid_list=[*range(15)]))
# takes the menu index, not the description string, so choice_values maps
# each _SORT_METHODS entry to its real index in Sorter.sort_func_methods.
_SORT_METHOD = FieldDef(
    key="",
    label="Sorting method",
    kind=FIELD_CHOICE,
    default="histogram similarity",
    choices=_SORT_METHODS,
    choice_values=tuple(range(len(_SORT_METHODS))),
)

_SORT_BY_SIMILAR = FieldDef(
    key="sort-by-similar",
    label="Sort by similar?",
    kind=FIELD_BOOL,
    default=True,
    help="Otherwise sort by dissimilar.",
    enabled_if=("=absolute pixel difference",),
)

_SORT_GPU_INDEX = FieldDef(
    key="which-gpu-index-to-choose",
    label="GPU index",
    kind=FIELD_TEXT,
    default=None,
    help="Index of the device to use, or 'cpu'. Computed at runtime as the single best-scoring detected device.",
    enabled_if=("=absolute pixel difference",),
)

_SORT_TARGET_COUNT = FieldDef(
    key="target-number-of-faces",
    label="Target number of faces",
    kind=FIELD_INT,
    default=2000,
    enabled_if=("=best faces|best faces faster",),
)

_SORT_FIELDS = (_SORT_METHOD, _SORT_BY_SIMILAR, _SORT_GPU_INDEX, _SORT_TARGET_COUNT)

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
)

_RESIZE_FACE_TYPE = FieldDef(
    key="change-face-type",
    label="Change face type",
    kind=FIELD_CHOICE,
    default="same",
    choices=("h", "mf", "f", "wf", "head", "same"),
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
        family="cura-faceset",
        kind=KIND_VIEWER,
        process=PROCESS_BATCH,
        consumes=("faceset_src",),
        optional=True,
        target="{WORKSPACE}/data_src/aligned",
    ),
    StepDef(
        name="4.2) data_src sort",
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
        family="cura-faceset",
        kind=KIND_VIEWER,
        process=PROCESS_BATCH,
        consumes=("faceset_dst",),
        optional=True,
        target="{WORKSPACE}/data_dst/aligned",
    ),
    StepDef(
        name="5.1) data_dst view aligned_debug results",
        family="cura-faceset",
        kind=KIND_VIEWER,
        process=PROCESS_BATCH,
        consumes=("debug_dst",),
        optional=True,
        target="{WORKSPACE}/data_dst/aligned_debug",
    ),
    StepDef(
        name="5.2) data_dst sort",
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
