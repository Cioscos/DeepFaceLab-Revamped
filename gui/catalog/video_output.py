"""Family video-output: the four "8) merged to *" steps.

Each script issues two `videoed video-from-sequence` invocations in
sequence — the video from `merged`, then the mask from `merged_mask` with
`--lossless` — the only family where one step means two invocations.
Bitrate is the only prompt that can ever surface, and only on the video
invocation of the two non-lossless containers.
"""
from gui.catalog.model import (
    FIELD_INT, KIND_MAIN, PROCESS_BATCH, PROCESS_PROMPT, FieldDef,
    Invocation, StepDef,
)

_BITRATE = FieldDef(
    key="bitrate-of-output-file-in-mbs",
    label="Bitrate of output file (MB/s)",
    kind=FIELD_INT,
    default=16,
)

_CONSUMES = ("merged", "merged_mask", "video_dst")
_PRODUCES = ("risultato", "risultato_mask")

STEPS = (
    StepDef(
        name="8) merged to avi",
        family="video-output",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged",
                "--output-file", "{WORKSPACE}/result.avi",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--include-audio",
            )),
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--output-file", "{WORKSPACE}/result_mask.avi",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--lossless",
            )),
        ),
        fields=(_BITRATE,),
        consumes=_CONSUMES,
        produces=_PRODUCES,
    ),
    StepDef(
        name="8) merged to mov lossless",
        family="video-output",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged",
                "--output-file", "{WORKSPACE}/result.mov",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--include-audio",
                "--lossless",
            )),
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--output-file", "{WORKSPACE}/result_mask.mov",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--lossless",
            )),
        ),
        consumes=_CONSUMES,
        produces=_PRODUCES,
    ),
    StepDef(
        name="8) merged to mp4 lossless",
        family="video-output",
        kind=KIND_MAIN,
        process=PROCESS_BATCH,
        invocations=(
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged",
                "--output-file", "{WORKSPACE}/result.mp4",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--include-audio",
                "--lossless",
            )),
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--output-file", "{WORKSPACE}/result_mask.mp4",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--lossless",
            )),
        ),
        consumes=_CONSUMES,
        produces=_PRODUCES,
    ),
    StepDef(
        name="8) merged to mp4",
        family="video-output",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged",
                "--output-file", "{WORKSPACE}/result.mp4",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--include-audio",
            )),
            Invocation(verb=("videoed", "video-from-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst/merged_mask",
                "--output-file", "{WORKSPACE}/result_mask.mp4",
                "--reference-file", "{WORKSPACE}/data_dst.*",
                "--lossless",
            )),
        ),
        fields=(_BITRATE,),
        consumes=_CONSUMES,
        produces=_PRODUCES,
    ),
)
