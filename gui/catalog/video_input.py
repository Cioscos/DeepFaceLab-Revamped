"""Family video-input: extracting frames from the source/destination videos.

All four steps resolve to the `videoed` subcommand and run their prompts
in-process, before the (synchronous, external) ffmpeg job starts.
"""
from gui.catalog.model import (
    FIELD_CHOICE, FIELD_INT, FIELD_TEXT, KIND_MAIN, PROCESS_PROMPT, FieldDef,
    Invocation, StepDef,
)

_FPS = FieldDef(
    key="enter-fps",
    label="FPS",
    kind=FIELD_INT,
    default=0,
    help="How many frames of every second of the video will be extracted. 0 - full fps",
)

_OUTPUT_FORMAT = FieldDef(
    key="output-image-format",
    label="Output image format",
    kind=FIELD_CHOICE,
    default="png",
    choices=("png", "jpg"),
    help="png is lossless, but extraction is x10 slower for HDD, requires x10 more disk space than jpg.",
)

_FROM_TIME = FieldDef(
    key="from-time",
    label="From time",
    kind=FIELD_TEXT,
    default="00:00:00.000",
)

_TO_TIME = FieldDef(
    key="to-time",
    label="To time",
    kind=FIELD_TEXT,
    default="00:00:00.000",
)

_AUDIO_TRACK_ID = FieldDef(
    key="specify-audio-track-id",
    label="Audio track ID",
    kind=FIELD_INT,
    default=0,
)

_CUT_BITRATE = FieldDef(
    key="bitrate-of-output-file-in-mbs",
    label="Bitrate of output file (MB/s)",
    kind=FIELD_INT,
    default=25,
)

_DENOISE_FACTOR = FieldDef(
    key="denoise-factor",
    label="Denoise factor",
    kind=FIELD_INT,
    default=7,
    valid_range=(1, 20),
)

STEPS = (
    StepDef(
        name="2) extract images from video data_src",
        family="video-input",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("videoed", "extract-video"), args=(
                "--input-file", "{WORKSPACE}/data_src.*",
                "--output-dir", "{WORKSPACE}/data_src",
            )),
        ),
        fields=(_FPS, _OUTPUT_FORMAT),
        consumes=("video_src",),
        produces=("frame_src",),
        mkdirs=("{WORKSPACE}/data_src",),
    ),
    StepDef(
        name="3) cut video (drop video on me)",
        family="video-input",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("videoed", "cut-video"), args=("--input-file",)),
        ),
        fields=(_FROM_TIME, _TO_TIME, _AUDIO_TRACK_ID, _CUT_BITRATE),
        optional=True,
        passthrough=True,
    ),
    StepDef(
        name="3) extract images from video data_dst FULL FPS",
        family="video-input",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("videoed", "extract-video"), args=(
                "--input-file", "{WORKSPACE}/data_dst.*",
                "--output-dir", "{WORKSPACE}/data_dst",
                "--fps", "0",
            )),
        ),
        fields=(_OUTPUT_FORMAT,),
        consumes=("video_dst",),
        produces=("frame_dst",),
        mkdirs=("{WORKSPACE}/data_dst",),
    ),
    StepDef(
        name="3.optional) denoise data_dst images",
        family="video-input",
        kind=KIND_MAIN,
        process=PROCESS_PROMPT,
        invocations=(
            Invocation(verb=("videoed", "denoise-image-sequence"), args=(
                "--input-dir", "{WORKSPACE}/data_dst",
            )),
        ),
        fields=(_DENOISE_FACTOR,),
        modifies=("frame_dst",),
        optional=True,
    ),
)
