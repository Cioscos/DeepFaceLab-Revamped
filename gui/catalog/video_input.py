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
    choice_help=(
        "Lossless. Best quality for training, but roughly ten times slower to write and ten times the disk space of jpg on a spinning disk.",
        "Lossy but fast and compact. The practical default unless disk space and write speed are not a concern.",
    ),
    help="png is lossless, but extraction is x10 slower for HDD, requires x10 more disk space than jpg.",
)

_FROM_TIME = FieldDef(
    key="from-time",
    label="From time",
    kind=FIELD_TEXT,
    default="00:00:00.000",
    help="Where the cut starts, as HH:MM:SS. Everything before this point is dropped from the video the rest of the pipeline will see.",
)

_TO_TIME = FieldDef(
    key="to-time",
    label="To time",
    kind=FIELD_TEXT,
    default="00:00:00.000",
    help="Where the cut ends, as HH:MM:SS. Everything after this point is dropped from the video the rest of the pipeline will see.",
)

_AUDIO_TRACK_ID = FieldDef(
    key="specify-audio-track-id",
    label="Audio track ID",
    kind=FIELD_INT,
    default=0,
    help="Which audio stream of the source file to keep in the cut, by index starting at 0. A non-zero index silently drops the audio track if the file does not have that many streams -- including an ordinary single-track file, where anything but 0 loses the audio entirely; the video track is always kept.",
)

_CUT_BITRATE = FieldDef(
    key="bitrate-of-output-file-in-mbs",
    label="Bitrate of output file (MB/s)",
    kind=FIELD_INT,
    default=25,
    help="Video quality of the re-encoded cut, in megabits per second. Higher means a larger file and fewer compression artifacts; 25 is a good default for 1080p.",
)

_DENOISE_FACTOR = FieldDef(
    key="denoise-factor",
    label="Denoise factor",
    kind=FIELD_INT,
    default=7,
    valid_range=(1, 20),
    help="How aggressively the destination frames are smoothed before extraction. Higher removes more grain and compression noise, but also erases fine skin detail the model would otherwise learn from.",
)

STEPS = (
    StepDef(
        name="2) extract images from video data_src",
        summary="Turns the source video into the still frames the faces will be extracted from.",
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
        summary="Trims a video to a time range and re-encodes it, keeping one chosen audio track.",
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
        summary="Turns the destination video into frames at its native frame rate, no frames skipped.",
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
        summary="Smooths grain and compression noise out of the destination frames before extraction.",
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
