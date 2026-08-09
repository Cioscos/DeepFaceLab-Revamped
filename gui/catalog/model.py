"""Data model of the step catalog, derived at design-time from the workflow formalization.

Changes to the user workflow must be reflected here.
"""
from dataclasses import dataclass

# Step kinds.
KIND_MAIN = "main"
KIND_VIEWER = "viewer"
KIND_CLEAR = "clear"
KIND_EBSYNTH = "ebsynth"

# Processing modes.
PROCESS_BATCH = "batch"
PROCESS_PROMPT = "prompt-poi-batch"
PROCESS_SESSION = "sessione"

# Field types.
FIELD_BOOL = "bool"
FIELD_INT = "int"
FIELD_FLOAT = "float"
FIELD_CHOICE = "choice"
FIELD_PATH = "path"
FIELD_TEXT = "text"

# Workflow stages, displayed in order.
STAGES = ("Video input", "Extraction", "Faceset curation", "XSeg",
          "Training", "Merge", "Video output", "Export")

# Map from family name to stage display name, or "" for menu-only steps.
FAMILY_STAGE = {
    "video-input": "Video input",
    "estrazione": "Extraction",
    "cura-faceset": "Faceset curation",
    "xseg": "XSeg",
    "addestramento": "Training",
    "fusione": "Merge",
    "video-output": "Video output",
    "esportazione": "Export",
    "workspace": "",  # Menu only
}


@dataclass(frozen=True)
class FieldDef:
    """Definition of a single form field for a step.

    Corresponds to a prompt in the step's options.
    """
    key: str                 # prompt_key from the original prompt text
    label: str               # English label shown in GUI
    kind: str                # One of the FIELD_* constants
    # Name of the model option this field feeds, when there is one -- the
    # key under which a trained model saves it in its data file. Not
    # derivable from `key`: of the 38 training fields only 10 have a key
    # that matches the option's name. Empty means "no saved value to show".
    # Placed after the last field without a default, not right after `key`,
    # because a dataclass field with a default cannot precede one without.
    option: str = ""
    default: object = None
    help: str = ""           # From help_message if present
    choices: tuple = ()      # For FIELD_CHOICE
    # When non-empty, the value sent for choices[i] is choice_values[i]
    # instead of the choice string itself -- for a FIELD_CHOICE whose
    # call-site expects something other than the displayed text (e.g. an
    # index into a menu built at runtime).
    choice_values: tuple = ()
    valid_range: tuple = ()  # (min, max) for FIELD_INT/FIELD_FLOAT, () = none
    # AND conditions: "key=v", "key!=v", "key=a|b", "key~=v" ("key~=v" is
    # true when v is a substring of the field's value -- for a free-text
    # field whose real gating condition is itself a substring test, e.g.
    # 'df' in self.options['archi'], not expressible as exact equality).
    enabled_if: tuple = ()


@dataclass(frozen=True)
class Invocation:
    """A single CLI invocation issued by the step."""
    verb: tuple              # e.g. ("videoed", "extract-video") or ("train",)
    args: tuple = ()         # With placeholders {WORKSPACE}/{INTERNAL}/{DFL_ROOT}


@dataclass(frozen=True)
class ArtifactDef:
    """Definition of an artifact (file or directory) in the workflow."""
    name: str
    patterns: tuple          # shell-style patterns resolved with glob (workflow.toml)
    origin: str              # "prodotto" | "fornito-dall-utente" | "asset"


@dataclass(frozen=True)
class StepDef:
    """Definition of a single workflow step.

    A step is one user-facing operation: extract, train, merge, etc.
    """
    name: str                # Identical to commands.toml
    family: str
    kind: str
    process: str
    invocations: tuple = ()
    fields: tuple = ()
    consumes: tuple = ()
    produces: tuple = ()
    modifies: tuple = ()
    mkdirs: tuple = ()
    optional: bool = False
    target: str = ""         # For KIND_VIEWER
    passthrough: bool = False    # "3) cut video": form adds a file picker
    needs_model_name: bool = False  # train/merge/export: GUI appends --force-model-name

    @property
    def stage(self):
        """The workflow stage this step belongs to."""
        return FAMILY_STAGE[self.family]
