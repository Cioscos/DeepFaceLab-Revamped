"""Workspace management: creation, guarded clearing, artifact and stage state.

A workspace is a folder with the three standard subdirectories the catalog's
artifacts resolve `{WORKSPACE}` against. This module never imports Qt at
module scope -- `RecentWorkspaces` only needs it for its real default
backend, and that import is deferred into the constructor so tests can pass
an in-memory settings stand-in without Qt being initialized.
"""
import glob
import json
import shutil
from pathlib import Path

from gui.catalog import all_steps
from gui.catalog.artifacts import ARTIFACTS
from gui.catalog.model import STAGES

STANDARD_SUBDIRS = ("data_src", "data_dst", "model")

STATE_DONE = "done"
STATE_READY = "ready"
STATE_BLOCKED = "blocked"

_RECENT_KEY = "recentWorkspaces"
_RECENT_MAX = 10


def default_workspace(dfl_root):
    """The workspace next to the installation: <package root>/workspace.

    Mirrors setenv.sh/.bat, where WORKSPACE is resolved as $INTERNAL/../workspace
    and dfl_root is $INTERNAL/DeepFaceLab -- so the package root is
    dfl_root's grandparent.
    """
    return Path(dfl_root).parent.parent / "workspace"


def create_workspace(path):
    """Create the three standard subdirectories (mkdir -p each)."""
    path = Path(path)
    for name in STANDARD_SUBDIRS:
        (path / name).mkdir(parents=True, exist_ok=True)


def clear_workspace(path):
    """Empty and recreate the three standard subdirectories.

    Refuses (ValueError) a directory that does not already look like a
    workspace -- i.e. has none of the standard subdirectories -- so this
    can never be pointed at an arbitrary folder and wipe it.
    """
    path = Path(path)
    if not any((path / name).is_dir() for name in STANDARD_SUBDIRS):
        raise ValueError(
            "%s does not look like a workspace (none of %s exists)"
            % (path, ", ".join(STANDARD_SUBDIRS))
        )
    for name in STANDARD_SUBDIRS:
        subdir = path / name
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)


def artifact_present(artifact, workspace):
    """True if any of the artifact's patterns resolves to something on disk.

    A pattern matches when it resolves (glob, with {WORKSPACE} substituted)
    to a file, or to a directory that exists and is not empty -- an empty
    directory does not count as the artifact being present.
    """
    for pattern in artifact.patterns:
        resolved = pattern.replace("{WORKSPACE}", str(workspace))
        for match in glob.glob(resolved):
            matched_path = Path(match)
            if matched_path.is_dir():
                if any(matched_path.iterdir()):
                    return True
            else:
                return True
    return False


def _present_checker(workspace):
    """A `present(artifact_name) -> bool` closure bound to `workspace`.

    Shared by `stage_states` and `step_state` so the one rule that is not
    obvious from the artifact's own patterns -- an installation asset (the
    generic XSeg model, pretrain faces) lives under {INTERNAL}, outside the
    workspace, and counts as always present -- lives in a single place. Two
    copies of this would have been the same duplication risk `step_state`
    was written to avoid on the consumes/produces predicate itself.
    """
    artifacts_by_name = {a.name: a for a in ARTIFACTS}

    def present(name):
        artifact = artifacts_by_name[name]
        if artifact.origin == "asset":
            return True
        return artifact_present(artifact, workspace)

    return present


def step_state(step, workspace):
    """The STATE_* of a single step, from its own consumes/produces alone.

    Not `stage_states`' aggregate: a stage is READY as soon as *any* of its
    steps can run, which is the right question for the pipeline bar's pill
    but the wrong one for a single row in the step list -- two steps of the
    same stage can have independent prerequisites (Extraction's "4) data_src
    faceset extract" consumes frame_src, "5) data_dst faceset extract"
    consumes frame_dst), and with only one of the two present the stage is
    already READY while the other step still has nothing to run on. STATE_DONE
    if the step produces at least one artifact and every one is present;
    otherwise STATE_READY if it consumes at least one artifact and every one
    is present; otherwise STATE_BLOCKED. An empty `produces` never counts as
    DONE by vacuous truth -- the same rule `stage_states` applies to its own
    aggregate, applied here to one step instead of a stage's whole scored set.
    """
    present = _present_checker(workspace)
    if step.produces and all(present(name) for name in step.produces):
        return STATE_DONE
    if step.consumes and all(present(name) for name in step.consumes):
        return STATE_READY
    return STATE_BLOCKED


def stage_states(workspace):
    """The STATE_* of each of the 8 STAGES, from artifacts on disk.

    The steps scored for a stage are its non-optional steps; if it has none
    (Faceset curation, XSeg, Export are entirely optional), its optional
    steps are scored instead -- a stage is never exempted from this check
    just because none of its steps happen to be required. STATE_DONE if the
    scored steps produce at least one artifact and every one of them is
    present; otherwise STATE_READY if at least one scored step has every
    artifact it consumes present; otherwise STATE_BLOCKED. An empty set of
    produced artifacts never counts as "done" by vacuous truth -- a stage
    whose scored steps produce nothing (e.g. Faceset curation, whose steps
    only consume/modify) can be READY or BLOCKED but never DONE.

    This aggregates across every scored step (a union of what they produce,
    an "any" over what any one of them can already consume) rather than
    reusing `step_state` per step -- a stage's DONE means the union of
    everything its scored steps produce is on disk, which is a different
    question from "is any single one of them individually DONE". The two
    functions share `_present_checker` instead, which is the part that
    would actually have been a duplication risk.
    """
    present = _present_checker(workspace)

    steps_by_stage = {}
    for step in all_steps():
        stage = step.stage
        if not stage:
            continue
        steps_by_stage.setdefault(stage, []).append(step)

    states = {}
    for stage in STAGES:
        steps = steps_by_stage.get(stage, ())
        required = [s for s in steps if not s.optional]
        scored = required if required else list(steps)

        produced = {name for s in scored for name in s.produces}
        if produced and all(present(name) for name in produced):
            states[stage] = STATE_DONE
            continue

        ready = any(
            s.consumes and all(present(name) for name in s.consumes)
            for s in scored
        )
        states[stage] = STATE_READY if ready else STATE_BLOCKED

    return states


def saved_model_names(model_dir):
    """The unique model-name prefixes of '*_data.dat' files in model_dir.

    A saved model's data file is named '<model_name>_<ModelClass>_data.dat'
    (models/ModelBase.py::get_strpath_storage_for_file); the prefix is
    everything before the last underscore preceding '_data.dat', which
    tolerates an underscore inside the model name itself.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return []
    names = set()
    for f in model_dir.iterdir():
        if not f.is_file() or not f.name.endswith("_data.dat"):
            continue
        suffix_start = f.name.rfind("_data.dat")
        prefix_end = f.name.rfind("_", 0, suffix_start)
        if prefix_end == -1:
            continue
        names.add(f.name[:prefix_end])
    return sorted(names)


class RecentWorkspaces:
    """The recently-opened workspace paths, most recent first, deduplicated.

    Persisted as a JSON-encoded list of strings under one QSettings key --
    the simplest representation that survives a round trip through both
    QSettings and a plain dict-like test double. `settings` is injectable
    (anything with `.value(key, default)` / `.setValue(key, value)`); the
    real default, QSettings("DeepFaceLab", "gui"), is imported lazily so
    importing this module never requires Qt to be initialized.
    """

    def __init__(self, settings=None):
        if settings is None:
            from PyQt5.QtCore import QSettings
            settings = QSettings("DeepFaceLab", "gui")
        self._settings = settings

    def add(self, path):
        paths = self.paths()
        text = str(path)
        if text in paths:
            paths.remove(text)
        paths.insert(0, text)
        del paths[_RECENT_MAX:]
        self._settings.setValue(_RECENT_KEY, json.dumps(paths))

    def paths(self):
        raw = self._settings.value(_RECENT_KEY, "[]")
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return [str(p) for p in decoded] if isinstance(decoded, list) else []
