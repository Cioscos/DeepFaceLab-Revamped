"""Read-only access to the options a trained model saved on disk.

Shown beside the form fields so resuming a model does not mean guessing
what it was trained with. Read-only in the strong sense: nothing here ever
reaches `answers()`. The form sends only what the user touched, precisely
so a value read here can never be written back on top of a newer one
written by another process meanwhile.
"""
import io
import pickle
from pathlib import Path

# The unpickler accepts nothing beyond what the application's own data file
# legitimately contains: plain containers (handled by the pickle opcodes
# themselves) and numpy arrays. Anything else -- and a data file has no
# reason to carry anything else -- is refused rather than constructed.
ALLOWED_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
}

# The largest candidate file this will hand to the unpickler at all. Refusing
# a global does not bound the size or nesting of the plain lists/dicts/numbers
# pickle opcodes build on their own before `find_class` is ever consulted, so
# a corrupted or foreign file could otherwise exhaust memory (or recursion
# depth) on the GUI's own thread. Sized from measurement, not a guess: a
# synthetic file built with the model's own pickler (`core/pickleex.py`) from
# a deliberately demanding but still realistic save -- resolution 640 (the
# field's own maximum), batch size 16, a 3-million-iteration run (the two
# fields that dominate a real file's size: `loss_history`, one entry per
# iteration for the model's whole lifetime, and `sample_for_preview`, the raw
# training-batch arrays) -- comes out to about 463 MiB. This leaves roughly
# 2x headroom over that on top of the fact that the scenario itself is
# already past what real hardware trains at.
MAX_DATA_FILE_BYTES = 1024 * 1024 * 1024  # 1 GiB


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError("refused: %s.%s" % (module, name))


def saved_options(model_dir, model_name, model_class=None):
    """The options of `model_name` in `model_dir`, or None.

    `model_class` narrows the glob to that one class's data file
    ("<name>_<class>_data.dat"). Without it every class's file is a
    candidate and the first in sorted order wins -- fine when only one
    class was ever trained under that name, wrong the moment two are
    present (e.g. `mio_AMP_data.dat` and `mio_SAEHD_data.dat`): the caller
    would show one class's saved options on the other's form. Pass it
    whenever the step's class is known (it always is -- see
    gui/execution/jobs.py::_model_class_from_step); the unscoped glob
    remains only for a caller that genuinely cannot know the class.

    None covers every case where there is nothing trustworthy to show: no
    file (including the instant during which the application's own atomic
    rewrite has unlinked it), a file that cannot be read, a file larger than
    `MAX_DATA_FILE_BYTES` (refused before it is ever opened, on the same
    footing as any other unreadable candidate), a model that has never
    trained (iter == 0, when the saved options are not yet anyone's), and a
    payload shaped differently from what is expected.
    """
    model_dir = Path(model_dir)
    pattern = "%s_%s_data.dat" % (model_name, model_class) if model_class else "%s_*_data.dat" % model_name
    try:
        candidates = sorted(model_dir.glob(pattern))
    except OSError:
        return None
    for path in candidates:
        try:
            if path.stat().st_size > MAX_DATA_FILE_BYTES:
                continue
        except OSError:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        try:
            data = _RestrictedUnpickler(io.BytesIO(raw)).load()
        except Exception:
            continue
        if not isinstance(data, dict) or not data.get("iter"):
            continue
        options = data.get("options")
        if isinstance(options, dict):
            return options
    return None
