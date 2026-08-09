"""Qt forms generated from the step catalog.

`StepForm` turns a `StepDef` (`gui.catalog.model`) into a `QWidget` with one
input per `FieldDef`, wired so that every value change re-evaluates every
field's `enabled_if` and disables the ones whose condition no longer holds.
`condition_met`/`_norm` are pure and exported for direct testing; the rest
of the module only touches Qt.

Every field's *logical* value -- what `condition_met` compares against -- is
one of: a bool, a number, a combo box's displayed choice text, or a line
edit's text, and is `None` whenever the field is "unanswered" (a
`default=None` field nobody has touched yet). `answers()`, though, sends a
field if and only if it is in `touched_keys()` -- the set of keys the user
has actually changed, tracked from each control's change signal and from
`set_value()`. An untouched field is left for the launched process to
resolve on its own, which for a model already trained means the value saved
on disk rather than the form's default; sending the default instead would
silently rewrite it. A touched field is still dropped if its logical value
is `None` (nothing meaningful to send) or if `enabled_if` currently disables
it. `FieldDef.choice_values`, when present, is applied only at that last
step, mapping the chosen label to the value the call site actually expects
-- `condition_met` and every other internal computation still work with the
label text, since that is what the catalog's `enabled_if` strings quote.
"""
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLineEdit, QPushButton, QSpinBox, QWidget,
)

from gui.catalog.model import FIELD_BOOL, FIELD_CHOICE, FIELD_FLOAT, FIELD_INT, FIELD_PATH, FIELD_TEXT

_INT_RANGE = (0, 2_000_000_000)
_FLOAT_RANGE = (0.0, 1_000_000_000.0)


def _norm(value):
    """Normalize a value for comparison: bool -> "y"/"n", else str(...).lower()."""
    if isinstance(value, bool):
        return "y" if value else "n"
    return str(value).lower()


def condition_met(cond, values):
    """Evaluate one `enabled_if` condition ("key=v", "key!=v" or "key~=v") against `values`.

    Splits on the first occurrence of one of the three operators -- `=`,
    `!=`, `~=` -- found by locating the first "=" and checking the
    character before it, so "!=" and "~=" are never mistaken for a plain
    "=" whose right-hand side happens to start with "!" or "~". The key may
    be empty (a condition on a field whose own key is ""). `=`/`!=` accept
    `a|b` alternatives on the right; `~=` is a substring test (the
    right-hand side is a substring of the field's normalized value).
    """
    eq = cond.index("=")
    if eq > 0 and cond[eq - 1] in "!~":
        op, key = cond[eq - 1] + "=", cond[:eq - 1]
    else:
        op, key = "=", cond[:eq]
    rhs = cond[eq + 1:]
    field_value = _norm(values.get(key))
    if op == "~=":
        return _norm(rhs) in field_value
    matches = field_value in {_norm(alt) for alt in rhs.split("|")}
    return matches if op == "=" else not matches


def _path_row(initial_text=""):
    """A QLineEdit plus a "Browse..." button that fills it via QFileDialog."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(initial_text)
    row.addWidget(edit)
    browse = QPushButton("Browse…")

    def _pick():
        path, _ = QFileDialog.getOpenFileName(container, "Select file")
        if path:
            edit.setText(path)

    browse.clicked.connect(_pick)
    row.addWidget(browse)
    return container, edit


def _build_bool(field):
    box = QCheckBox()
    box.setChecked(bool(field.default))
    return box, box.isChecked, lambda v: box.setChecked(bool(v)), box.stateChanged


def _build_number(field, is_float):
    box = QDoubleSpinBox() if is_float else QSpinBox()
    lo, hi = field.valid_range if field.valid_range else (_FLOAT_RANGE if is_float else _INT_RANGE)
    box.setRange(lo, hi)
    unset = [field.default is None]
    if field.default is not None:
        box.setValue(field.default)

    def _touched(*_args):
        unset[0] = False

    box.valueChanged.connect(_touched)

    def get():
        return None if unset[0] else box.value()

    def set_(v):
        if v is None:
            box.setValue(box.minimum())
            unset[0] = True
        else:
            box.setValue(v)
            unset[0] = False

    return box, get, set_, box.valueChanged


def _build_choice(field):
    combo = QComboBox()
    blank = field.default is None
    if blank:
        combo.addItem("")
    combo.addItems(list(field.choices))
    if blank:
        combo.setCurrentIndex(0)
    elif field.default in field.choices:
        # Reached only when `blank` is False (the `if blank:` branch above
        # already handles that case), so the index needs no "+1 for the
        # blank placeholder entry" offset here -- unlike its live twin in
        # set_() below, where blank can still be True at call time.
        combo.setCurrentIndex(field.choices.index(field.default))

    def get():
        idx = combo.currentIndex()
        if blank and idx == 0:
            return None
        return combo.currentText()

    def set_(v):
        if v is None and blank:
            combo.setCurrentIndex(0)
        elif v in field.choices:
            combo.setCurrentIndex(field.choices.index(v) + (1 if blank else 0))
        else:
            combo.setCurrentText(str(v))

    return combo, get, set_, combo.currentIndexChanged


def _build_text(field):
    edit = QLineEdit("" if field.default is None else str(field.default))

    def get():
        text = edit.text()
        return text if text != "" else None

    def set_(v):
        edit.setText("" if v is None else str(v))

    return edit, get, set_, edit.textChanged


def _build_path(field):
    container, edit = _path_row("" if field.default is None else str(field.default))

    def get():
        text = edit.text()
        return text if text != "" else None

    def set_(v):
        edit.setText("" if v is None else str(v))

    return container, edit, get, set_, edit.textChanged


def _build_control(field):
    """Return (layout_widget, value_widget, get, set_, changed_signal) for `field`."""
    if field.kind == FIELD_BOOL:
        widget, get, set_, signal = _build_bool(field)
    elif field.kind == FIELD_INT:
        widget, get, set_, signal = _build_number(field, is_float=False)
    elif field.kind == FIELD_FLOAT:
        widget, get, set_, signal = _build_number(field, is_float=True)
    elif field.kind == FIELD_CHOICE:
        widget, get, set_, signal = _build_choice(field)
    elif field.kind == FIELD_PATH:
        container, widget, get, set_, signal = _build_path(field)
        return container, widget, get, set_, signal
    elif field.kind == FIELD_TEXT:
        widget, get, set_, signal = _build_text(field)
    else:
        raise ValueError(f"unknown field kind: {field.kind!r}")
    return widget, widget, get, set_, signal


class StepForm(QWidget):
    """A form generated from a `StepDef`, one row per field, kept in sync with `enabled_if`."""

    def __init__(self, step, parent=None):
        super().__init__(parent)
        self._step = step
        self._controls = {}     # key -> (get, set_)
        self._rows = {}         # key -> the widget placed in the form row
        self._touched = set()   # keys the user actually changed
        self._layout = QFormLayout()
        self.setLayout(self._layout)

        self._input_edit = None
        if step.passthrough:
            container, self._input_edit = _path_row()
            self._layout.addRow("Input file", container)

        self._model_combo = None
        if step.needs_model_name:
            self._model_combo = QComboBox()
            self._model_combo.setEditable(True)
            self._layout.addRow("Model name", self._model_combo)

        for field in step.fields:
            layout_widget, value_widget, get, set_, signal = _build_control(field)
            if field.help:
                value_widget.setToolTip(field.help)
            self._controls[field.key] = (get, set_)
            self._rows[field.key] = layout_widget
            self._layout.addRow(field.label, layout_widget)
            signal.connect(self._revalidate)
            signal.connect(lambda *_a, key=field.key: self._touched.add(key))

        self._revalidate()

    def _raw_values(self):
        return {key: get() for key, (get, _set) in self._controls.items()}

    def _revalidate(self, *_args):
        values = self._raw_values()
        for field in self._step.fields:
            enabled = all(condition_met(c, values) for c in field.enabled_if)
            row = self._rows[field.key]
            row.setEnabled(enabled)
            label = self._layout.labelForField(row)
            if label is not None:
                label.setEnabled(enabled)

    def touched_keys(self):
        """The keys the user actually changed. `answers()` sends only these."""
        return set(self._touched)

    def answers(self):
        """The enabled fields the user touched, `{key: python_value}`.

        Only touched fields: an untouched one must leave the decision to the
        process, which for a model already trained resolves it to the value
        saved on disk. Sending the form's own default instead would silently
        rewrite it.
        """
        values = self._raw_values()
        result = {}
        for field in self._step.fields:
            if field.key not in self._touched:
                continue
            if not all(condition_met(c, values) for c in field.enabled_if):
                continue
            raw = values[field.key]
            if raw is None:
                continue
            if field.kind == FIELD_CHOICE and field.choice_values:
                raw = field.choice_values[field.choices.index(raw)]
            result[field.key] = raw
        return result

    def set_value(self, key, value):
        """Set the widget of field `key` to `value`, marking it as touched."""
        _get, set_ = self._controls[key]
        set_(value)
        self._touched.add(key)
        self._revalidate()

    def extra_args(self):
        """The passthrough input file, as a one-element tuple, or `()`."""
        if self._input_edit is not None and self._input_edit.text():
            return (self._input_edit.text(),)
        return ()

    def set_input_file(self, path):
        """Set the passthrough input file picker, if this step has one."""
        if self._input_edit is not None:
            self._input_edit.setText(path)

    def set_saved_values(self, values):
        """Annotate each field that has a saved value with what it is.

        Purely informative: nothing here changes a widget's value, and
        nothing reaches `answers()`. Every label is reset to its plain form
        first -- called again with a different (or empty) `values` after the
        model selection changes, the previous model's annotations must not
        survive on fields the new `values` says nothing about.
        """
        for field in self._step.fields:
            row = self._rows[field.key]
            label = self._layout.labelForField(row)
            if label is None:
                continue
            label.setText(field.label)
            if field.option and field.option in values:
                label.setText("%s  (saved: %s)" % (field.label, values[field.option]))

    def set_model_names(self, names):
        """Populate the "Model name" combo box, if this step has one."""
        if self._model_combo is not None:
            self._model_combo.clear()
            self._model_combo.addItems(list(names))
            if names:
                self._model_combo.setCurrentIndex(0)

    def model_name(self):
        """The chosen/typed model name, or "" if this step has no such field."""
        return self._model_combo.currentText() if self._model_combo is not None else ""
