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

`self.fascia` (`gui.fascia_aiuto.FasciaAiuto`) is the help strip fixed under
the fields: `gui.fascia_aiuto.osserva()` hooks every field's widget (the
container, for a path field -- that is what the mouse actually crosses) so
hovering it, focusing it from the keyboard, or highlighting one of its
dropdown entries shows that field's `help`/`choice_help` there. The row's
*name* and the row container are hooked to the same filter, so the whole
row answers and not only the box where the value is typed -- pointing at
the name is what a person does when the name is what they do not
understand. It only
reads -- it never marks a field touched, so `answers()` keeps sending only
what the user actually changed. Every combo entry also carries its
`choice_help` as a native `Qt.ToolTipRole`, set in `_build_choice`.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QWidget,
)

from gui import testi
from gui import theme
from gui.catalog.model import FIELD_BOOL, FIELD_CHOICE, FIELD_FLOAT, FIELD_INT, FIELD_PATH, FIELD_TEXT
from gui.fascia_aiuto import FasciaAiuto, osserva

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
    browse = QPushButton(testi.BROWSE)
    browse.setToolTip(testi.BROWSE_TIP)

    def _pick():
        path, _ = QFileDialog.getOpenFileName(container, testi.DIALOG_SELECT_FILE)
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
    combo = theme.tendina()
    blank = field.default is None
    if blank:
        combo.addItem("")
    combo.addItems(list(field.choices))
    for indice, aiuto in enumerate(field.choice_help):
        # Same offset as `blank` above: the placeholder entry at index 0
        # has no counterpart in `choice_help`, so every real choice's
        # tooltip is shifted by one when there is one.
        combo.setItemData(indice + (1 if blank else 0), aiuto, Qt.ToolTipRole)
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


_UNREPRESENTABLE = object()


def _for_widget(field, value):
    """A saved option as the widget of `field` wants it, or _UNREPRESENTABLE.

    The options on disk are not plain Python: the data file stores numpy
    scalars (`resolution` is an np.int64), which Qt's setters refuse. And a
    choice the combo box does not offer cannot be shown at all -- better to
    say so than to leave the widget on whatever it happened to hold.
    """
    if value is None:
        return None
    if field.kind == FIELD_BOOL:
        return bool(value)
    if field.kind == FIELD_CHOICE:
        if field.choice_values:
            # The widget shows labels; the model saves the mapped value.
            if value not in field.choice_values:
                return _UNREPRESENTABLE
            return field.choices[field.choice_values.index(value)]
        return value if value in field.choices else _UNREPRESENTABLE
    if field.kind in (FIELD_INT, FIELD_FLOAT):
        try:
            return int(value) if field.kind == FIELD_INT else float(value)
        except (TypeError, ValueError):
            return _UNREPRESENTABLE
    return str(value)


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


def _field_row(control_widget):
    """Wrap a field's control in `[control][badge]`, the widget that goes in the form row.

    The badge is hidden until `set_saved_values` has something to show it.
    This container -- not the control alone -- is what `_revalidate` enables
    and disables through `_rows[key]`, so the badge grays out with the rest
    of a disabled row.
    """
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(control_widget, 1)
    badge = QLabel("")
    badge.setProperty("ruolo", "pastiglia")
    badge.setVisible(False)
    row.addWidget(badge)
    return container, badge


class StepForm(QWidget):
    """A form generated from a `StepDef`, one row per field, kept in sync with `enabled_if`."""

    def __init__(self, step, parent=None):
        super().__init__(parent)
        self._step = step
        self._controls = {}     # key -> (get, set_)
        self._rows = {}         # key -> the widget placed in the form row (control + badge)
        self._badges = {}       # key -> the saved-value pill beside the control
        self._touched = set()   # keys the user actually changed
        self._layout = QFormLayout()
        self.setLayout(self._layout)

        self._input_edit = None
        if step.passthrough:
            container, self._input_edit = _path_row()
            self._layout.addRow(testi.INPUT_FILE, container)

        self._model_combo = None
        if step.needs_model_name:
            self._model_combo = theme.tendina()
            self._model_combo.setEditable(True)
            self._layout.addRow(testi.MODEL_NAME, self._model_combo)

        # Sections group the fields by what a user is looking for
        # (`StepDef.sections`); a step with none gets one untitled group, in
        # declaration order -- the flat form it always had. `hover_widgets`
        # keeps the bare control (the same object `osserva` was hooked to
        # before this row grew a badge) so the fascia's per-field wiring,
        # combo-box highlighting included, is untouched by the wrapping.
        by_key = {field.key: field for field in step.fields}
        groups = step.sections or (("", tuple(field.key for field in step.fields)),)
        hover_widgets = {}
        # The whole row is the hover target, not just the control: the label
        # is built here instead of letting `addRow(str, ...)` build one
        # inside Qt, which is what made it unreachable before -- a name with
        # nothing behind it is exactly what a person points at to find out
        # what a field means.
        self._labels = {}

        for title, keys in groups:
            if title:
                section_label = QLabel(title)
                section_label.setProperty("ruolo", "sezione")
                self._layout.addRow(section_label)
            for field_key in keys:
                field = by_key[field_key]
                layout_widget, value_widget, get, set_, signal = _build_control(field)
                if field.help:
                    value_widget.setToolTip(field.help)
                self._controls[field.key] = (get, set_)
                row, badge = _field_row(layout_widget)
                self._badges[field.key] = badge
                self._rows[field.key] = row
                hover_widgets[field.key] = layout_widget
                label = QLabel(field.label)
                if field.help:
                    label.setToolTip(field.help)
                self._labels[field.key] = label
                self._layout.addRow(label, row)
                signal.connect(self._revalidate)
                signal.connect(lambda *_a, key=field.key: self._touched.add(key))

        # `self.fascia` is built and wired here, but not placed in this
        # form's own layout: `StepView` (`gui.main_window`) puts it in a
        # fixed row of its own, outside the scroll area this form sits in,
        # so the strip stays visible while the fields above it scroll.
        self.fascia = FasciaAiuto()
        self.fascia.riposo(testi.HELP_REST)
        for field in step.fields:
            # Same offset `_build_choice` applies to the combo's items: a
            # field with no default carries a blank entry at index 0, and
            # without this the strip would explain the *previous* value --
            # the fastest way to make the one surface meant to explain lie.
            scarto = 1 if field.kind == FIELD_CHOICE and field.default is None else 0
            osserva(hover_widgets[field.key], self.fascia, field.label, field.help,
                    per_voce=field.choice_help, scarto=scarto,
                    anche=(self._labels[field.key], self._rows[field.key]))

        self._revalidate()

    def section_titles(self):
        """The titles of this form's sections, in order -- `[]` for a flat form."""
        return [title for title, _keys in self._step.sections]

    def row_keys(self):
        """Field keys in the order their rows sit in the form (section order,
        when the step has sections; declaration order otherwise)."""
        return list(self._rows.keys())

    def saved_badge(self, key):
        """The pill beside field `key` that shows its value on disk."""
        return self._badges[key]

    def label_widget(self, key):
        """The name shown to the left of field `key`'s control.

        Built here rather than by `QFormLayout.addRow(str, ...)` precisely
        so it can be reached: it is a hover surface for the help strip, and
        one Qt builds inside itself is not.
        """
        return self._labels[key]

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

    def set_remembered_values(self, values):
        """Preload the values this project remembers for this step.

        Enters through the same door as the model's own saved values --
        fill the widget, then drop the key from `_touched` -- because the
        same invariant holds either way: what the user has not touched is
        never sent to the launched process. It does not use `set_value`,
        which marks the key as touched instead -- the right behavior for
        something the user actually did, the wrong one for a preload.

        Must be called BEFORE `set_saved_values`: when both have something
        to say about a field, the truth is what the model actually has on
        disk. No badge appears next to the field for this -- the badge
        means "this is what the model has saved", and a value remembered
        by the project is not that.
        """
        for field in self._step.fields:
            if field.key not in values:
                continue
            self._write_untouched(field, values[field.key])

    def set_saved_values(self, values):
        """Show what the chosen model was trained with: in the widget, and
        in a badge beside it.

        The widget is what a person reads. Putting the saved value only in
        the label was the first attempt, and it read as if resuming a
        trained model had forgotten its settings -- the fields all showed
        the catalog's defaults, which for a model trained at resolution 224
        with batch 12 said 128 and 8. The badge that shows it now replaced
        the label's own "(saved: ...)" suffix -- same reason to exist, its
        own widget instead of borrowed label text.

        Filling them in does *not* make them answers: `_touched` is cleared
        for every field written here, so `answers()` still sends only what
        the user changed by hand. That separation is the whole point --
        showing a value must never be a way of writing it back.

        A field whose option is absent from `values` goes back to its
        declared default, badge included: this is also the path taken
        when the model name changes, and the previous model's settings must
        not stay on screen under a new name.
        """
        for field in self._step.fields:
            badge = self._badges[field.key]
            badge.setVisible(False)
            if not field.option:
                continue
            if field.option not in values:
                self._write_untouched(field, field.default)
                continue
            value = values[field.option]
            badge.setText(testi.saved_value(value))
            badge.setVisible(True)
            if not self._write_untouched(field, value):
                self._write_untouched(field, field.default)
        self._revalidate()

    def _write_untouched(self, field, value):
        """Put `value` in `field`'s widget without counting as a user edit.

        False when the widget did not take it as given. A QSpinBox clamps to
        its range in silence and a QDoubleSpinBox rounds to its decimals, so
        "it did not raise" is not the same as "it holds what the model
        holds" -- and a field showing a reshaped value would be a field
        lying about the model. The caller puts the declared default back in
        that case; the label goes on telling the truth either way.
        """
        get_, set_ = self._controls[field.key]
        wanted = _for_widget(field, value)
        if wanted is _UNREPRESENTABLE:
            return False
        try:
            set_(wanted)
        except (TypeError, ValueError, OverflowError):
            return False
        finally:
            # After set_, always: the control's change signal is delivered
            # synchronously and adds the key to `_touched` exactly as a
            # human edit would. This is the line that keeps a shown value
            # from becoming a sent one.
            self._touched.discard(field.key)
        return get_() == wanted

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
