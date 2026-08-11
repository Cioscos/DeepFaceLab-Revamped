"""The application shell: pipeline bar, step list, step view, console dock.

Wires navigation, execution, telemetry and the three special step kinds.
`select_step`/the pipeline bar/the step view build the picture; everything
that starts a subprocess, reads its output, tails its telemetry file or
touches the workspace on disk lives on `MainWindow` itself, one handler per
behavior in `gui.execution`/`gui.telemetry`/`gui.workspace`'s vocabulary.
"""
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl, QProcess, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QAction, QActionGroup, QApplication, QDockWidget, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTabBar, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from gui import testi
from gui.catalog import all_steps, step_by_name
from gui.catalog.model import (
    KIND_CLEAR, KIND_EBSYNTH, KIND_MAIN, KIND_VIEWER, PROCESS_SESSION, STAGES,
)
from gui.delegato_passi import RUOLO_SOMMARIO, RUOLO_STATO, DelegatoPassi
from gui.execution.jobs import JobManager, StepConflict, _model_class_from_step, _resolve
from gui.forms import StepForm
from gui.preferenze import ScalaTesto
from gui.rimozione import svuota
from gui.saved_options import saved_options
from gui.status_line import (
    RitmoIterazioni, iterazione_utilizzabile, obiettivo_valido, pezzi_di_stato)
from gui.telemetry import EventTail
from gui.theme import CONSOLE_FONT_POINT_SIZE, SCALE_NAMES, STATO_COLORE, apply_dark_theme
from gui.training_panel import TrainingPanel
from gui.workspace import (
    STANDARD_SUBDIRS, STATE_BLOCKED, STATE_DONE, STATE_READY,
    RecentWorkspaces, clear_workspace, create_workspace, default_workspace,
    saved_model_names, stage_reasons, step_reason,
)

# Console tab titles for a failed job: exit code -1 is Job's sentinel for
# "the process never started at all" (see gui.execution.jobs.Job).
_FAILED_TO_START = -1

# Kept as an alias: gui.theme.STATO_COLORE is the single source now that the
# color itself lives in the style sheet as a QPushButton[stato=...] rule
# (see PipelineBar.refresh()), but this name is still what
# tests_gui/test_theme.py reads to check the blocked color's lightness.
_STAGE_COLOR = STATO_COLORE


def _step_label(step):
    """The step list's display text: name, plus the two layout badges."""
    label = step.name
    if step.optional:
        label += testi.step_list_optional_suffix()
    if step.process == PROCESS_SESSION:
        label += testi.step_list_external_window_suffix()
    return label


def _placeholder_text(step, workspace, dfl_root):
    """Description shown in place of a form for a non-main-kind step.

    `step.target` (KIND_VIEWER only) carries the same {WORKSPACE}/{DFL_ROOT}/
    {INTERNAL} placeholders as an invocation's args, resolved the same way
    (`gui.execution.jobs._resolve`) so the shown path is the real one, not
    the catalog template.
    """
    target = _resolve(step.target, workspace, dfl_root) if step.kind == KIND_VIEWER else None
    return testi.placeholder(step.kind, target)


def _is_training_step(step):
    """True if any of `step`'s invocations runs `main.py train`.

    Gates telemetry attachment. Keyed on the invocation's verb rather than
    `step.family`: "5.XSeg) train" belongs to the "xseg" family but still
    invokes `main.py train` and writes the same events channel as every
    other training step, so a family-based gate misses it.
    """
    return any(invocation.verb == ("train",) for invocation in step.invocations)


def _reproducible_command(step, job, workspace, dfl_root, python_exe, extra_args):
    """The exact command line and environment for a job that never started.

    Rebuilds the first invocation the same way `JobManager.try_start` does
    (`_resolve` against the same workspace/dfl_root), rather than reaching
    into the `Job`'s internals -- a process that fails to start always fails
    on its first invocation, since the sequence only advances past one on a
    clean exit.
    """
    invocation = step.invocations[0]
    args = [_resolve(a, workspace, dfl_root) for a in invocation.args]
    args.extend(extra_args)
    program_args = [str(dfl_root / "main.py")] + list(invocation.verb) + args
    command = " ".join([str(python_exe)] + program_args)
    return testi.reproducible_command(
        command, workspace, job.workdir / "answers.json", job.events_path, job.commands_path)


class PipelineBar(QWidget):
    """One button per pipeline stage, colored by `gui.workspace.stage_states`.

    Clicking a button -- or calling `select()` programmatically -- emits
    `stage_selected` with the stage's display name. Colors are computed at
    construction and whenever `refresh()` is called.
    """

    stage_selected = pyqtSignal(str)

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._buttons = {}
        self._selected = None
        self._states = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for stage in STAGES:
            button = QPushButton(stage)
            button.setCheckable(True)
            button.setToolTip(testi.STAGE_TIP)
            button.clicked.connect(lambda _checked, s=stage: self.select(s))
            layout.addWidget(button)
            self._buttons[stage] = button
        self.refresh()

    def stage_names(self):
        """The 8 pipeline stages, in display order."""
        return list(STAGES)

    def select(self, name):
        """Select stage `name`: update button state and emit `stage_selected`."""
        if name not in self._buttons:
            raise KeyError(name)
        for stage, button in self._buttons.items():
            button.setChecked(stage == name)
        self._selected = name
        self.stage_selected.emit(name)

    def selected(self):
        """The currently selected stage name, or None before the first `select()`."""
        return self._selected

    def stage_state(self, name):
        """The STATE_* of `name`, as of the last `refresh()`."""
        return self._states.get(name, STATE_BLOCKED)

    def set_workspace(self, workspace):
        """Point at a different workspace and recompute every stage's color."""
        self._workspace = workspace
        self.refresh()

    def refresh(self):
        """Recompute every stage's color from the workspace's artifacts on disk.

        The color itself is a style sheet rule keyed on the "stato" dynamic
        property (QPushButton[stato="done"] etc., in gui.theme) rather than
        a per-button setStyleSheet: changing a dynamic property does not
        repaint by itself, so it is followed by the unpolish/polish pair Qt
        needs to pick the new rule up.

        The tooltip is rebuilt here too, and says *why* the pill has that
        colour: a colour alone is a dead end -- it shows that nothing can
        run without saying what to do about it, which is the one question
        the pipeline bar exists to answer.
        """
        reasons = stage_reasons(self._workspace)
        self._states = {stage: reason[0] for stage, reason in reasons.items()}
        for stage, button in self._buttons.items():
            state, missing, ready = reasons.get(stage, (STATE_BLOCKED, [], []))
            button.setProperty("stato", state)
            button.setToolTip("\n".join(
                [testi.STAGE_TIP, testi.stage_state_tip(state, missing, ready)]))
            button.style().unpolish(button)
            button.style().polish(button)


class StepView(QWidget):
    """Center panel: the selected step's form (or a placeholder), Start, status.

    `start_button` is the single trigger for every step kind -- what it does
    depends on `step.kind`, decided by `MainWindow`. `open_button` exists
    only for `KIND_VIEWER` (None otherwise): opening a folder is not "start
    a job", so it gets its own control instead of overloading Start.
    """

    def __init__(self, workspace, dfl_root, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._dfl_root = dfl_root
        self.step = None
        self.form = None
        self.placeholder = None
        self.open_button = None
        layout = QVBoxLayout(self)

        self._header = QWidget()
        header = QHBoxLayout(self._header)
        header.setContentsMargins(0, 0, 0, 0)
        self._step_name_label = QLabel("")
        header.addWidget(self._step_name_label)
        self._badge_optional = QLabel(testi.BADGE_OPTIONAL)
        self._badge_optional.setProperty("ruolo", "pastiglia")
        header.addWidget(self._badge_optional)
        self._badge_external = QLabel(testi.BADGE_EXTERNAL_WINDOW)
        self._badge_external.setProperty("ruolo", "pastiglia")
        header.addWidget(self._badge_external)
        header.addStretch(1)
        layout.addWidget(self._header)
        self._header.setVisible(False)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        # A long form (36 fields, six section titles, a help strip) can ask
        # for far more height than any real window offers -- without a
        # scroll area nothing below the fold is reachable, Start included.
        # Only the body scrolls: Start and the job strip stay put below it,
        # always on screen, because they are the command, not a field to
        # scroll past.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidget(self._body)
        layout.addWidget(self._scroll, 1)
        # The current form's help strip (`StepForm.fascia`) lives here, not
        # inside `self._body`: it is the surface that explains a field to
        # someone who does not know to go looking for a tooltip, so it has
        # to stay on screen next to whatever the mouse is over, even when
        # the fields above it are scrolled out of view. `set_step()` moves
        # each form's strip in here and clears the previous one out before
        # the old form is torn down; a step with no form leaves this empty
        # and it collapses to no height, no gap left behind.
        self._fascia_holder = QWidget()
        self._fascia_layout = QVBoxLayout(self._fascia_holder)
        self._fascia_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._fascia_holder)
        self.start_button = QPushButton(testi.START)
        self.start_button.setObjectName("start")
        self.start_button.setToolTip(testi.START_TIP)
        layout.addWidget(self.start_button)
        self.job_strip = QWidget()
        strip = QHBoxLayout(self.job_strip)
        strip.setContentsMargins(0, 0, 0, 0)
        self.job_status = QLabel("")
        strip.addWidget(self.job_status, 1)
        # Grouped by what each button targets, like the trainer's own
        # commands: the view (console), the model on disk (save/backup),
        # the run itself (stop/force stop).
        self.console_button = QPushButton(testi.SHOW_CONSOLE)
        self.console_button.setToolTip(testi.SHOW_CONSOLE_TIP)
        strip.addWidget(self.console_button)
        self.save_button = QPushButton(testi.SAVE)
        self.save_button.setToolTip(testi.SAVE_TIP)
        strip.addWidget(self.save_button)
        self.backup_button = QPushButton(testi.BACKUP)
        self.backup_button.setToolTip(testi.BACKUP_TIP)
        strip.addWidget(self.backup_button)
        self.stop_button = QPushButton(testi.STOP)
        self.stop_button.setProperty("ruolo", "stop")
        self.stop_button.setToolTip(testi.STOP_TIP)
        strip.addWidget(self.stop_button)
        self.force_stop_button = QPushButton(testi.FORCE_STOP)
        self.force_stop_button.setToolTip(testi.FORCE_STOP_TIP)
        strip.addWidget(self.force_stop_button)
        layout.addWidget(self.job_strip)
        self.job_strip.setVisible(False)
        self.job = None
        self.is_training = False
        self.status_panel = QLabel("")
        self.status_panel.setWordWrap(True)
        layout.addWidget(self.status_panel)

    def set_workspace(self, workspace):
        """Point at a different workspace; takes effect on the next `set_step`."""
        self._workspace = workspace

    def set_job(self, job, is_training):
        """Show (or hide, with job=None) the strip of the job of this step.

        Save/Backup and Force stop only make sense for a training run: the
        first two because nothing else reads the command channel, the third
        because for every other step Stop already is the hard kill. The four
        action buttons are also shown only while the job is still running --
        a finished job has nothing left for any of them to do. Disabling
        them instead of hiding them was the first attempt and read wrong on
        screen: a greyed-out row of buttons still says "these are the things
        you can do here", when in fact the process they addressed no longer
        exists. Console is the exception and stays: its history is worth
        reopening long after the job itself has ended, which is also why the
        strip as a whole does not disappear with the job.
        """
        self.job = job
        self.is_training = is_training
        self.job_strip.setVisible(job is not None)
        if job is None:
            return
        self.stop_button.setVisible(job.running)
        for b in (self.save_button, self.backup_button, self.force_stop_button):
            b.setVisible(is_training and job.running)
        self.job_status.setText(testi.JOB_RUNNING if job.running else testi.JOB_FINISHED)

    def set_step(self, step):
        """Rebuild the panel for `step` (a StepDef), or clear it for None."""
        self.step = step
        self.set_job(None, False)
        # Through `gui.rimozione`, like every other place that empties a
        # layout: a widget detached without being hidden first is a
        # top-level window Qt shows again by itself (see that module).
        svuota(self._body_layout)
        # The previous step's help strip, if any, belongs to a `StepForm`
        # about to be discarded (or to no form at all, for a viewer/clear
        # step): detach it before that happens, so a stale strip never
        # lingers in the fixed row below the scroll area.
        svuota(self._fascia_layout)
        self.open_button = None
        if step is None:
            self.form = None
            self.placeholder = None
            self.status_panel.setText("")
            self._header.setVisible(False)
            self._step_name_label.setText("")
            self._badge_optional.setVisible(False)
            self._badge_external.setVisible(False)
            return
        self.start_button.setVisible(step.kind != KIND_VIEWER)
        if step.kind == KIND_MAIN:
            self.form = StepForm(step)
            self.placeholder = None
            if step.needs_model_name:
                self.form.set_model_names(saved_model_names(self._workspace / "model"))
                self.form._model_combo.currentTextChanged.connect(self._refresh_saved_values)
            self._body_layout.addWidget(self.form)
            self._fascia_layout.addWidget(self.form.fascia)
            self._refresh_saved_values()
        else:
            self.form = None
            self.placeholder = QLabel(_placeholder_text(step, self._workspace, self._dfl_root))
            self.placeholder.setWordWrap(True)
            self._body_layout.addWidget(self.placeholder)
            if step.kind == KIND_VIEWER:
                self.open_button = QPushButton(testi.OPEN_FOLDER)
                self.open_button.setToolTip(testi.OPEN_FOLDER_TIP)
                target = _resolve(step.target, self._workspace, self._dfl_root)
                self.open_button.clicked.connect(
                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(target)))
                self._body_layout.addWidget(self.open_button)
        # The two layout badges move here, beside the step's name, and leave
        # status_panel for what it is meant to hold: what actually happened
        # (a failed job's last lines, telemetry) rather than a fact about
        # the catalog that never changes while this step is on screen.
        self._header.setVisible(True)
        self._step_name_label.setText(step.name)
        self._badge_optional.setVisible(step.optional)
        self._badge_external.setVisible(step.process == PROCESS_SESSION)
        self.status_panel.setText("")

    def _refresh_saved_values(self):
        """Show the chosen model's saved options beside the fields.

        Called unconditionally, even when there is nothing to show: the
        model name just changed (typed, or picked from the combo), and a
        previous model's annotations must not be left on screen labelled
        as this one's saved values. set_saved_values(values or {}) is what
        clears them.
        """
        if self.form is None or not self.step.needs_model_name:
            return
        model_class = _model_class_from_step(self.step)
        values = saved_options(self._workspace / "model", self.form.model_name(), model_class)
        self.form.set_saved_values(values or {})


class MainWindow(QMainWindow):
    """Top-level window: pipeline bar, step list, step view, console dock, menus.

    Owns the `JobManager` and dispatches `start_button` by `step.kind`:
    `KIND_MAIN` runs a job, `KIND_CLEAR` runs the guarded workspace wipe,
    `KIND_EBSYNTH` launches the bundled application detached, `KIND_VIEWER`
    has no Start action at all (`open_button` covers it).
    """

    def __init__(self, python_exe, dfl_root, workspace=None, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle(testi.WINDOW_TITLE)
        self._python_exe = python_exe
        self._dfl_root = Path(dfl_root)
        self.workspace = Path(workspace) if workspace is not None else default_workspace(self._dfl_root)
        self._scala = ScalaTesto(settings)
        self.job_manager = JobManager(python_exe, self._dfl_root)
        self.job_manager.job_started.connect(lambda _job: self._refresh_running_jobs_menu())
        self.job_manager.job_finished.connect(self._on_any_job_finished)

        self.pipeline_bar = PipelineBar(self.workspace)
        self.step_list = QListWidget()
        self.step_list.setItemDelegate(DelegatoPassi(self.step_list))
        self.step_view = StepView(self.workspace, self._dfl_root)

        passi = QWidget()
        passi_layout = QVBoxLayout(passi)
        passi_layout.addWidget(self.pipeline_bar)
        body = QSplitter(Qt.Horizontal)
        body.addWidget(self.step_list)
        body.addWidget(self.step_view)
        passi_layout.addWidget(body, 1)

        self.central_tabs = QTabWidget()
        self.central_tabs.setTabsClosable(True)
        self.central_tabs.tabCloseRequested.connect(self._on_central_tab_close_requested)
        self.central_tabs.addTab(passi, testi.TAB_STEPS)
        # The "Steps" tab has no close button: it is the window itself.
        self.central_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.setCentralWidget(self.central_tabs)

        self.console_dock = QDockWidget(testi.CONSOLE_DOCK, self)
        self.console_tabs = QTabWidget()
        self.console_tabs.setTabsClosable(True)
        self.console_tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.console_dock.setWidget(self.console_tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.console_dock.hide()

        self._panels = {}          # Job -> TrainingPanel, alive with its tab closed
        self._consoles = {}        # Job -> QTextEdit, or None when the view is closed
        self._jobs_in_order = []   # jobs in the order they were started
        self._tails = []           # active EventTail instances
        self._job_status_text = {}  # Job -> the strip's status text, survives navigation
        self._stop_requested = set()  # jobs a clean Stop was asked of (the save/close telemetry applies to)
        self._shutdown_pending = False  # True once the user confirmed closing with jobs active
        self._shutdown_dialog = None    # non-modal "closing..." dialog, while jobs wind down

        self.pipeline_bar.stage_selected.connect(self._on_stage_selected)
        self.step_list.currentItemChanged.connect(self._on_item_changed)
        self.step_view.start_button.clicked.connect(self._on_start_clicked)
        self.step_view.stop_button.clicked.connect(self._on_stop_clicked)
        self.step_view.force_stop_button.clicked.connect(self._on_force_stop_clicked)
        self.step_view.save_button.clicked.connect(
            lambda: self._send_to_current_job("save"))
        self.step_view.backup_button.clicked.connect(
            lambda: self._send_to_current_job("backup"))
        self.step_view.console_button.clicked.connect(self._on_console_button_clicked)

        self._build_menus()
        self.pipeline_bar.select(self.pipeline_bar.stage_names()[0])

    def _on_any_job_finished(self, _job, _code):
        self.pipeline_bar.refresh()
        self._refresh_step_badges()
        self._refresh_running_jobs_menu()

    def select_step(self, name):
        """Navigate to step `name`: select its stage (if it has one) and build its view."""
        step = step_by_name(name)
        if step.stage:
            if self.pipeline_bar.selected() != step.stage:
                self.pipeline_bar.select(step.stage)
            for row in range(self.step_list.count()):
                item = self.step_list.item(row)
                if item.data(Qt.UserRole) == name:
                    self.step_list.setCurrentRow(row)
                    return
        self.step_list.clearSelection()
        self.step_view.set_step(step)

    def _on_stage_selected(self, stage_name):
        self.step_list.clear()
        for step in all_steps():
            if step.stage == stage_name:
                item = QListWidgetItem(_step_label(step))
                item.setData(Qt.UserRole, step.name)
                item.setData(RUOLO_SOMMARIO, step.summary)
                item.setToolTip(step.summary)
                self.step_list.addItem(item)
        self._refresh_step_badges()
        self.step_view.set_step(None)

    def _refresh_step_badges(self):
        """Re-stamp RUOLO_STATO on every row already in the list, in place.

        Each row's own state (`gui.workspace.step_state`, its own consumes/
        produces, not the stage's aggregate) can change without the list
        being rebuilt -- a job finishing, or a workspace clear -- and
        step_list.clear() is not the fix: it would drop the user's current
        selection along with the rows. Called once from _on_stage_selected
        right after building the rows, and again from _on_any_job_finished
        and run_clear_workspace, so there is exactly one place that knows
        how to compute a row's badge.

        The row's tooltip is re-stamped in the same pass, for the same
        reason the stage pills' is: the badge says "blocked", the tooltip
        says which artifact is missing -- and both come out of the same
        `step_reason` call, so they cannot disagree.
        """
        for row in range(self.step_list.count()):
            item = self.step_list.item(row)
            step = step_by_name(item.data(Qt.UserRole))
            state, missing, satisfied = step_reason(step, self.workspace)
            item.setData(RUOLO_STATO, state)
            item.setToolTip("\n".join(
                [step.summary, testi.step_state_tip(state, missing, satisfied)]))

    def _on_item_changed(self, current, _previous):
        if current is None:
            self.step_view.set_step(None)
            return
        self.step_view.set_step(step_by_name(current.data(Qt.UserRole)))
        job = self._job_for_step(self.step_view.step)
        self.step_view.set_job(job, job is not None and _is_training_step(job.step))
        if job is not None:
            # set_job() above just rebuilt the label from job.running alone
            # ("running"/"finished") -- restore whatever more specific text
            # (stopping/saved at iter N/finished) was last recorded for this
            # job, so navigating away and back does not erase it.
            self._set_job_status(job, self._job_status_text.get(job, self.step_view.job_status.text()))

    def _job_for_step(self, step):
        """The most recent job of `step`, or None.

        Scans `_jobs_in_order` newest-first and returns the first match, not
        the whole history: on 51 `main` steps exactly one -- "3) cut video
        (drop video on me)" -- neither produces nor modifies any artifact,
        so the conflict matrix has nothing to contend and lets a second copy
        of that step start, and this is the one the step view should show.
        """
        for job in reversed(self._jobs_in_order):
            if job.step is step:
                return job
        return None

    def _build_menus(self):
        """Build every menu once. Called only from `__init__`.

        `switch_workspace` used to call this again on every switch, after
        `menubar.clear()` -- which empties the menu *display* but does not
        delete the `QAction` objects, since they are parented to `self`,
        not to the menu: each switch leaked a full set of actions as
        children of the window. Only "Recent" ever needs to change after
        construction, so it alone is rebuilt, by `_refresh_recent_menu`.
        """
        menubar = self.menuBar()

        workspace_menu = menubar.addMenu(testi.MENU_WORKSPACE)
        open_action = QAction(testi.MENU_OPEN_WORKSPACE, self)
        open_action.setToolTip(testi.MENU_OPEN_WORKSPACE_TIP)
        open_action.triggered.connect(lambda: self._open_workspace_dialog())
        workspace_menu.addAction(open_action)
        new_action = QAction(testi.MENU_NEW_WORKSPACE, self)
        new_action.setToolTip(testi.MENU_NEW_WORKSPACE_TIP)
        new_action.triggered.connect(lambda: self._new_workspace_dialog())
        workspace_menu.addAction(new_action)
        self._recent_menu = workspace_menu.addMenu(testi.MENU_RECENT)
        workspace_menu.addSeparator()
        clear_action = QAction(testi.MENU_CLEAR_WORKSPACE, self)
        clear_action.setToolTip(testi.MENU_CLEAR_WORKSPACE_TIP)
        clear_action.triggered.connect(lambda: self.run_clear_workspace())
        workspace_menu.addAction(clear_action)
        self._refresh_recent_menu()

        view_menu = menubar.addMenu(testi.MENU_VIEW)
        self.toggle_console_action = QAction(testi.CONSOLE_DOCK, self)
        self.toggle_console_action.setToolTip(testi.TOGGLE_CONSOLE_TIP)
        self.toggle_console_action.triggered.connect(
            lambda: self.console_dock.setVisible(not self.console_dock.isVisible()))
        view_menu.addAction(self.toggle_console_action)
        self.running_jobs_menu = view_menu.addMenu(testi.MENU_RUNNING_JOBS)
        self._refresh_running_jobs_menu()

        text_size_menu = view_menu.addMenu(testi.MENU_TEXT_SIZE)
        self.text_size_actions = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for nome in SCALE_NAMES:
            action = QAction(testi.TEXT_SIZE_LABELS[nome], self)
            action.setCheckable(True)
            action.setChecked(nome == self._scala.nome())
            action.triggered.connect(lambda _checked=False, n=nome: self.set_text_scale(n))
            group.addAction(action)
            text_size_menu.addAction(action)
            self.text_size_actions[nome] = action

        misc_menu = menubar.addMenu(testi.MENU_MISC)
        for step in all_steps():
            if step.kind == KIND_VIEWER:
                action = QAction(step.name, self)
                action.setToolTip(testi.MISC_STEP_TIP)
                action.triggered.connect(lambda _checked=False, n=step.name: self.select_step(n))
                misc_menu.addAction(action)
        if sys.platform == "win32":
            for step in all_steps():
                if step.kind == KIND_EBSYNTH:
                    action = QAction(step.name, self)
                    action.setToolTip(testi.MISC_STEP_TIP)
                    action.triggered.connect(lambda _checked=False, n=step.name: self.select_step(n))
                    misc_menu.addAction(action)

    def _refresh_recent_menu(self):
        """Rebuild the "Recent" submenu from `RecentWorkspaces`.

        Actions are parented to the submenu itself, not to `self`: `QMenu.
        clear()` deletes actions it owns, so this leaks nothing across
        repeated calls the way rebuilding the whole menu bar did.
        """
        self._recent_menu.clear()
        for path in RecentWorkspaces().paths():
            action = QAction(path, self._recent_menu)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self.switch_workspace(p))
            self._recent_menu.addAction(action)

    def _refresh_running_jobs_menu(self):
        """Rebuild the list of active jobs. Actions belong to the submenu, so
        `clear()` deletes them -- the same ownership rule as the recent
        workspaces submenu.

        An empty submenu is hidden outright rather than left in place: a
        `QMenu` shows its arrow whether or not it has anything behind it,
        and an arrow that opens onto nothing reads as a broken menu. The
        entry comes back by itself the moment a job starts -- every caller
        of this method is already a point where that changed.
        """
        self.running_jobs_menu.clear()
        for job in self.job_manager.active_jobs():
            action = QAction(job.step.name, self.running_jobs_menu)
            action.setToolTip(testi.RUNNING_JOB_TIP)
            action.triggered.connect(
                lambda _checked=False, n=job.step.name: self.select_step(n))
            self.running_jobs_menu.addAction(action)
        self.running_jobs_menu.menuAction().setVisible(
            not self.running_jobs_menu.isEmpty())

    def set_text_scale(self, nome):
        """Change the whole application's text size and remember the choice.

        The style sheet is applied to `QApplication.instance()`, not to this
        window alone: it is one process-wide style sheet, and every open
        window (a training tab, a dialog) reads from it.
        """
        self._scala.imposta(nome)
        apply_dark_theme(QApplication.instance(), self._scala.fattore())
        self.text_size_actions[nome].setChecked(True)

    # -- workspace switching -------------------------------------------------

    def switch_workspace(self, path):
        """Point the whole window at `path`. Refused while any job is active.

        A running job's environment (WORKSPACE, and every path resolved
        against it) is fixed at launch -- switching from under it would
        leave the job writing into a workspace the GUI no longer shows.
        """
        active = self.job_manager.active_jobs()
        if active:
            QMessageBox.warning(
                self, testi.TITLE_JOBS_RUNNING,
                testi.msg_cannot_switch_workspace(len(active), self.workspace))
            return
        path = Path(path)
        self.workspace = path
        self.pipeline_bar.set_workspace(path)
        self.step_view.set_workspace(path)
        RecentWorkspaces().add(path)
        self.step_list.clear()
        self.step_view.set_step(None)
        self._refresh_recent_menu()
        self.pipeline_bar.select(self.pipeline_bar.stage_names()[0])

    def _open_workspace_dialog(self):
        path = QFileDialog.getExistingDirectory(self, testi.DIALOG_OPEN_WORKSPACE, str(self.workspace))
        if path:
            self.switch_workspace(path)

    def _new_workspace_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, testi.DIALOG_NEW_WORKSPACE_LOCATION, str(self.workspace.parent))
        if path:
            create_workspace(path)
            self.switch_workspace(path)

    def run_clear_workspace(self):
        """Confirm (default No), then empty and recreate the workspace's subdirectories.

        Refused while any job is active -- the same guard as
        `switch_workspace`: a job's process writes into the workspace on
        disk, and `clear_workspace` runs `shutil.rmtree` on it, so clearing
        under a running job is a live filesystem race, not a switch of the
        window's own state.
        """
        active = self.job_manager.active_jobs()
        if active:
            QMessageBox.warning(
                self, testi.TITLE_JOBS_RUNNING,
                testi.msg_cannot_clear_workspace(len(active), self.workspace))
            return
        answer = QMessageBox.question(
            self, testi.TITLE_CLEAR_WORKSPACE,
            testi.msg_confirm_clear_workspace(", ".join(STANDARD_SUBDIRS)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        clear_workspace(self.workspace)
        self.pipeline_bar.refresh()
        self._refresh_step_badges()

    def _launch_ebsynth(self):
        """Detached launch of the bundled EBSynth, mirroring the generated .bat step."""
        workdir = self._dfl_root.parent / "EbSynth"
        exe = workdir / "EbSynth.exe"
        sample = workdir / "SampleProject" / "sample.ebs"
        QProcess.startDetached(str(exe), [str(sample)], str(workdir))

    # -- execution -------------------------------------------------------------

    def _on_start_clicked(self):
        step = self.step_view.step
        if step is None:
            return
        if self._shutdown_pending:
            # closeEvent already told every job that was running at
            # confirmation time to stop; a job started after that would
            # never get the same treatment (no "close" for a training run,
            # no place in _on_job_finished_while_closing's wait), so the
            # only correct fix is to refuse the start outright, not to
            # patch the shutdown path to notice it after the fact.
            QMessageBox.warning(self, testi.TITLE_CLOSING, testi.MSG_CLOSING_CANNOT_START)
            return
        if step.kind == KIND_CLEAR:
            self.run_clear_workspace()
            return
        if step.kind == KIND_EBSYNTH:
            self._launch_ebsynth()
            return
        if step.kind != KIND_MAIN:
            return

        form = self.step_view.form
        if step.passthrough and not form.extra_args():
            QMessageBox.warning(self, testi.TITLE_MISSING_INPUT, testi.MSG_MISSING_INPUT)
            return
        if step.needs_model_name and not form.model_name():
            QMessageBox.warning(self, testi.TITLE_MISSING_MODEL_NAME, testi.MSG_MISSING_MODEL_NAME)
            return

        answers = form.answers()
        extra_args = form.extra_args()
        if step.needs_model_name:
            extra_args = extra_args + ("--force-model-name", form.model_name())

        try:
            job = self.job_manager.try_start(step, answers, self.workspace, extra_args=extra_args)
        except StepConflict as exc:
            QMessageBox.warning(self, testi.TITLE_STEP_BUSY, str(exc))
            return
        self._attach_job(step, job, extra_args)

    def console_of(self, job):
        """The open console view of `job`, or None when it is closed."""
        return self._consoles.get(job)

    def open_console(self, job):
        """Build a console view for `job` and fill it from the job's buffer.

        Idempotent: an already open console is raised, not duplicated. The
        text comes from the buffer rather than from the lines seen so far,
        which is what makes a closed-and-reopened tab show the output that
        arrived while it was closed.

        A freshly built tab is raised too, and that has to be said out loud:
        QTabWidget selects a tab on its own only when it is the first one in
        an empty widget, so from the second job on the new console would be
        added behind whatever was already on screen.
        """
        existing = self._consoles.get(job)
        if existing is not None:
            self.console_tabs.setCurrentWidget(existing)
            return existing
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(int(CONSOLE_FONT_POINT_SIZE))
        # Belt and braces: on some platform themes (notably the offscreen QPA
        # backend the test suite runs under) FixedFont resolves to a family
        # that is not actually fixed-pitch. The console is process output --
        # alignment is the point -- so force the flag rather than trust it.
        font.setFixedPitch(True)
        text_edit.setFont(font)
        text_edit.setPlainText("\n".join(job.captured_lines))
        self._consoles[job] = text_edit
        self.console_tabs.addTab(text_edit, self._console_title(job))
        self.console_tabs.setCurrentWidget(text_edit)
        self.console_dock.show()
        return text_edit

    def close_console(self, job):
        """Drop the view. The job, its process and its buffer are untouched."""
        text_edit = self._consoles.get(job)
        if text_edit is None:
            return
        self._consoles[job] = None
        index = self.console_tabs.indexOf(text_edit)
        if index != -1:
            self.console_tabs.removeTab(index)
        text_edit.deleteLater()

    def _on_tab_close_requested(self, index):
        widget = self.console_tabs.widget(index)
        for job, console in self._consoles.items():
            if console is widget:
                self.close_console(job)
                return

    def _console_title(self, job):
        return job.step.name

    # -- the training tabs ---------------------------------------------------

    def panel_of(self, job):
        """The training tab of `job`, open or closed, or None if it has none."""
        return self._panels.get(job)

    def open_panel(self, job):
        """Show `job`'s training tab, building it the first time.

        The panel lives in the dictionary, not in the QTabWidget: closing its
        tab drops a view, not the state -- the same rule the console follows,
        and for the same reason (a training run prints, plots and previews
        for hours).

        Raising the tab is explicit on purpose, on a fresh one as much as on
        a reopened one: QTabWidget selects a tab by itself only when it is
        the first in an empty widget, and here the first is always "Steps".
        """
        panel = self._panels.get(job)
        if panel is None:
            panel = TrainingPanel(job.previews_path, parent=self)
            panel.comando.connect(lambda op, job=job: self._command_to_job(job, op))
            self._panels[job] = panel
        index = self.central_tabs.indexOf(panel)
        if index == -1:
            index = self.central_tabs.addTab(panel, self._panel_title(job))
        self.central_tabs.setCurrentIndex(index)
        return panel

    def close_panel(self, job):
        """Drop the view. The panel keeps every point, image and message."""
        panel = self._panels.get(job)
        if panel is None:
            return
        index = self.central_tabs.indexOf(panel)
        if index != -1:
            self.central_tabs.removeTab(index)
            # removeTab hands ownership back with no parent: reparenting to
            # the window keeps the object alive and hidden until it is added
            # again, instead of leaving its lifetime to the garbage collector.
            panel.setParent(self)

    def _panel_title(self, job):
        """The tab title, with a marker only while the run is alive.

        A marker meaning "running" that never goes away lies, in the same way
        the strip's greyed-out buttons did once the process behind them was
        gone.
        """
        return testi.running_tab_title(job.step.name) if job.running else job.step.name

    def _command_to_job(self, job, op):
        """One tab's button reaching one job's command channel.

        The panel emits its op without saying who it belongs to -- the
        binding is here, per job, which is what keeps two open tabs from
        both talking to the last one started.
        """
        if op == "close":
            # Same bookkeeping as the strip's Stop: without it the "save" and
            # "end" events of the shutdown are read as an ordinary periodic
            # save, and the strip goes on saying "running" while the trainer
            # is winding down.
            self._stop_requested.add(job)
            self._set_job_status(job, testi.STATUS_STOPPING)
        job.send_command(op)

    def _on_central_tab_close_requested(self, index):
        if index == 0:
            return      # "Steps" has no close button; a request by index is refused too
        widget = self.central_tabs.widget(index)
        for job, panel in self._panels.items():
            if panel is widget:
                self.close_panel(job)
                return

    def _attach_job(self, step, job, extra_args):
        self._jobs_in_order.append(job)
        self._job_status_text[job] = testi.JOB_RUNNING
        self.open_console(job)
        if self.step_view.step is step:
            self.step_view.set_job(job, _is_training_step(step))
        self._refresh_running_jobs_menu()

        def _on_output(line):
            console = self._consoles.get(job)
            if console is not None:
                console.append(line)
        job.output.connect(_on_output)

        def _on_output_update(line):
            """Rewrite the console's last line, the way a carriage return
            rewrites a terminal row: select from the end of the document
            back to the start of its block and replace the text. Selecting
            a *block* rather than a visual line is what keeps this right for
            a status line long enough to wrap."""
            console = self._consoles.get(job)
            if console is None:
                return
            cursor = console.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
        job.output_update.connect(_on_output_update)

        tail = None
        if step.process == PROCESS_SESSION and _is_training_step(step):
            self.open_panel(job)
            tail = EventTail(job.events_path, parent=self)
            state = {"target_iter": 0, "ritmo": RitmoIterazioni()}
            tail.event.connect(
                lambda event, step=step, job=job, state=state:
                self._on_telemetry_event(step, job, state, event))
            self._tails.append(tail)

        job.finished.connect(
            lambda code, step=step, job=job, tail=tail, extra_args=extra_args:
            self._on_job_finished(step, job, code, tail, extra_args))

    def _set_job_status(self, job, text):
        """Record `text` as `job`'s current strip status and show it if `job`
        is the one on screen right now.

        The single place that writes the strip's job_status label outside
        of set_job()'s own "running"/"finished" default, so the sequence a
        clean stop walks through -- stopping, saved at iter N, finished --
        survives being stored, not just displayed once: set_job() rebuilds
        the label from job.running alone on every navigation, and without
        this dict a stop requested while looking at a different step would
        read as a plain "running" the moment the user came back to it.
        """
        self._job_status_text[job] = text
        if self.step_view.job is job:
            self.step_view.job_status.setText(text)

    def _on_stop_clicked(self):
        """Stop the strip's job: politely for a training, hard for anything else.

        A training reads the command channel and answers a "close" the way
        it answers Enter in the preview window -- it saves, then exits, and
        that can take longer than a minute on a real model, so nothing here
        sets a deadline. Every other step has no such channel: for those,
        Stop is the process-tree kill it has always been.
        """
        job = self.step_view.job
        if job is None or not job.running:
            return
        if self.step_view.is_training:
            job.send_command("close")
            self._stop_requested.add(job)
            self._set_job_status(job, testi.STATUS_STOPPING)
            return
        self.job_manager.stop(job)

    def _on_force_stop_clicked(self):
        job = self.step_view.job
        if job is not None:
            self.job_manager.stop(job)

    def _send_to_current_job(self, op):
        job = self.step_view.job
        if job is not None and job.running:
            job.send_command(op)

    def _on_console_button_clicked(self):
        job = self.step_view.job
        if job is None:
            return
        if self.console_of(job) is None:
            self.open_console(job)
        else:
            self.close_console(job)

    def _on_job_finished(self, step, job, code, tail, extra_args):
        if tail is not None:
            tail.stop()
            if tail in self._tails:
                self._tails.remove(tail)
        self._stop_requested.discard(job)
        panel = self._panels.get(job)
        if panel is not None:
            panel.job_finito(code)
            index = self.central_tabs.indexOf(panel)
            if index != -1:
                self.central_tabs.setTabText(index, self._panel_title(job))
        if self.step_view.job is job:
            self.step_view.set_job(job, self.step_view.is_training)
        self._set_job_status(job, testi.job_finished_status(code))
        self._refresh_running_jobs_menu()
        if code != 0:
            console = self._consoles.get(job)
            if console is not None:
                index = self.console_tabs.indexOf(console)
                self.console_tabs.setTabText(index, testi.failed_console_title(step.name))
                self.console_tabs.tabBar().setTabTextColor(index, QColor("#ff6b6b"))
            if self.step_view.step is step:
                self.step_view.status_panel.setText("\n".join(job.captured_lines[-15:]))
            if code == _FAILED_TO_START:
                message = _reproducible_command(
                    step, job, self.workspace, self._dfl_root, self._python_exe, extra_args)
                QMessageBox.warning(self, testi.TITLE_FAILED_TO_START, message)

    def _on_telemetry_event(self, step, job, state, event):
        """Where the events channel lands, and where nothing may escape.

        This is a Qt slot: an exception raised here reaches no caller.
        PyQt5 turns it into qFatal and the process dies with "Aborted (core
        dumped)", taking every other live training with it. The event is
        JSON written by another process, so a key with an unexpected type
        -- `"iter": "cinque"`, a null loss -- is a real possibility rather
        than a hypothetical, and it reaches *both* readers below: the
        training tab, and the strip's own formatting six lines further
        down. A net around one of the two is decoration; this one is around
        everything the slot does.
        """
        try:
            self._apply_telemetry_event(step, job, state, event)
        except Exception as error:
            panel = self._panels.get(job)
            if panel is not None:
                panel.evento_non_applicato(error)

    def _apply_telemetry_event(self, step, job, state, event):
        panel = self._panels.get(job)
        if panel is not None:
            try:
                panel.applica_evento(event)
            except Exception as error:
                # The tab and the strip read the same event and neither
                # needs the other: a tab that cannot apply it -- a preview
                # file that will not decode, a malformed VRAM reading only
                # the tab looks at -- must not cost the strip the update it
                # could perfectly well make. Hence a second, narrower net
                # inside the one above, whose job is only "never die".
                panel.evento_non_applicato(error)
        event_type = event.get("type")
        if event_type == "hello":
            # Validato prima di essere ricordato, come l'iterazione: il
            # confronto con `target_iter` avviene a ogni evento successivo,
            # quindi un obiettivo storto non rompe l'evento che lo porta --
            # rompe tutti quelli buoni che vengono dopo.
            state["target_iter"] = obiettivo_valido(event.get("target_iter", 0))
            return
        if event_type == "save":
            # Only meaningful mid-stop: EventLog.save() also fires on every
            # ordinary periodic save, and overwriting "running" with a
            # message implying the trainer is about to close would be wrong
            # outside that window.
            if job in self._stop_requested:
                # The iteration comes off the channel like every other
                # number here, and this is the one place that formats one
                # with %d without going through pezzi_di_stato. An
                # unreadable one drops the number rather than inventing a
                # zero: the sentence still says the thing that matters,
                # which is that the trainer saved and has not closed yet.
                saved_at = event.get("iter", 0)
                self._set_job_status(job, testi.status_saved_waiting_to_close(
                    saved_at if iterazione_utilizzabile(saved_at) else None))
            return
        if event_type == "end":
            if job in self._stop_requested:
                self._set_job_status(job, testi.STATUS_CLOSING)
            return
        if event_type != "iter":
            return
        iteration = event.get("iter", 0)
        # The tracker validates before it remembers, which is what keeps a
        # single malformed event from poisoning every good one after it --
        # see `gui.status_line.RitmoIterazioni`.
        rate = state["ritmo"].aggiorna(iteration, time.monotonic())

        # The strip says the same thing the training tab says under its plot,
        # so the wording lives in one place (`gui.status_line`). VRAM is the
        # tab's alone: it has the room for it, a one-line strip does not.
        parts = pezzi_di_stato(iteration, event.get("losses") or [], rate,
                               state.get("target_iter", 0))

        if self.step_view.step is step:
            self.step_view.status_panel.setText(" | ".join(parts))

    # -- closing -----------------------------------------------------------

    def closeEvent(self, event):
        """Never leave a child behind.

        The window used to ask its jobs to stop and then let the close go
        through; `app.exec_()` returns as soon as the last window closes, so
        the escalation scheduled on a timer could die before firing, leaving
        orphans. Now the close is refused until every job is gone: training
        jobs are asked over the command channel (they save first, which on a
        real model takes longer than a minute -- so no deadline), the others
        are killed, and "Force stop" is always one click away.
        """
        if not self.job_manager.active_jobs():
            for tail in list(self._tails):
                tail.stop()
            self._tails.clear()
            super().closeEvent(event)
            return

        event.ignore()
        if self._shutdown_pending:
            return
        answer = QMessageBox.question(
            self, testi.TITLE_JOBS_RUNNING,
            testi.MSG_CONFIRM_CLOSE_WITH_ACTIVE_JOBS,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self._shutdown_pending = True
        for job in self.job_manager.active_jobs():
            if _is_training_step(job.step):
                job.send_command("close")
            else:
                self.job_manager.stop(job)
        self.job_manager.job_finished.connect(self._on_job_finished_while_closing)

        # Non-modal: the strip's own "Force stop" only reaches the job of
        # the step currently on screen, which need not be the one holding
        # the close up. This one is always available, whatever step is
        # selected, without blocking the window like QMessageBox.question.
        self._shutdown_dialog = QMessageBox(self)
        self._shutdown_dialog.setWindowTitle(testi.TITLE_CLOSING)
        self._shutdown_dialog.setText(testi.MSG_CLOSING_WAIT)
        self._shutdown_dialog.setStandardButtons(QMessageBox.NoButton)
        force_button = self._shutdown_dialog.addButton(testi.FORCE_STOP, QMessageBox.ActionRole)
        force_button.clicked.connect(self.force_shutdown)
        self._shutdown_dialog.setModal(False)
        self._shutdown_dialog.show()

    def _on_job_finished_while_closing(self, _job, _code):
        if self._shutdown_pending and not self.job_manager.active_jobs():
            self._close_shutdown_dialog()
            self.close()

    def force_shutdown(self):
        """Kill every job's process tree, then close. Wired to the dialog's button."""
        self._shutdown_pending = True
        self.job_manager.stop_all()
        if not self.job_manager.active_jobs():
            self._close_shutdown_dialog()
            self.close()

    def _close_shutdown_dialog(self):
        if self._shutdown_dialog is not None:
            self._shutdown_dialog.close()
            self._shutdown_dialog = None
