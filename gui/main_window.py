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
from PyQt5.QtGui import QColor, QDesktopServices, QFontDatabase
from PyQt5.QtWidgets import (
    QAction, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from gui.catalog import all_steps, step_by_name
from gui.catalog.model import (
    KIND_CLEAR, KIND_EBSYNTH, KIND_MAIN, KIND_VIEWER, PROCESS_SESSION, STAGES,
)
from gui.execution.jobs import JobManager, StepConflict, _model_class_from_step, _resolve
from gui.forms import StepForm
from gui.saved_options import saved_options
from gui.telemetry import EventTail
from gui.theme import CONSOLE_FONT_POINT_SIZE
from gui.workspace import (
    STANDARD_SUBDIRS, STATE_BLOCKED, STATE_DONE, STATE_READY,
    RecentWorkspaces, clear_workspace, create_workspace, default_workspace,
    saved_model_names, stage_states,
)

# Console tab titles for a failed job: exit code -1 is Job's sentinel for
# "the process never started at all" (see gui.execution.jobs.Job).
_FAILED_TO_START = -1

_STAGE_COLOR = {
    STATE_DONE: "#2e7d32",
    STATE_READY: "#1565c0",
    STATE_BLOCKED: "#9e9e9e",
}


def _button_style(state):
    return "QPushButton { background-color: %s; color: white; }" % _STAGE_COLOR[state]


def _step_label(step):
    """The step list's display text: name, plus the two layout badges."""
    label = step.name
    if step.optional:
        label += "  (optional)"
    if step.process == PROCESS_SESSION:
        label += "  [opens an external window]"
    return label


def _placeholder_text(step, workspace, dfl_root):
    """Description shown in place of a form for a non-main-kind step.

    `step.target` (KIND_VIEWER only) carries the same {WORKSPACE}/{DFL_ROOT}/
    {INTERNAL} placeholders as an invocation's args, resolved the same way
    (`gui.execution.jobs._resolve`) so the shown path is the real one, not
    the catalog template.
    """
    if step.kind == KIND_VIEWER:
        return "Opens %s in the system file manager." % _resolve(step.target, workspace, dfl_root)
    if step.kind == KIND_CLEAR:
        return "Empties and recreates the workspace's standard subdirectories."
    if step.kind == KIND_EBSYNTH:
        return "Launches the bundled EBSynth application."
    return ""


def _is_training_step(step):
    """True if any of `step`'s invocations runs `main.py train`.

    Gates telemetry attachment. Keyed on the invocation's verb rather than
    `step.family`: "5.XSeg) train" belongs to the "xseg" family but still
    invokes `main.py train` and writes the same events channel as every
    other training step, so a family-based gate misses it.
    """
    return any(invocation.verb == ("train",) for invocation in step.invocations)


def _format_eta(seconds):
    """Format a countdown in seconds as H:MM:SS, or M:SS under an hour."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%d:%02d" % (minutes, secs)


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
    return "\n".join([
        command,
        "",
        "WORKSPACE=%s" % workspace,
        "DFL_ANSWERS_FILE=%s" % (job.workdir / "answers.json"),
        "DFL_EVENTS_FILE=%s" % job.events_path,
        "DFL_COMMANDS_FILE=%s" % job.commands_path,
    ])


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
        """Recompute every stage's color from the workspace's artifacts on disk."""
        self._states = stage_states(self._workspace)
        for stage, button in self._buttons.items():
            button.setStyleSheet(_button_style(self._states.get(stage, STATE_BLOCKED)))


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
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        layout.addWidget(self._body, 1)
        self.start_button = QPushButton("Start")
        layout.addWidget(self.start_button)
        self.job_strip = QWidget()
        strip = QHBoxLayout(self.job_strip)
        strip.setContentsMargins(0, 0, 0, 0)
        self.job_status = QLabel("")
        strip.addWidget(self.job_status, 1)
        self.stop_button = QPushButton("Stop")
        self.force_stop_button = QPushButton("Force stop")
        self.save_button = QPushButton("Save")
        self.backup_button = QPushButton("Backup")
        self.console_button = QPushButton("Show console")
        for b in (self.stop_button, self.force_stop_button, self.save_button,
                  self.backup_button, self.console_button):
            strip.addWidget(b)
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
        action buttons are also *enabled* only while the job is still
        running -- a finished job has nothing left for any of them to do,
        and leaving them clickable would silently no-op rather than show
        that the job is over. Console stays enabled either way: its history
        is worth reopening long after the job itself has ended.
        """
        self.job = job
        self.is_training = is_training
        self.job_strip.setVisible(job is not None)
        if job is None:
            return
        for b in (self.save_button, self.backup_button, self.force_stop_button):
            b.setVisible(is_training)
        for b in (self.stop_button, self.force_stop_button, self.save_button, self.backup_button):
            b.setEnabled(job.running)
        self.job_status.setText("running" if job.running else "finished")

    def set_step(self, step):
        """Rebuild the panel for `step` (a StepDef), or clear it for None."""
        self.step = step
        self.set_job(None, False)
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.open_button = None
        if step is None:
            self.form = None
            self.placeholder = None
            self.status_panel.setText("")
            return
        self.start_button.setVisible(step.kind != KIND_VIEWER)
        if step.kind == KIND_MAIN:
            self.form = StepForm(step)
            self.placeholder = None
            if step.needs_model_name:
                self.form.set_model_names(saved_model_names(self._workspace / "model"))
                self.form._model_combo.currentTextChanged.connect(self._refresh_saved_values)
            self._body_layout.addWidget(self.form)
            self._refresh_saved_values()
        else:
            self.form = None
            self.placeholder = QLabel(_placeholder_text(step, self._workspace, self._dfl_root))
            self.placeholder.setWordWrap(True)
            self._body_layout.addWidget(self.placeholder)
            if step.kind == KIND_VIEWER:
                self.open_button = QPushButton("Open folder")
                target = _resolve(step.target, self._workspace, self._dfl_root)
                self.open_button.clicked.connect(
                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(target)))
                self._body_layout.addWidget(self.open_button)
        badges = []
        if step.optional:
            badges.append("optional")
        if step.process == PROCESS_SESSION:
            badges.append("opens an external window")
        self.status_panel.setText(", ".join(badges))

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

    def __init__(self, python_exe, dfl_root, workspace=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DeepFaceLab")
        self._python_exe = python_exe
        self._dfl_root = Path(dfl_root)
        self.workspace = Path(workspace) if workspace is not None else default_workspace(self._dfl_root)
        self.job_manager = JobManager(python_exe, self._dfl_root)
        self.job_manager.job_started.connect(lambda _job: self._refresh_running_jobs_menu())
        self.job_manager.job_finished.connect(self._on_any_job_finished)

        self.pipeline_bar = PipelineBar(self.workspace)
        self.step_list = QListWidget()
        self.step_view = StepView(self.workspace, self._dfl_root)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(self.pipeline_bar)
        body = QSplitter(Qt.Horizontal)
        body.addWidget(self.step_list)
        body.addWidget(self.step_view)
        central_layout.addWidget(body, 1)
        self.setCentralWidget(central)

        self.console_dock = QDockWidget("Console", self)
        self.console_tabs = QTabWidget()
        self.console_tabs.setTabsClosable(True)
        self.console_tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.console_dock.setWidget(self.console_tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.console_dock.hide()

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
                self.step_list.addItem(item)
        self.step_view.set_step(None)

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

        workspace_menu = menubar.addMenu("&Workspace")
        open_action = QAction("Open…", self)
        open_action.triggered.connect(lambda: self._open_workspace_dialog())
        workspace_menu.addAction(open_action)
        new_action = QAction("New…", self)
        new_action.triggered.connect(lambda: self._new_workspace_dialog())
        workspace_menu.addAction(new_action)
        self._recent_menu = workspace_menu.addMenu("Recent")
        workspace_menu.addSeparator()
        clear_action = QAction("Clear workspace…", self)
        clear_action.triggered.connect(lambda: self.run_clear_workspace())
        workspace_menu.addAction(clear_action)
        self._refresh_recent_menu()

        view_menu = menubar.addMenu("&View")
        self.toggle_console_action = QAction("Console", self)
        self.toggle_console_action.triggered.connect(
            lambda: self.console_dock.setVisible(not self.console_dock.isVisible()))
        view_menu.addAction(self.toggle_console_action)
        self.running_jobs_menu = view_menu.addMenu("Running jobs")
        self._refresh_running_jobs_menu()

        misc_menu = menubar.addMenu("&Misc")
        for step in all_steps():
            if step.kind == KIND_VIEWER:
                action = QAction(step.name, self)
                action.triggered.connect(lambda _checked=False, n=step.name: self.select_step(n))
                misc_menu.addAction(action)
        if sys.platform == "win32":
            for step in all_steps():
                if step.kind == KIND_EBSYNTH:
                    action = QAction(step.name, self)
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
            action.triggered.connect(lambda _checked=False, p=path: self.switch_workspace(p))
            self._recent_menu.addAction(action)

    def _refresh_running_jobs_menu(self):
        """Rebuild the list of active jobs. Actions belong to the submenu, so
        `clear()` deletes them -- the same ownership rule as the recent
        workspaces submenu."""
        self.running_jobs_menu.clear()
        for job in self.job_manager.active_jobs():
            action = QAction(job.step.name, self.running_jobs_menu)
            action.triggered.connect(
                lambda _checked=False, n=job.step.name: self.select_step(n))
            self.running_jobs_menu.addAction(action)

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
                self, "Jobs running",
                "Cannot switch workspace: %d job(s) are still using %s."
                % (len(active), self.workspace))
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
        path = QFileDialog.getExistingDirectory(self, "Open workspace", str(self.workspace))
        if path:
            self.switch_workspace(path)

    def _new_workspace_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "New workspace location", str(self.workspace.parent))
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
                self, "Jobs running",
                "Cannot clear workspace: %d job(s) are still using %s."
                % (len(active), self.workspace))
            return
        answer = QMessageBox.question(
            self, "Clear workspace",
            "This empties and recreates: %s. This cannot be undone. Continue?"
            % ", ".join(STANDARD_SUBDIRS),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        clear_workspace(self.workspace)
        self.pipeline_bar.refresh()

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
            QMessageBox.warning(self, "Closing", "The window is closing: cannot start a new job.")
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
            QMessageBox.warning(self, "Missing input", "Select an input file before starting.")
            return
        if step.needs_model_name and not form.model_name():
            QMessageBox.warning(self, "Missing model name", "Enter or choose a model name before starting.")
            return

        answers = form.answers()
        extra_args = form.extra_args()
        if step.needs_model_name:
            extra_args = extra_args + ("--force-model-name", form.model_name())

        try:
            job = self.job_manager.try_start(step, answers, self.workspace, extra_args=extra_args)
        except StepConflict as exc:
            QMessageBox.warning(self, "Step busy", str(exc))
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
        """
        existing = self._consoles.get(job)
        if existing is not None:
            self.console_tabs.setCurrentWidget(existing)
            return existing
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(CONSOLE_FONT_POINT_SIZE)
        # Belt and braces: on some platform themes (notably the offscreen QPA
        # backend the test suite runs under) FixedFont resolves to a family
        # that is not actually fixed-pitch. The console is process output --
        # alignment is the point -- so force the flag rather than trust it.
        font.setFixedPitch(True)
        text_edit.setFont(font)
        text_edit.setPlainText("\n".join(job.captured_lines))
        self._consoles[job] = text_edit
        self.console_tabs.addTab(text_edit, self._console_title(job))
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

    def _attach_job(self, step, job, extra_args):
        self._jobs_in_order.append(job)
        self._job_status_text[job] = "running"
        self.open_console(job)
        if self.step_view.step is step:
            self.step_view.set_job(job, _is_training_step(step))
        self._refresh_running_jobs_menu()

        def _on_output(line):
            console = self._consoles.get(job)
            if console is not None:
                console.append(line)
        job.output.connect(_on_output)

        tail = None
        if step.process == PROCESS_SESSION and _is_training_step(step):
            tail = EventTail(job.events_path, parent=self)
            state = {"iter": None, "time": None, "target_iter": 0}
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
            self._set_job_status(job, "stopping — waiting for the trainer to save")
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
        if self.step_view.job is job:
            self.step_view.set_job(job, self.step_view.is_training)
        self._set_job_status(job, "finished (exit %d)" % code)
        self._refresh_running_jobs_menu()
        if code != 0:
            console = self._consoles.get(job)
            if console is not None:
                index = self.console_tabs.indexOf(console)
                self.console_tabs.setTabText(index, "%s ✗" % step.name)
                self.console_tabs.tabBar().setTabTextColor(index, QColor("#ff6b6b"))
            if self.step_view.step is step:
                self.step_view.status_panel.setText("\n".join(job.captured_lines[-15:]))
            if code == _FAILED_TO_START:
                message = _reproducible_command(
                    step, job, self.workspace, self._dfl_root, self._python_exe, extra_args)
                QMessageBox.warning(self, "Failed to start", message)

    def _on_telemetry_event(self, step, job, state, event):
        event_type = event.get("type")
        if event_type == "hello":
            state["target_iter"] = event.get("target_iter", 0)
            return
        if event_type == "save":
            # Only meaningful mid-stop: EventLog.save() also fires on every
            # ordinary periodic save, and overwriting "running" with a
            # message implying the trainer is about to close would be wrong
            # outside that window.
            if job in self._stop_requested:
                self._set_job_status(job, "saved at iter %d — waiting for the trainer to close" % event.get("iter", 0))
            return
        if event_type == "end":
            if job in self._stop_requested:
                self._set_job_status(job, "closing…")
            return
        if event_type != "iter":
            return
        now = time.monotonic()
        iteration = event.get("iter", 0)
        rate = None
        if state["iter"] is not None and now > state["time"]:
            elapsed = now - state["time"]
            if elapsed > 0:
                rate = (iteration - state["iter"]) / elapsed
        state["iter"], state["time"] = iteration, now

        parts = ["iter %d" % iteration]
        losses = event.get("losses") or []
        if losses:
            parts.append("loss " + ", ".join("%.4f" % v for v in losses))
        if rate is not None:
            parts.append("%.2f it/s" % rate)
            target_iter = state.get("target_iter", 0)
            if target_iter and target_iter > iteration and rate > 0:
                parts.append("ETA %s" % _format_eta((target_iter - iteration) / rate))

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
            self, "Jobs running",
            "Active jobs are still running. Stop them and close?",
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
        self._shutdown_dialog.setWindowTitle("Closing")
        self._shutdown_dialog.setText("Waiting for the active jobs to stop before closing.")
        self._shutdown_dialog.setStandardButtons(QMessageBox.NoButton)
        force_button = self._shutdown_dialog.addButton("Force stop", QMessageBox.ActionRole)
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
