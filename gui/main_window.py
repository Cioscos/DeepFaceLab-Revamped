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
from PyQt5.QtGui import QColor, QDesktopServices
from PyQt5.QtWidgets import (
    QAction, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from gui.catalog import all_steps, step_by_name
from gui.catalog.model import (
    KIND_CLEAR, KIND_EBSYNTH, KIND_MAIN, KIND_VIEWER, PROCESS_SESSION, STAGES,
)
from gui.execution.jobs import JobManager, StepConflict, _resolve
from gui.forms import StepForm
from gui.telemetry import EventTail
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
    STATE_BLOCKED: "#757575",
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
        self.status_panel = QLabel("")
        self.status_panel.setWordWrap(True)
        layout.addWidget(self.status_panel)

    def set_workspace(self, workspace):
        """Point at a different workspace; takes effect on the next `set_step`."""
        self._workspace = workspace

    def set_step(self, step):
        """Rebuild the panel for `step` (a StepDef), or clear it for None."""
        self.step = step
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
            self._body_layout.addWidget(self.form)
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
        self.job_manager.job_finished.connect(lambda _job, _code: self.pipeline_bar.refresh())

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
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self.console_tabs.setCornerWidget(self._stop_button)
        self.console_tabs.currentChanged.connect(self._on_console_tab_changed)
        self.console_dock.setWidget(self.console_tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        self.console_dock.hide()

        self._jobs_by_tab = {}   # console tab index -> Job
        self._tails = []         # active EventTail instances

        self.pipeline_bar.stage_selected.connect(self._on_stage_selected)
        self.step_list.currentItemChanged.connect(self._on_item_changed)
        self.step_view.start_button.clicked.connect(self._on_start_clicked)

        self._build_menus()
        self.pipeline_bar.select(self.pipeline_bar.stage_names()[0])

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

    def _attach_job(self, step, job, extra_args):
        # Register the job BEFORE addTab, not after: for the very first tab,
        # addTab's insertion makes it the current one and emits
        # currentChanged(0) synchronously, before addTab even returns. If
        # `_jobs_by_tab[0]` isn't set yet at that point, `_on_console_tab_
        # changed` finds nothing for index 0 and leaves Stop disabled with
        # no way to enable it later -- the session's first job could never
        # be stopped from the console.
        index = self.console_tabs.count()
        self._jobs_by_tab[index] = job
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        self.console_tabs.addTab(text_edit, step.name)

        def _on_output(line):
            text_edit.append(line)
            self.console_dock.show()
        job.output.connect(_on_output)

        tail = None
        if step.process == PROCESS_SESSION and _is_training_step(step):
            tail = EventTail(job.events_path, parent=self)
            state = {"iter": None, "time": None, "target_iter": 0}
            tail.event.connect(
                lambda event, step=step, state=state: self._on_telemetry_event(step, state, event))
            self._tails.append(tail)

        job.finished.connect(
            lambda code, step=step, job=job, index=index, tail=tail, extra_args=extra_args:
            self._on_job_finished(step, job, index, code, tail, extra_args))

    def _on_console_tab_changed(self, index):
        job = self._jobs_by_tab.get(index)
        self._stop_button.setEnabled(job is not None and job.running)

    def _on_stop_clicked(self):
        job = self._jobs_by_tab.get(self.console_tabs.currentIndex())
        if job is not None:
            self.job_manager.stop(job)

    def _on_job_finished(self, step, job, index, code, tail, extra_args):
        if tail is not None:
            tail.stop()
            if tail in self._tails:
                self._tails.remove(tail)
        if code != 0:
            self.console_tabs.setTabText(index, "%s ✗" % step.name)
            self.console_tabs.tabBar().setTabTextColor(index, QColor("red"))
            if self.step_view.step is step:
                self.step_view.status_panel.setText("\n".join(job.captured_lines[-15:]))
            if code == _FAILED_TO_START:
                message = _reproducible_command(
                    step, job, self.workspace, self._dfl_root, self._python_exe, extra_args)
                QMessageBox.warning(self, "Failed to start", message)
        if self.console_tabs.currentIndex() == index:
            self._stop_button.setEnabled(False)

    def _on_telemetry_event(self, step, state, event):
        event_type = event.get("type")
        if event_type == "hello":
            state["target_iter"] = event.get("target_iter", 0)
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
        if self.job_manager.active_jobs():
            answer = QMessageBox.question(
                self, "Jobs running",
                "Active jobs are still running. Stop them and close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.job_manager.stop_all()
        for tail in list(self._tails):
            tail.stop()
        self._tails.clear()
        super().closeEvent(event)
