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
    QAction, QActionGroup, QApplication, QCheckBox, QDockWidget, QFileDialog, QFrame,
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTabBar, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)

from gui import testi
from gui.catalog import all_steps, step_by_name
from gui.catalog.model import (
    FAMILY_STAGE, KIND_CLEAR, KIND_EBSYNTH, KIND_MAIN, KIND_VIEWER, PROCESS_SESSION,
    STAGES,
)
from gui.delegato_passi import RUOLO_SOMMARIO, RUOLO_STATO, DelegatoPassi
from gui.duplicazione import DialogoDuplicazione
from gui.execution.jobs import JobManager, StepConflict, _model_class_from_step, _resolve
from gui.forms import StepForm
from gui.preferenze import ScalaTesto
from gui.progetti import (
    ArchivioProgetti, dimensione, identita_workspace, leggi_progetto, radice_progetti,
    ricorda_risposte, risposte_ricordate, serve_migrazione, slug, stesso_workspace,
)
from gui.rimozione import svuota
from gui.saved_options import saved_options
from gui.selettore_progetti import SelettoreProgetti
from gui.status_line import (
    RitmoIterazioni, iterazione_utilizzabile, obiettivo_valido, pezzi_di_stato)
from gui.telemetry import EventTail
from gui.theme import CONSOLE_FONT_POINT_SIZE, SCALE_NAMES, STATO_COLORE, apply_dark_theme
from gui.training_panel import TrainingPanel
from gui.workspace import (
    STANDARD_SUBDIRS, STATE_BLOCKED, STATE_DONE, STATE_READY,
    RecentWorkspaces, clear_workspace, default_workspace,
    saved_model_names, stage_reasons, step_reason,
)

# Console tab titles for a failed job: exit code -1 is Job's sentinel for
# "the process never started at all" (see gui.execution.jobs.Job).
_FAILED_TO_START = -1

# Derivato da FAMILY_STAGE, non riscritto come letterale: un refuso qui
# ripristinerebbe in silenzio l'elenco dei diciassette passi al posto della
# pagina, e nessun test se ne accorgerebbe finche' non guarda una lista vuota.
STAGE_CURA_FACESET = FAMILY_STAGE["cura-faceset"]

# Stessa cautela di STAGE_CURA_FACESET, stesso motivo: un refuso qui
# ripristinerebbe in silenzio l'elenco dei sei passi di estrazione al posto
# della pagina.
STAGE_ESTRAZIONE = FAMILY_STAGE["estrazione"]

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
            # The project's memory goes in first, the model's own saved
            # options after (_refresh_saved_values, below): when both have
            # something to say about a field, the truth is what the model
            # has on disk, not what a previous run of this same step typed.
            ricordate = risposte_ricordate(self._workspace, step.name)
            if ricordate:
                self.form.set_remembered_values(ricordate)
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
        self.archivio = ArchivioProgetti(radice_progetti(self._dfl_root))
        self.workspace = Path(workspace) if workspace is not None else self._workspace_iniziale()
        if settings is None:
            from PyQt5.QtCore import QSettings
            settings = QSettings("DeepFaceLab", "gui")
        self._settings = settings
        self._scala = ScalaTesto(self._settings)
        self.job_manager = JobManager(python_exe, self._dfl_root)
        self.selettore = SelettoreProgetti(self.archivio)
        self.selettore.progetto_scelto.connect(
            lambda progetto: self.switch_workspace(progetto.cartella))
        self.job_manager.job_started.connect(self._on_job_started)
        self.job_manager.job_finished.connect(self._on_any_job_finished)

        self.pipeline_bar = PipelineBar(self.workspace)
        self.step_list = QListWidget()
        self.step_list.setItemDelegate(DelegatoPassi(self.step_list))
        self.step_view = StepView(self.workspace, self._dfl_root)

        passi = QWidget()
        passi_layout = QVBoxLayout(passi)
        passi_layout.addWidget(self.selettore)
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

        self._pagina_faceset = None  # PaginaCuraFaceset, costruita al primo apri_pagina_faceset
        self._pagina_estrazione = None  # PaginaEstrazione, costruita al primo apri_pagina_estrazione
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
        self._aggiorna_selettore()

    def _workspace_iniziale(self):
        """Il workspace da mostrare a un avvio senza `workspace=` esplicito.

        Rispecchia il primo dei tre livelli che setenv.bat/.sh risolvono: il
        progetto che il puntatore nomina, se ne nomina uno valido,
        altrimenti la radice -- la stessa ricaduta di sempre per
        un'installazione non ancora migrata. Senza questo la GUI ricadeva
        sempre sulla radice, mentre uno script lanciato a mano nello stesso
        momento segue il puntatore: i due si contraddicevano a ogni
        riavvio.
        """
        progetto = self.archivio.attivo()
        return progetto.cartella if progetto is not None else default_workspace(self._dfl_root)

    def _on_job_started(self, _job):
        # Slot diretto di job_manager.job_started: protetto con
        # _esegui_protetta come le altre azioni che toccano l'archivio dei
        # progetti (qui, _aggiorna_selettore scandisce il disco).
        self._esegui_protetta(self._on_job_started_ora)

    def _on_job_started_ora(self):
        self._refresh_running_jobs_menu()
        self._aggiorna_selettore()

    def _on_any_job_finished(self, job, _code):
        # Slot diretto di job_manager.job_finished, protetto con
        # _esegui_protetta per la stessa ragione di _on_job_started sopra --
        # un job che finisce non deve poter portarsi via l'intera finestra
        # per un errore di lettura del disco.
        self._esegui_protetta(lambda: self._on_any_job_finished_ora(job))

    def _on_any_job_finished_ora(self, job):
        # Un job di un altro progetto non tocca cio' che questa finestra sta
        # guardando: rileggere pipeline_bar/step badges e' lavoro sprecato
        # (entrambi scandiscono il disco) e una barra che si ridisegna sotto
        # le mani, per un progetto che non e' quello aperto, legge come un
        # errore. Il selettore invece si aggiorna sempre: e' l'unica
        # superficie che mostra anche i progetti che non sono quello aperto.
        # `job` puo' essere None -- una chiamata diretta che vuole solo
        # forzare il ricalcolo dal disco, senza un job vero dietro -- e in
        # quel caso non c'e' nessun workspace da confrontare: si aggiorna.
        if job is None or stesso_workspace(identita_workspace(self.workspace), job.identita):
            self.pipeline_bar.refresh()
            self._refresh_step_badges()
        self._aggiorna_selettore()
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
        # Un passo con stage ma assente dalla lista -- da questo ciclo, ogni
        # passo della famiglia cura-faceset: pipeline_bar.select() sopra ha
        # gia' portato in primo piano la pagina (apri_pagina_faceset), non
        # la scheda "Steps" dove vive step_view. Il Misc menu (i tre
        # KIND_VIEWER) e la navigazione da job continuano a passare da qui,
        # quindi la scheda va riportata avanti o la vista costruita sotto
        # resterebbe nascosta -- isVisible() falso, non un crash.
        self.central_tabs.setCurrentIndex(0)
        self.step_list.clearSelection()
        self.step_view.set_step(step)

    def _on_stage_selected(self, stage_name):
        if stage_name == STAGE_CURA_FACESET:
            # La fase E' la pagina: la lista dei passi non elenca piu' i
            # diciassette, e la scheda si apre (o torna in primo piano --
            # lezione della voce 3.21: QTabWidget seleziona da se' solo la
            # prima scheda di un contenitore vuoto).
            self.step_list.clear()
            self.apri_pagina_faceset()
            return
        if stage_name == STAGE_ESTRAZIONE:
            # Stessa scelta di STAGE_CURA_FACESET: la fase E' la pagina, le
            # sei voci del catalogo non compaiono piu' come lista.
            self.step_list.clear()
            self.apri_pagina_estrazione()
            return
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
        """The most recent job of `step` on *this window's* workspace, or None.

        Scans `_jobs_in_order` newest-first and returns the first match, not
        the whole history: on 51 `main` steps exactly one -- "3) cut video
        (drop video on me)" -- neither produces nor modifies any artifact,
        so the conflict matrix has nothing to contend and lets a second copy
        of that step start, and this is the one the step view should show.

        The workspace check (C1, found by re-reading the whole multi-project
        cycle end to end) is not optional: the catalog's step objects are
        module-level singletons (`gui/catalog/__init__.py`), so with a
        training active on project A and the window on project C, selecting
        the same-named step used to hand the step view A's job just because
        `job.step is step` matched -- from there Stop sent `close` to A's
        training and Force stop killed it, from a screen that shows C. Same
        predicate as `_job_attivi_su`: identity or path, never `==` on a bare
        path. `_refresh_running_jobs_menu` deliberately does *not* filter --
        it is the one surface that lists the jobs of projects you are not
        looking at.
        """
        identita = identita_workspace(self.workspace)
        for job in reversed(self._jobs_in_order):
            if job.step is step and stesso_workspace(identita, job.identita):
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

        self._project_menu = menubar.addMenu(testi.MENU_PROJECT)
        new_action = QAction(testi.MENU_NEW_PROJECT, self)
        new_action.setToolTip(testi.MENU_NEW_PROJECT_TIP)
        new_action.triggered.connect(lambda: self._new_project_dialog())
        self._project_menu.addAction(new_action)
        open_action = QAction(testi.MENU_OPEN_PROJECT, self)
        open_action.setToolTip(testi.MENU_OPEN_PROJECT_TIP)
        open_action.triggered.connect(lambda: self._open_project_dialog())
        self._project_menu.addAction(open_action)
        self._recent_menu = self._project_menu.addMenu(testi.MENU_RECENT)
        rename_action = QAction(testi.MENU_RENAME_PROJECT, self)
        rename_action.setToolTip(testi.MENU_RENAME_PROJECT_TIP)
        rename_action.triggered.connect(lambda: self._rename_project_dialog())
        self._project_menu.addAction(rename_action)
        duplicate_action = QAction(testi.MENU_DUPLICATE_PROJECT, self)
        duplicate_action.setToolTip(testi.MENU_DUPLICATE_PROJECT_TIP)
        duplicate_action.triggered.connect(lambda: self._duplicate_project_dialog())
        self._project_menu.addAction(duplicate_action)
        delete_action = QAction(testi.MENU_DELETE_PROJECT, self)
        delete_action.setToolTip(testi.MENU_DELETE_PROJECT_TIP)
        delete_action.triggered.connect(lambda: self._delete_project_dialog())
        self._project_menu.addAction(delete_action)
        self._project_menu.addSeparator()
        clear_action = QAction(testi.MENU_CLEAR_WORKSPACE, self)
        clear_action.setToolTip(testi.MENU_CLEAR_WORKSPACE_TIP)
        clear_action.triggered.connect(lambda: self.run_clear_workspace())
        self._project_menu.addAction(clear_action)
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
            titolo = testi.tab_title(self._nome_progetto_di(job), job.step.name)
            action = QAction(titolo, self.running_jobs_menu)
            action.setToolTip(testi.RUNNING_JOB_TIP)
            action.triggered.connect(
                lambda _checked=False, j=job: self._vai_al_job(j))
            self.running_jobs_menu.addAction(action)
        self.running_jobs_menu.menuAction().setVisible(
            not self.running_jobs_menu.isEmpty())

    def _nome_progetto_di(self, job):
        """The readable name of `job`'s project, or the folder name when
        it isn't (or isn't anymore) a project -- an unmigrated install."""
        progetto = leggi_progetto(job.workspace)
        return progetto.nome if progetto is not None else job.workspace.name

    def _vai_al_job(self, job):
        """Reach the job itself, not the same-named step of whatever
        project is on screen.

        With jobs on more than one project, selecting by step name lands on
        the right row of the wrong list: another project's step can share
        the name, with a different workspace underneath it.

        Protetta con _esegui_protetta: select_step (in fondo) solleva
        KeyError se il nome del passo non e' nel catalogo -- non dovrebbe
        mai succedere per un job vero, ma e' comunque uno slot Qt raggiunto
        da un click, non vale il rischio di un crash per una difesa che
        costa una riga.
        """
        self._esegui_protetta(lambda: self._vai_al_job_ora(job))

    def _vai_al_job_ora(self, job):
        pannello = self._panels.get(job)
        if pannello is not None:
            self.central_tabs.setCurrentWidget(pannello)
            return
        console = self.console_of(job)
        if console is not None:
            self.console_tabs.setCurrentWidget(console)
            self.console_dock.show()
            return
        if job.workspace != self.workspace:
            self.switch_workspace(job.workspace)
        self.select_step(job.step.name)

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

    def _esegui_protetta(self, azione):
        """Esegue `azione()` proteggendo lo slot Qt che la chiama.

        Uno slot Qt e' un vicolo cieco per un'eccezione -- stessa ragione di
        `_on_telemetry_event`, sopra: PyQt5 la trasforma in qFatal e il
        processo se ne va con dentro ogni altro training aperto nella stessa
        finestra. Usato da ogni azione del menu Project che tocca il disco --
        crea, apri, rinomina, elimina, svuota, riallinea la cartella, passa
        a un altro progetto -- cosi' un permesso negato, un antivirus che
        tiene un handle, una finestra di Esplora risorse aperta sulla
        cartella diventano un avviso, non un crash. Un aiutante solo, non
        una copia per azione: due copie di questa protezione potrebbero
        divergere proprio nel punto in cui serve.

        Cattura `Exception` senza distinguere: un `AttributeError` da un
        difetto di programmazione arriva all'utente con lo stesso avviso di
        un permesso negato sul disco. E' una scelta, non una svista --
        `_on_telemetry_event`, sopra, ha gia' la stessa postura in questo
        file -- ma e' il rischio tipico di questa rete: un vero difetto puo'
        diventare un avviso invece di un traceback visibile durante lo
        sviluppo. Va saputo, non solo accettato in silenzio.

        Torna True se `azione()` e' arrivata in fondo, False se ha
        sollevato -- chi chiama puo' usarlo per non proseguire dopo un
        fallimento.
        """
        try:
            azione()
        except Exception as error:
            QMessageBox.warning(
                self, testi.TITLE_PROJECT_ACTION_FAILED, testi.msg_project_action_failed(str(error)))
            return False
        return True

    def _aggiorna_selettore(self):
        """Ricostruisce il pulsante-selettore dallo stato corrente.

        `occupati` porta la cartella di ogni job attivo, con le ripetizioni
        -- non un insieme (era cosi' prima di questa correzione, ed
        e' per cui il tooltip diceva sempre "1 running": due job sullo
        stesso progetto collassavano nella stessa voce di un insieme).
        Confrontate per percorso (`SelettoreProgetti._occupati_quanti`) e
        non per identita' del filesystem come `_job_attivi_su`: qui il costo
        di uno `stat` per progetto a ogni ridisegno non e' giustificato da
        un pallino/conteggio che nel peggiore dei casi sbaglia di poco -- un
        difetto cosmetico, non una corsa sui dati.
        """
        occupati = [job.workspace for job in self.job_manager.active_jobs()]
        self.selettore.aggiorna(self.archivio.elenca(), occupati)
        self.selettore.imposta_corrente(leggi_progetto(self.workspace))

    def proponi_migrazione(self):
        """Propone di spostare il vecchio workspace dentro un progetto.

        Mai automatica: sposta i dati dell'utente, e una cartella che si
        muove da sola e' cio' che si scopre quando un backup non trova piu'
        la sorgente. Chiamato una volta, all'avvio, dopo che la finestra e'
        costruita ma prima che sia mostrata. Protetto con _esegui_protetta
        come ogni altra azione di questo menu che tocca il disco: un
        permesso negato durante lo spostamento non deve costare la finestra
        intera all'avvio.
        """
        self._esegui_protetta(self._proponi_migrazione_ora)

    def _proponi_migrazione_ora(self):
        if str(self._settings.value("skipMigration", "")) == "1":
            return
        if not serve_migrazione(self.archivio.radice):
            return
        finestra = QMessageBox(self)
        finestra.setWindowTitle(testi.TITLE_MIGRATE)
        finestra.setText(testi.msg_migrate(self.archivio.radice))
        finestra.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        finestra.setDefaultButton(QMessageBox.No)
        non_chiedere = QCheckBox(testi.DONT_ASK_AGAIN, finestra)
        finestra.setCheckBox(non_chiedere)
        if finestra.exec_() != QMessageBox.Yes:
            if non_chiedere.isChecked():
                self._settings.setValue("skipMigration", "1")
            return
        nome, ok = QInputDialog.getText(
            self, testi.TITLE_MIGRATE, testi.PROMPT_MIGRATE_NAME)
        if not ok or not nome.strip():
            return
        # migra() scrive project.json prima di spostare le sottocartelle
        # (vedi il suo stesso docstring): un fallimento a meta' spostamento
        # lascia comunque un progetto valido e gia' visibile su disco, con
        # dentro la parte che ce l'ha fatta. Il refresh nel finally lo porta
        # subito nel selettore anche in quel caso -- non dopo un riavvio --
        # cosi' chi vede l'avviso di errore ha gia' davanti dove sono finiti
        # i dati spostati, invece di dover indovinare.
        try:
            progetto = self.archivio.migra(nome.strip())
        finally:
            self._aggiorna_selettore()
        self.switch_workspace(progetto.cartella)

    def _new_project_dialog(self):
        nome, ok = QInputDialog.getText(
            self, testi.DIALOG_NEW_PROJECT, testi.DIALOG_NEW_PROJECT_NAME_LABEL)
        if not ok or not nome.strip():
            return
        self._esegui_protetta(lambda: self._crea_progetto_ora(nome.strip()))

    def _crea_progetto_ora(self, nome):
        progetto = self.archivio.crea(nome)
        self.switch_workspace(progetto.cartella)

    def _open_project_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, testi.DIALOG_OPEN_PROJECT, str(self.archivio.radice))
        if not path:
            return
        self._esegui_protetta(lambda: self._apri_progetto_ora(path))

    def _apri_progetto_ora(self, path):
        try:
            progetto = self.archivio.apri(path)
        except ValueError:
            QMessageBox.warning(self, testi.TITLE_NOT_A_PROJECT, testi.msg_not_a_project(path))
            return
        self.switch_workspace(progetto.cartella)

    def _rename_project_dialog(self):
        """Cambia il nome leggibile del progetto corrente, poi propone di
        riallineare anche la cartella -- vedi _propose_folder_tidy."""
        progetto = leggi_progetto(self.workspace)
        if progetto is None:
            return
        nome, ok = QInputDialog.getText(
            self, testi.TITLE_RENAME_PROJECT, testi.PROMPT_RENAME_PROJECT, text=progetto.nome)
        if not ok or not nome.strip():
            return
        self._esegui_protetta(lambda: self._rinomina_progetto_ora(progetto, nome.strip()))

    def _rinomina_progetto_ora(self, progetto, nome):
        progetto = self.archivio.rinomina(progetto, nome)
        self._aggiorna_selettore()
        self._propose_folder_tidy(progetto)

    def _propose_folder_tidy(self, progetto):
        """Se la cartella puo' essere riallineata al nome, adesso, lo chiede.

        Chiamato subito dopo una rinomina e di nuovo ogni volta che il
        progetto viene aperto (switch_workspace, sotto): la condizione puo'
        diventare vera piu' tardi -- il job che la bloccava finisce, il nome
        di destinazione si libera -- senza che l'utente abbia rifatto Rename.
        Un rifiuto (compreso "No" alla conferma) semplicemente non fa nulla:
        si puo' riprovare piu' tardi. Protetto con _esegui_protetta come ogni
        altra azione di questo menu che tocca il disco (os.replace dentro
        riconcilia, la scrittura del puntatore, i recenti) -- un fallimento
        qui non deve costare la finestra intera, e riportarlo separatamente
        da chi ha chiamato (rinomina o switch_workspace) dice all'utente cosa
        e' davvero andato storto: il progetto e' comunque gia' rinominato o
        gia' aperto a quel punto, solo il riallineamento non e' riuscito.
        """
        self._esegui_protetta(lambda: self._propose_folder_tidy_ora(progetto))

    def _propose_folder_tidy_ora(self, progetto):
        job_attivi = len(self._job_attivi_su(progetto.cartella))
        if not self.archivio.riconciliabile(progetto, job_attivi):
            return
        vecchia = progetto.cartella
        nuova = self.archivio.radice / slug(progetto.nome)
        risposta = QMessageBox.question(
            self, testi.TITLE_TIDY_FOLDER, testi.msg_tidy_folder(vecchia, nuova),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if risposta != QMessageBox.Yes:
            return
        risultato = self.archivio.riconcilia(progetto)
        if risultato is None:
            return
        if self.workspace == vecchia:
            self.workspace = risultato
            self.pipeline_bar.set_workspace(risultato)
            self.step_view.set_workspace(risultato)
        # La vecchia cartella non esiste piu' -- os.replace l'ha appena
        # spostata -- quindi va tolta dai recenti, non solo affiancata dalla
        # nuova: restare cliccabile punterebbe la finestra su un percorso
        # morto. Diverso dal caso di un disco esterno scollegato (una voce
        # legittima che qui non si tocca): li' non sappiamo che il percorso
        # non esiste, qui lo sappiamo per certo, l'abbiamo spostato noi.
        recenti = RecentWorkspaces()
        recenti.remove(vecchia)
        recenti.add(risultato)
        # Senza questo, il menu a schermo resta con la voce morta finche'
        # qualcos'altro non ricostruisce il sottomenu Recent -- e cliccarla
        # riaprirebbe un progetto fantasma sulla cartella appena spostata.
        self._refresh_recent_menu()
        self._aggiorna_selettore()

    def _duplicate_project_dialog(self):
        """Copia il modello, i dataset o i video del progetto corrente in
        uno nuovo -- il dialogo (nome, cosa, avanzamento annullabile) vive
        in gui.duplicazione, cosi' la copia gira sul suo thread invece di
        bloccare la finestra per i minuti che un model/ o un data_dst/
        possono richiedere.

        Protetta con _esegui_protetta come ogni altra voce di questo menu
        che tocca il disco: la copia stessa e' gia' al riparo di un
        try/except dentro CopiaProgetto.run (ogni eccezione che duplica()
        solleva, non solo DuplicazioneIncompleta -- C2, revisione finale),
        ma switch_workspace, alla fine, e' comunque un'azione sincrona sul
        thread dell'interfaccia che puo' sollevare per le stesse ragioni di
        ogni altra qui.
        """
        progetto = leggi_progetto(self.workspace)
        if progetto is None:
            return
        self._esegui_protetta(lambda: self._duplica_progetto_ora(progetto))

    def _duplica_progetto_ora(self, progetto):
        nuovo = DialogoDuplicazione(self.archivio, progetto, self).esegui()
        if nuovo is None:
            return
        self.switch_workspace(nuovo.cartella)

    def _delete_project_dialog(self):
        """Elimina il progetto corrente -- l'unica operazione davvero
        irreversibile del menu Project.

        Rifiutata con un job attivo sul progetto, per la stessa ragione di
        run_clear_workspace: elimina fa shutil.rmtree sulla cartella, e farlo
        sotto un processo che ci scrive e' una corsa sui dati, non solo un
        difetto cosmetico. Il conteggio passa da _job_attivi_su, non da un
        confronto diretto fra percorsi -- due nomi diversi della stessa
        cartella devono contare come la stessa cartella.

        La conferma ha "No" come predefinito, nomina il percorso per esteso
        e la dimensione su disco -- niente ridigitazione del nome, che
        sarebbe l'unico punto della finestra con quello stile: la dimensione
        in GB comunica lo stesso peso in modo piu' utile.
        """
        progetto = leggi_progetto(self.workspace)
        if progetto is None:
            return
        active = self._job_attivi_su(progetto.cartella)
        if active:
            QMessageBox.warning(
                self, testi.TITLE_JOBS_RUNNING,
                testi.msg_cannot_delete_project(len(active), progetto.cartella))
            return
        gigabyte = dimensione(progetto.cartella) / 1e9
        risposta = QMessageBox.question(
            self, testi.TITLE_DELETE_PROJECT,
            testi.msg_confirm_delete_project(progetto.cartella, gigabyte),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if risposta != QMessageBox.Yes:
            return
        # Protetto con _esegui_protetta: shutil.rmtree puo' sollevare per un
        # permesso negato, un antivirus o Esplora risorse che tengono un
        # handle aperto sulla cartella -- scenari ordinari su Windows, non
        # ipotetici. Senza questa rete, l'eccezione arriverebbe fino a
        # QAction.triggered (uno slot Qt) e PyQt5 la trasformerebbe in
        # qFatal: il processo se ne andrebbe con dentro ogni altro training
        # aperto nella stessa finestra, per un'operazione che si voleva solo
        # rifiutare con un avviso.
        self._esegui_protetta(lambda: self._elimina_progetto_ora(progetto))

    def _elimina_progetto_ora(self, progetto):
        self.archivio.elimina(progetto)
        RecentWorkspaces().remove(progetto.cartella)
        restanti = self.archivio.elenca()
        self.switch_workspace(restanti[0].cartella if restanti else self.archivio.radice)

    def _job_attivi_su(self, workspace):
        """I job attivi che scrivono in `workspace`, di qualunque progetto sia.

        Il confronto passa dal predicato di gui/progetti.py, non da == fra
        percorsi: due nomi diversi della stessa cartella devono contare come
        la stessa cartella, o una cancellazione partirebbe sotto un processo
        che ci sta scrivendo.
        """
        identita = identita_workspace(workspace)
        return [job for job in self.job_manager.active_jobs()
                if stesso_workspace(identita, job.identita)]

    def switch_workspace(self, path):
        """Punta la finestra su `path`. Permesso anche con job in corso.

        L'ambiente di un job (WORKSPACE, e ogni percorso risolto contro di
        esso) e' fissato al lancio: cambiare progetto qui non lo tocca, il
        job continua a scrivere dove e' partito. Cio' che cambia e' solo
        cosa mostra la finestra -- compreso il puntatore al progetto attivo
        (`ArchivioProgetti.imposta_attivo`), scritto solo quando `path` e'
        davvero un progetto: e' cio' che fa seguire la riga di comando.

        E' lo slot diretto del segnale del selettore di progetti
        (`progetto_scelto`) e di ogni voce del sottomenu Recent, oltre a
        essere chiamato da altre azioni gia' protette (New/Open/Delete): per
        questo e' protetta anche lei, con _esegui_protetta -- non solo chi
        la chiama.
        """
        self._esegui_protetta(lambda: self._switch_workspace_ora(Path(path)))

    def _switch_workspace_ora(self, path):
        self.workspace = path
        # La cache dei volti vive fuori dai progetti (decisione dell'utente:
        # duplicare o rinominare un progetto non deve copiare gigabyte di
        # maschere), e il prezzo di quella scelta si paga qui -- cancellare
        # un progetto lascia una cache orfana, e senza questa riga
        # `_internal/_e` cresce per sempre. Al CAMBIO di progetto e non alla
        # costruzione della pagina: la pagina puo' non essere mai aperta, e
        # questo e' il momento in cui il disco e' appena cambiato. La
        # potatura si decide sull'origine registrata in ogni `meta.json`,
        # quindi non ha bisogno di sapere quali progetti esistono; non
        # solleva mai (ogni ramo cattura OSError da se').
        from gui.faceset import cache as cache_faceset
        cache_faceset.pota_orfane(self._dfl_root.parent / "_e")
        self.pipeline_bar.set_workspace(path)
        self.step_view.set_workspace(path)
        if self._pagina_faceset is not None:
            self._pagina_faceset.imposta_workspace(path)
        if self._pagina_estrazione is not None:
            # apri() e non imposta_workspace(): la pagina di estrazione
            # segue (progetto, lato), non solo il progetto -- si riapre sul
            # lato che stava gia' mostrando.
            self._pagina_estrazione.apri(path, self._pagina_estrazione.lato())
        RecentWorkspaces().add(path)
        progetto = leggi_progetto(path)
        if progetto is not None:
            self.archivio.imposta_attivo(progetto)
        else:
            # I1, variante (revisione finale): la finestra mostra una
            # cartella che non e' un progetto (la radice, un'installazione
            # non ancora migrata) -- il puntatore non deve continuare a
            # nominare il progetto precedente, o la riga di comando
            # lavorerebbe su un progetto che la GUI non mostra piu' nemmeno
            # a sessione aperta, non solo al prossimo riavvio.
            self.archivio.pulisci_attivo()
        self.step_list.clear()
        self.step_view.set_step(None)
        self._refresh_recent_menu()
        self._aggiorna_selettore()
        self.pipeline_bar.select(self.pipeline_bar.stage_names()[0])
        if progetto is not None:
            self._propose_folder_tidy(progetto)

    def run_clear_workspace(self):
        """Confirm (default No), then empty and recreate the workspace's subdirectories.

        Refused while a job is active on *this* workspace -- a job's process
        writes into the workspace on disk, and `clear_workspace` runs
        `shutil.rmtree` on it, so clearing under a running job is a live
        filesystem race. A job active on a different project does not block
        this: `_job_attivi_su` scopes the check to the workspace being
        cleared.

        Protetta con _esegui_protetta come ogni altra voce del menu Project
        che tocca il disco: `clear_workspace` fa `shutil.rmtree` senza un
        proprio try/except, la stessa classe del Critical di Delete... --
        non una simile, la stessa -- e questa voce sta nello stesso
        `self._project_menu`, un click accanto a quella gia' protetta.
        """
        active = self._job_attivi_su(self.workspace)
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
        self._esegui_protetta(self._clear_workspace_ora)

    def _clear_workspace_ora(self):
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
        # Only what the user actually touched -- the same set answers()
        # already restricted itself to -- goes into the project's memory,
        # so the next time this step is opened it preloads exactly what was
        # sent, not the whole form. A project.json the disk refuses to
        # write (permission, an antivirus holding a handle) must not take
        # the window down with it: this runs through the same guard as
        # every other action that touches the project on disk.
        self._esegui_protetta(lambda: self._ricorda_risposte(step, answers, form))

    def _ricorda_risposte(self, step, answers, form):
        # La fusione e le sue conseguenze stanno in gui/progetti.py: la
        # pagina di estrazione ricorda dalla stessa funzione, e due copie
        # della stessa regola sono due regole appena una delle due cambia.
        # `last_step`/`last_model_name` vivono accanto alle risposte nella
        # stessa memoria, quindi passano di li' e il file si scrive una
        # volta sola.
        extra = {"last_step": step.name}
        if step.needs_model_name:
            extra["last_model_name"] = form.model_name()
        ricorda_risposte(self.workspace, step.name, answers, extra)

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
        return testi.tab_title(self._nome_progetto_di(job), job.step.name)

    # -- la pagina di cura del faceset ----------------------------------------

    def apri_pagina_faceset(self):
        """Mostra la scheda della pagina, costruendola la prima volta.

        Una sola pagina per finestra, indicizzata per progetto e non per
        job -- a differenza di TrainingPanel. `indexOf` prima di `addTab` e'
        la stessa cautela di `open_panel`: QTabWidget seleziona da se' solo
        la prima scheda di un contenitore vuoto (voce 3.21).
        """
        if self._pagina_faceset is None:
            from gui.faceset import avvio
            from gui.faceset.pagina import PaginaCuraFaceset
            avvio.configura(self._python_exe, self._dfl_root)
            self._pagina_faceset = PaginaCuraFaceset(
                self._dfl_root.parent / "_e", self._settings)
            # Prima del workspace: e' il gestore a dire quali azioni sono
            # libere, e imposta_workspace rigenera gia' la barra.
            self._pagina_faceset.imposta_job_manager(self.job_manager)
            self._pagina_faceset.imposta_workspace(self.workspace)
        indice = self.central_tabs.indexOf(self._pagina_faceset)
        if indice < 0:
            indice = self.central_tabs.addTab(self._pagina_faceset, testi.TAB_FACESET)
            # Come "Steps": una scheda sola, sempre presente, che segue lo
            # stato del progetto -- non un artefatto per job da poter
            # scartare e ricostruire. Il bottone di chiusura non farebbe
            # niente (non e' in self._panels), quindi non c'e'.
            self.central_tabs.tabBar().setTabButton(indice, QTabBar.RightSide, None)
        self.central_tabs.setCurrentIndex(indice)
        return self._pagina_faceset

    # -- la pagina di estrazione ----------------------------------------

    def apri_pagina_estrazione(self):
        """Mostra la scheda della pagina di estrazione, costruendola la
        prima volta. Stessa cautela di apri_pagina_faceset: `indexOf` prima
        di `addTab` (voce 3.21), una sola pagina per finestra."""
        if self._pagina_estrazione is None:
            from gui.estrazione import avvio
            from gui.estrazione.pagina import PaginaEstrazione
            avvio.configura(self._python_exe, self._dfl_root)
            self._pagina_estrazione = PaginaEstrazione(self._dfl_root.parent / "_e")
            self._pagina_estrazione.imposta_job_manager(self.job_manager)
            self._pagina_estrazione.apri(self.workspace, "src")
        indice = self.central_tabs.indexOf(self._pagina_estrazione)
        if indice < 0:
            indice = self.central_tabs.addTab(self._pagina_estrazione, testi.TAB_ESTRAZIONE)
            self.central_tabs.tabBar().setTabButton(indice, QTabBar.RightSide, None)
        self.central_tabs.setCurrentIndex(indice)
        return self._pagina_estrazione

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
        titolo = testi.tab_title(self._nome_progetto_di(job), job.step.name)
        return testi.running_tab_title(titolo) if job.running else titolo

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
            # set_job() above just wrote its own unprefixed default
            # ("running"/"finished") -- re-set it through _set_job_status so
            # the strip carries the project name (I2) from the first paint,
            # not only after the next navigation or status change.
            self._set_job_status(job, testi.JOB_RUNNING)
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

        `text` itself is stored raw (unprefixed), and the project name is
        added only where it is displayed: with jobs on more than one
        project (I2, found alongside C1), a bare
        "running"/"stopping…" gives no way to notice which project it
        belongs to -- the same reason C1 fixed which job the strip can
        attach to at all. Wrapping happens here, in the one place every
        caller of this method already goes through, so every status this
        strip ever shows -- not just the first one -- carries the name.
        """
        self._job_status_text[job] = text
        if self.step_view.job is job:
            self.step_view.job_status.setText(
                testi.job_strip_status(self._nome_progetto_di(job), text))

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
                self.console_tabs.setTabText(index, testi.failed_console_title(self._console_title(job)))
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
        # Il servizio di estrazione manuale non e' un Job -- e' un QProcess
        # avviato fuori dal job_manager (avvio.py/servizio.py) -- quindi
        # `active_jobs()` sotto non lo vede affatto. Senza di lui la pagina puo'
        # restare aperta in modalita' manuale, la finestra chiudersi, e il
        # figlio sopravviverle -- un processo appeso resta un processo
        # appeso anche se, a differenza del servizio di FacesetDetail,
        # ExtractManual non carica alcun modello in VRAM (geometria pura).
        # Idempotente e innocuo se richiamato piu' volte: closeEvent puo'
        # rientrare mentre aspetta che i job finiscano (event.ignore() sotto).
        if self._pagina_estrazione is not None:
            self._pagina_estrazione.ferma_servizio()
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
