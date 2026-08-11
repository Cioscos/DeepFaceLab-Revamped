"""Ogni testo che l'utente legge, in un posto solo.

Qui dentro si scrive **in inglese**: e' cio' che compare a schermo. Ovunque
altro nel pacchetto si continua a lavorare in italiano -- nomi, commenti,
docstring -- perche' quello non si vede.

Una guardia impedisce che una stringa letterale finisca dentro un widget
altrove: se serve un testo nuovo, nasce qui. I testi con un buco dentro sono
funzioni, non stringhe con un `%` lasciato in giro, cosi' chi li chiama non
puo' sbagliare il numero di argomenti senza accorgersene.
"""
from gui.catalog.model import KIND_CLEAR, KIND_EBSYNTH, KIND_VIEWER
from gui.workspace import STATE_BLOCKED, STATE_DONE, STATE_READY

# -- la finestra ----------------------------------------------------------
WINDOW_TITLE = "DeepFaceLab"
TAB_STEPS = "Steps"
CONSOLE_DOCK = "Console"
STAGE_TIP = "Show this stage's steps in the list on the left."

# -- i menu' ----------------------------------------------------------------
MENU_WORKSPACE = "&Workspace"
MENU_VIEW = "&View"
MENU_MISC = "&Misc"
MENU_RECENT = "Recent"
MENU_RUNNING_JOBS = "Running jobs"

MENU_OPEN_WORKSPACE = "Open…"
MENU_OPEN_WORKSPACE_TIP = "Open an existing workspace folder."
MENU_NEW_WORKSPACE = "New…"
MENU_NEW_WORKSPACE_TIP = "Create a new workspace folder with the standard subdirectories."
MENU_CLEAR_WORKSPACE = "Clear workspace…"
MENU_CLEAR_WORKSPACE_TIP = "Empty and recreate the current workspace's standard subdirectories."
TOGGLE_CONSOLE_TIP = "Show or hide the console dock."
MISC_STEP_TIP = "Jump to this step in the Steps tab."
RUNNING_JOB_TIP = "Switch to this running step."

MENU_TEXT_SIZE = "Text size"
TEXT_SIZE_LABELS = {"normal": "Normal", "large": "Large", "xlarge": "Extra large"}

# -- i comandi della vista passo --------------------------------------------
START = "Start"
START_TIP = "Run this step with the options above. The console tab opens with its output."
STOP = "Stop"
STOP_TIP = "Ask the running step to stop. A training run saves the model before it goes."
FORCE_STOP = "Force stop"
FORCE_STOP_TIP = "Kill the process tree immediately, without waiting for a clean stop."
SAVE = "Save"
SAVE_TIP = "Ask the running training to save a checkpoint now, without stopping it."
BACKUP = "Backup"
BACKUP_TIP = "Ask the running training to write a backup now, without stopping it."
SHOW_CONSOLE = "Show console"
SHOW_CONSOLE_TIP = "Open or close this step's console output."
OPEN_FOLDER = "Open folder"
OPEN_FOLDER_TIP = "Open this path in the system file manager."

# -- lo stato del job, sulla striscia e sulla scheda console ---------------
JOB_RUNNING = "running"
JOB_FINISHED = "finished"
STATUS_STOPPING = "stopping — waiting for the trainer to save"
STATUS_CLOSING = "closing…"

RUNNING_TAB_MARKER = "▶ "
FAILED_TAB_MARKER = " ✗"


def status_saved_waiting_to_close(iteration=None):
    """The strip's text while a clean stop is mid-save.

    `iteration` is the number the trainer just saved at, or None when the
    channel sent one that could not be read -- the sentence still says a
    save happened, it just drops the number instead of inventing one.
    """
    if iteration is None:
        return "saved — waiting for the trainer to close"
    return "saved at iter %d — waiting for the trainer to close" % iteration


def job_finished_status(exit_code):
    """The strip's text once the process has exited."""
    return "finished (exit %d)" % exit_code


def running_tab_title(step_name):
    """A training tab's title while its job is alive."""
    return RUNNING_TAB_MARKER + step_name


def failed_console_title(step_name):
    """A console tab's title once its job has exited with a non-zero code."""
    return step_name + FAILED_TAB_MARKER


# -- i due badge di layout, in cima alla vista passo -------------------------
BADGE_OPTIONAL = "optional"
BADGE_EXTERNAL_WINDOW = "opens an external window"


def step_list_optional_suffix():
    """Appended to a step's label in the list on the left."""
    return "  (%s)" % BADGE_OPTIONAL


def step_list_external_window_suffix():
    """Appended to a step's label in the list on the left."""
    return "  [%s]" % BADGE_EXTERNAL_WINDOW


# -- il badge di stato di una voce della lista dei passi (gui.delegato_passi)
STEP_BADGE_DONE = "done"
STEP_BADGE_READY = "ready"
STEP_BADGE_BLOCKED = "blocked"

_STEP_BADGE_LABEL = {
    STATE_DONE: STEP_BADGE_DONE,
    STATE_READY: STEP_BADGE_READY,
    STATE_BLOCKED: STEP_BADGE_BLOCKED,
}


def step_badge_label(stato):
    """The list item badge's visible word for a workspace.STATE_*, or ""
    when `stato` is not one of the three (missing, or a stray value)."""
    return _STEP_BADGE_LABEL.get(stato, "")


# -- perche' un passo o una fase e' in quello stato -------------------------
# Il colore dice *cosa*, queste frasi dicono *perche'*. Senza, "blocked" e'
# un vicolo cieco: l'utente vede il rosso e non sa quale passo fare prima,
# che e' esattamente la domanda per cui la pipeline esiste.
_ELENCO = ", "
STEP_TIP_DONE = "Done: it has already produced %s."
STEP_TIP_READY = "Ready to run: everything it needs is here (%s)."
STEP_TIP_BLOCKED = "Blocked: %s is missing. Run the step that produces it first."
STEP_TIP_BLOCKED_NO_INPUT = (
    "Blocked: this step declares no inputs, so its state cannot be worked out "
    "from what is on disk.")
STAGE_TIP_DONE = "Done: everything this stage produces is on disk (%s)."
STAGE_TIP_READY = "Ready to run: %s can start now."
STAGE_TIP_BLOCKED = "Blocked: %s is missing, so nothing here can run yet."
STAGE_TIP_BLOCKED_NO_INPUT = "Blocked: nothing here can run on this workspace yet."


def _artefatti(nomi):
    return _ELENCO.join(artifact_label(nome) for nome in nomi)


def step_state_tip(stato, mancanti, soddisfatti):
    """Why a step's badge says what it says. `""` for an unknown state.

    Takes what `gui.workspace.step_reason` returns, unpacked -- the rule
    lives there, the wording lives here, and neither knows the other's job.
    """
    if stato == STATE_DONE:
        return STEP_TIP_DONE % _artefatti(soddisfatti)
    if stato == STATE_READY:
        return STEP_TIP_READY % _artefatti(soddisfatti)
    if stato == STATE_BLOCKED:
        return (STEP_TIP_BLOCKED % _artefatti(mancanti) if mancanti
                else STEP_TIP_BLOCKED_NO_INPUT)
    return ""


def stage_state_tip(stato, mancanti, pronti):
    """Why a stage's pill has the colour it has, after `STAGE_TIP`."""
    if stato == STATE_DONE:
        return STAGE_TIP_DONE % _artefatti(pronti)
    if stato == STATE_READY:
        return STAGE_TIP_READY % (pronti[0] if pronti else "")
    if stato == STATE_BLOCKED:
        return (STAGE_TIP_BLOCKED % _artefatti(mancanti) if mancanti
                else STAGE_TIP_BLOCKED_NO_INPUT)
    return ""


# -- il segnaposto al posto del form, per i passi non KIND_MAIN -------------
PLACEHOLDER_VIEWER = "Opens %s in the system file manager."
PLACEHOLDER_CLEAR = "Empties and recreates the workspace's standard subdirectories."
PLACEHOLDER_EBSYNTH = "Launches the bundled EBSynth application."


def placeholder(kind, target=None):
    """The description shown in place of a form for a non-KIND_MAIN step."""
    if kind == KIND_VIEWER:
        return PLACEHOLDER_VIEWER % target
    if kind == KIND_CLEAR:
        return PLACEHOLDER_CLEAR
    if kind == KIND_EBSYNTH:
        return PLACEHOLDER_EBSYNTH
    return ""


# -- le finestre di dialogo native -------------------------------------------
DIALOG_OPEN_WORKSPACE = "Open workspace"
DIALOG_NEW_WORKSPACE_LOCATION = "New workspace location"
DIALOG_SELECT_FILE = "Select file"

# -- i titoli e i corpi dei QMessageBox --------------------------------------
TITLE_JOBS_RUNNING = "Jobs running"
TITLE_CLEAR_WORKSPACE = "Clear workspace"
TITLE_CLOSING = "Closing"
TITLE_MISSING_INPUT = "Missing input"
TITLE_MISSING_MODEL_NAME = "Missing model name"
TITLE_STEP_BUSY = "Step busy"
TITLE_FAILED_TO_START = "Failed to start"

MSG_CLOSING_CANNOT_START = "The window is closing: cannot start a new job."
MSG_MISSING_INPUT = "Select an input file before starting."
MSG_MISSING_MODEL_NAME = "Enter or choose a model name before starting."
MSG_CLOSING_WAIT = "Waiting for the active jobs to stop before closing."
MSG_CONFIRM_CLOSE_WITH_ACTIVE_JOBS = "Active jobs are still running. Stop them and close?"


def msg_cannot_switch_workspace(job_count, workspace):
    return "Cannot switch workspace: %d job(s) are still using %s." % (job_count, workspace)


def msg_cannot_clear_workspace(job_count, workspace):
    return "Cannot clear workspace: %d job(s) are still using %s." % (job_count, workspace)


def msg_confirm_clear_workspace(subdirs):
    return "This empties and recreates: %s. This cannot be undone. Continue?" % subdirs


# -- the artifacts, as the user reads them ----------------------------------
# The workflow's own artifact names are pinned to the formalization
# (`docs/…/workflow.toml`) and cannot be renamed without breaking that
# synchronization -- two of them, `modello` and `risultato`, are Italian, and
# they were reaching the "Step busy" dialog verbatim. This table translates
# on the way to the screen only; nothing downstream ever sees the label.
ARTIFACT_LABELS = {
    "video_src": "The source video",
    "video_dst": "The destination video",
    "frame_src": "The source frames",
    "frame_dst": "The destination frames",
    "faceset_src": "The source faceset",
    "faceset_dst": "The destination faceset",
    "debug_dst": "The destination debug faces",
    "modello": "The model",
    "merged": "The merged frames",
    "merged_mask": "The merged mask frames",
    "risultato": "The result video",
    "risultato_mask": "The result mask video",
    "xseg_generico": "The generic XSeg model",
    "pretrain": "The pretraining faceset",
}


def artifact_label(name):
    """A workflow artifact's name as the user should read it.

    An artifact with no declared label shows its own name rather than
    raising: a dialog that explains why a step was refused is the worst
    possible place to fail. A guard in the test suite requires a label for
    every entry of `gui.catalog.artifacts.ARTIFACTS`, so the degradation is
    the net under a mistake, never the normal path.
    """
    return ARTIFACT_LABELS.get(name, name)


def job_busy(artifact, running_step_name, is_artifact=True):
    """StepConflict's message: what is contended, and by whom.

    `is_artifact=False` for the one caller that contends over a *model*
    rather than a workflow artifact (the model lock, in
    gui/execution/jobs.py): a model the user named "modello" must keep its
    own name, not be relabelled through the table above -- and it is quoted,
    because it is a name the user typed, while an artifact's label is a
    phrase and reads worse inside quotes.
    """
    if not is_artifact:
        return "'%s' is busy: %s is using it" % (artifact, running_step_name)
    return "%s is busy: %s is using it" % (artifact_label(artifact), running_step_name)


def job_failed_to_start(program):
    """The console line appended when a job's process never starts at all."""
    return "failed to start: %s" % program


def reproducible_command(command_line, workspace, answers_path, events_path, commands_path):
    """The full text of the "Failed to start" dialog: the command line an
    operator could paste into a shell, followed by the environment the GUI
    set for it."""
    return "\n".join([
        command_line,
        "",
        "WORKSPACE=%s" % workspace,
        "DFL_ANSWERS_FILE=%s" % answers_path,
        "DFL_EVENTS_FILE=%s" % events_path,
        "DFL_COMMANDS_FILE=%s" % commands_path,
    ])


# -- i form (gui.forms) ------------------------------------------------------
BROWSE = "Browse…"
BROWSE_TIP = "Pick a file from disk and fill this field with its path."
INPUT_FILE = "Input file"
MODEL_NAME = "Model name"
SAVED_ANNOTATION = "saved %s"


def saved_value(value):
    """The text of the pill beside a field, showing its value on disk."""
    return SAVED_ANNOTATION % value


# -- la fascia d'aiuto (gui.fascia_aiuto) -------------------------------------
HELP_REST = "Hover a field, or move to it with Tab, to see what it does here."


def help_choice_title(label, choice):
    """The help strip's title while one dropdown entry is highlighted:
    the field's own label plus the entry currently under the mouse or
    keyboard, so what follows is unmistakably about *that* value."""
    return "%s — %s" % (label, choice)


# -- la scheda del training (gui/training_panel.py) --------------------------
LIVE = "Live"
REFRESH_PREVIEW = "Refresh preview"
PREVIEW = "preview"
PREVIEW_LABEL = "Preview"
LIVE_TIP = "Stop browsing history and follow the live %s again." % PREVIEW
REFRESH_PREVIEW_TIP = ("Ask the running training to draw a new %s now, "
                       "without waiting for the next automatic one." % PREVIEW)

RANGE_ALL = "All iterations"
RANGE_LAST = "Last %s"

# -- la barra sopra il grafico della loss (gui/training_panel.py) ------------
LOSS_CHART = "Loss chart"
RANGE_LABEL = "Range:"
RANGE_TIP = "This range picks how many recent iterations the chart below plots."
RANGE_ALL_TIP = "Plot every recorded iteration, from the start of the run."
RANGE_LAST_TIP = "Plot only the most recent iterations, dropping the rest."

SAMPLE_SELECTOR_TIP = ("Jump the large preview to this sample -- same as clicking "
                       "its thumbnail in the filmstrip below.")
PREVIEW_SELECTOR_TIP = ("Which of the images the model draws this panel shows. "
                        "Each one is a grid of faces with its own labels.")

# -- i due gesti, ripetuti dove si possono fare ------------------------------
# Il click e il doppio click sulle anteprime non hanno nessun segno a schermo:
# senza queste righe l'unico modo di scoprirli e' provarli a caso, il che vale
# quanto non averli.
_GESTI_CELLA = "Click to bring it into the large frame, double-click to open it at full size."
_GESTO_GRANDE = "Double-click to open it at full size, in its own window."


def cell_tip(label):
    """A side cell of the preview grid: what it shows, and the two gestures."""
    return "%s. %s" % (label, _GESTI_CELLA)


def big_frame_tip(label):
    """The large frame, when the model's descriptor names the cell it holds."""
    return "%s. %s" % (label, _GESTO_GRANDE)


BIG_FRAME_WHOLE_TIP = ("The whole preview image, as the model composed it: no "
                       "descriptor, so no per-cell labels. " + _GESTO_GRANDE)


def thumbnail_tip(index):
    """A thumbnail in the filmstrip: which sample, and the two gestures."""
    return "%s. %s" % (sample_label(index), _GESTI_CELLA)

SLIDER_NO_HISTORY = ("No history yet: turn on the model's write_preview_history "
                     "option to be able to scroll back in time.")
SLIDER_WITH_HISTORY = ("Drag to stop at a past iteration: preview and graph move "
                       "together. Drop it at the end to go live again.")


def sample_label(index):
    """The tooltip on a sample thumbnail in the filmstrip below the preview,
    and the matching entry in the sample selector beside it."""
    return "Sample %d" % index


def range_last_label(iterations):
    """A `self.intervallo` entry: "Last <n>", thousands separated by a thin
    space so a five- or six-digit count stays readable in a narrow combo box."""
    return RANGE_LAST % "{:,}".format(iterations).replace(",", " ")


def loss_legend_html(color_src, color_dst):
    """The chart legend's rich text: which color plots which series. The
    colors come from gui.loss_plot.COLORI so the legend can never drift
    away from what the chart actually draws."""
    return ('<span style="color:%s">src</span>&nbsp;&nbsp;'
           '<span style="color:%s">dst</span>' % (color_src, color_dst))


def cursor_live():
    """The chart cursor's caption while the panel follows the live training."""
    return LIVE


def cursor_at_iteration(iteration):
    """The chart cursor's caption while the panel is stopped in history:
    which iteration the preview and the graph are both showing."""
    return "Iteration %d" % iteration


def job_finished(exit_code):
    """The status strip once the training process has exited."""
    return "job finished (exit code %d)" % exit_code


# -- le tessere di stato (gui.tessere_stato) ----------------------------------
TILE_ITERATION = "Iteration"
TILE_ITERATION_HISTORY = "Iteration (history)"
TILE_LOSS = "Loss src / dst"
TILE_SPEED = "Speed"
TILE_ETA = "ETA"
TILE_VRAM = "VRAM"

#La chiave della tessera che il pannello sostituisce quando e' fermo nello
#storico: non viene da `status_line.valori_di_stato`, che parla sempre della
#corsa viva, ma dal pannello, che e' il solo a sapere dove il cursore si e'
#fermato. Chiave diversa e non solo valore diverso, cosi' `valori()` continua
#a dire cosa la tessera sta davvero raccontando.
TILE_KEY_ITERATION = "iteration"
TILE_KEY_ITERATION_HISTORY = "iteration in history"

_TILE_LABELS = {
    TILE_KEY_ITERATION: TILE_ITERATION,
    TILE_KEY_ITERATION_HISTORY: TILE_ITERATION_HISTORY,
    "loss src / dst": TILE_LOSS,
    "speed": TILE_SPEED,
    "ETA": TILE_ETA,
    "VRAM": TILE_VRAM,
}

_TILE_TIPS = {
    TILE_KEY_ITERATION: "How many training iterations this run has completed so far.",
    TILE_KEY_ITERATION_HISTORY:
        "The iteration the preview and the chart are stopped at. The run itself "
        "keeps going: press Live to follow it again.",
    "loss src / dst":
        "How wrong the model still is on the source and on the destination faces. "
        "Lower is better, and the trend over hours matters more than the number.",
    "speed": "Training iterations per second, measured over the last few of them.",
    "ETA": "Time left to the target iteration at the current speed. Absent when the "
           "run has no target, or when the speed is not known yet.",
    "VRAM": "Video memory the run is using, out of what the GPU has.",
}


def tile_label(key):
    """The caption shown on a status tile, from the internal key that
    `status_line.valori_di_stato` produces. An unknown key shows itself
    instead of vanishing without a word."""
    return _TILE_LABELS.get(key, key)


def tile_tip(key):
    """What a status tile's number means, for whoever has never seen one.
    Empty for an unknown key: a tooltip that says nothing is better than a
    tooltip that repeats the caption."""
    return _TILE_TIPS.get(key, "")


def loss_values_dropped(count):
    """How many loss values could not be plotted, when there are any."""
    if count == 1:
        return "1 loss value cannot be plotted (NaN or infinite)"
    return "%d loss values cannot be plotted (NaN or infinite)" % count


def preview_unreadable():
    """A preview was announced on the events channel but its file could not
    be read -- the previous image stays on screen instead."""
    return "preview announced but unreadable: keeping the last one"


def preview_not_drawable(kind, error):
    """The panel's own drawing code raised while building the preview grid."""
    return "preview cannot be drawn: %s: %s" % (kind, error)


def event_not_applied(kind, error):
    """An event from the channel could not be applied, of any kind."""
    return "event not applied: %s: %s" % (kind, error)


def loss_history_unreadable(error):
    """The loss history CSV could not be read."""
    return "loss history cannot be read: %s" % error


def iteration_not_usable(value):
    """The message inside the `ValueError` an `iter` event raises when its
    iteration cannot be trusted -- ends up quoted, via `event_not_applied`,
    on the status line, so it has to be in English like everything else
    there."""
    return "iteration not usable: %r" % (value,)


def preview_window_title(cell_label, iteration):
    """The natural-size preview window's title: which cell, at which
    iteration. Composed here, not where the window is built -- that window
    is a snapshot and never updates, so the iteration is the one fact worth
    stating up front."""
    if iteration is None:
        return "%s (iteration unknown)" % cell_label
    return "%s (iteration %d)" % (cell_label, iteration)
