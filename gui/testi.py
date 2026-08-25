"""Ogni testo che l'utente legge, in un posto solo.

Qui dentro si scrive **in inglese**: e' cio' che compare a schermo. Ovunque
altro nel pacchetto si continua a lavorare in italiano -- nomi, commenti,
docstring -- perche' quello non si vede.

Una guardia impedisce che una stringa letterale finisca dentro un widget
altrove: se serve un testo nuovo, nasce qui. I testi con un buco dentro sono
funzioni, non stringhe con un `%` lasciato in giro, cosi' chi li chiama non
puo' sbagliare il numero di argomenti senza accorgersene.
"""
# I codici di guasto del servizio di dettaglio. Un modulo di dati puri,
# senza nessun import suo: gui/ non importa mainscripts, e questa e' la
# stessa eccezione gia' fatta per MotoriCatalog, per la stessa ragione --
# l'elenco ha una sorgente sola, e leggerlo non costa torch.
from mainscripts import DettaglioGuasti

from gui.catalog.model import KIND_CLEAR, KIND_EBSYNTH, KIND_VIEWER
from gui.faceset.indice import STATO_ASSENTE, STATO_PARZIALE
from gui.workspace import STATE_BLOCKED, STATE_DONE, STATE_READY

# -- la finestra ----------------------------------------------------------
WINDOW_TITLE = "DeepFaceLab"
TAB_STEPS = "Steps"
TAB_FACESET = "Faceset curation"
TAB_ESTRAZIONE = "Extraction"
CONSOLE_DOCK = "Console"
STAGE_TIP = "Show this stage's steps in the list on the left."

# -- i menu' ----------------------------------------------------------------
MENU_PROJECT = "&Project"
MENU_VIEW = "&View"
MENU_MISC = "&Misc"
MENU_RECENT = "Recent"
MENU_RUNNING_JOBS = "Running jobs"

MENU_NEW_PROJECT = "New…"
MENU_NEW_PROJECT_TIP = "Create a new project with the standard subdirectories."
MENU_OPEN_PROJECT = "Open…"
MENU_OPEN_PROJECT_TIP = "Open an existing project folder."
MENU_RENAME_PROJECT = "Rename…"
MENU_RENAME_PROJECT_TIP = "Change the project's display name."
MENU_DUPLICATE_PROJECT = "Duplicate…"
MENU_DUPLICATE_PROJECT_TIP = "Copy this project's model, datasets or videos into a new one."
MENU_DELETE_PROJECT = "Delete…"
MENU_DELETE_PROJECT_TIP = "Delete this project and everything in it."
MENU_CLEAR_WORKSPACE = "Clear workspace…"
MENU_CLEAR_WORKSPACE_TIP = "Empty and recreate the current workspace's standard subdirectories."
TOGGLE_CONSOLE_TIP = "Show or hide the console dock."
MISC_STEP_TIP = "Jump to this step in the Steps tab."
RUNNING_JOB_TIP = "Switch to this running step."

MENU_REFRESH_STATE = "Refresh state"
MENU_REFRESH_STATE_TIP = ("Re-read the project folders from disk. Use it after "
                          "changing files outside the app: step states are read "
                          "when a job ends or the project changes, not continuously.")

MENU_TEXT_SIZE = "Text size"
TEXT_SIZE_LABELS = {"normal": "Normal", "large": "Large", "xlarge": "Extra large"}

# -- il selettore di progetti -------------------------------------------------
PROJECT_SELECTOR_TIP = "The project every step runs against. A dot marks projects with running steps."
NO_PROJECT = "No project"


def project_button(nome):
    return "Project: %s" % nome


def project_with_jobs(nome, quanti):
    return "%s — %d running" % (nome, quanti)


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


def tab_title(progetto, passo):
    """The title of a tab: which project it comes from, and which step it is."""
    return "%s · %s" % (progetto, passo)


def job_strip_status(progetto, testo):
    """The step view's job strip: which project the job belongs to, next to
    its current status ("running", "stopping — waiting…", "finished (exit
    0)", …). Without the project name here, a strip reading "running" gives
    no way to notice it belongs to a different project than the one on
    screen (I2, found alongside C1 by re-reading the whole multi-project
    cycle end to end)."""
    return "%s · %s" % (progetto, testo)


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
DIALOG_NEW_PROJECT = "New project"
DIALOG_NEW_PROJECT_NAME_LABEL = "Project name:"
DIALOG_OPEN_PROJECT = "Open project"
DIALOG_SELECT_FILE = "Select file"

TITLE_DUPLICATE_PROJECT = "Duplicate project"
COPYING_PROJECT = "Copying…"
CANCEL = "Cancel"
DUPLICATE_WHAT = "What should the new project start from?"
DUPLICATE_MODEL = "Model (trained weights)"
DUPLICATE_DATASET = "Datasets (data_src, data_dst)"
DUPLICATE_VIDEO = "Source videos"

# -- i titoli e i corpi dei QMessageBox --------------------------------------
TITLE_JOBS_RUNNING = "Jobs running"
TITLE_CLEAR_WORKSPACE = "Clear workspace"
TITLE_CLOSING = "Closing"
TITLE_MISSING_INPUT = "Missing input"
TITLE_MISSING_MODEL_NAME = "Missing model name"
TITLE_STEP_BUSY = "Step busy"
TITLE_FAILED_TO_START = "Failed to start"
TITLE_NOT_A_PROJECT = "Not a project"
TITLE_RENAME_PROJECT = "Rename project"
TITLE_TIDY_FOLDER = "Rename the folder too?"
TITLE_DELETE_PROJECT = "Delete project"
TITLE_PROJECT_ACTION_FAILED = "Action failed"
TITLE_DUPLICATE_INCOMPLETE = "Duplication incomplete"
TITLE_MIGRATE = "Move your workspace into a project?"

MSG_CLOSING_CANNOT_START = "The window is closing: cannot start a new job."
MSG_MISSING_INPUT = "Select an input file before starting."
MSG_MISSING_MODEL_NAME = "Enter or choose a model name before starting."
MSG_CLOSING_WAIT = "Waiting for the active jobs to stop before closing."
MSG_CONFIRM_CLOSE_WITH_ACTIVE_JOBS = "Active jobs are still running. Stop them and close?"
PROMPT_RENAME_PROJECT = "New name for this project:"
PROMPT_MIGRATE_NAME = "Name for this project:"
DONT_ASK_AGAIN = "Don't ask again"


def msg_not_a_project(path):
    return "%s has no project.json: it is not a project." % path


def msg_tidy_folder(vecchio, nuovo):
    return ("The project's name no longer matches its folder.\n\n"
            "Rename\n    %s\nto\n    %s ?\n\n"
            "Shortcuts, backups or scripts pointing at the old path will "
            "need updating." % (vecchio, nuovo))


def msg_cannot_clear_workspace(job_count, workspace):
    return "Cannot clear workspace: %d job(s) are still using %s." % (job_count, workspace)


def msg_cannot_delete_project(job_count, workspace):
    return "Cannot delete project: %d job(s) are still using %s." % (job_count, workspace)


def msg_confirm_delete_project(percorso, gigabyte):
    return ("Delete this project and everything in it?\n\n"
            "    %s\n    %.1f GB on disk\n\n"
            "This cannot be undone." % (percorso, gigabyte))


def msg_project_action_failed(error):
    """Shown when a project action (new, open, rename, delete, move) raised
    partway through -- caught before it could reach a Qt slot and take the
    whole application, and every training running in it, down with it."""
    return "The action could not be completed: %s" % error


def msg_duplicate_incomplete(destinazione):
    """Shown when a canceled duplication could not remove its own partial
    copy -- a denied permission, a handle still open on the folder. The
    destination is never a valid project (see DuplicazioneIncompleta), but
    it is not gone either, and the user is the only one who can clear it."""
    return ("The duplication was canceled, but the partial copy could not "
            "be removed:\n\n    %s\n\nCheck that folder by hand." % destinazione)


def msg_confirm_clear_workspace(subdirs):
    return "This empties and recreates: %s. This cannot be undone. Continue?" % subdirs


def msg_migrate(radice):
    return ("Your data currently sits directly in\n    %s\n\n"
            "Projects let you keep several sets of data side by side and "
            "work on one while another is training.\n\n"
            "Move data_src, data_dst, model and any videos into a project "
            "of their own? Nothing is deleted, and command-line scripts "
            "will follow." % radice)


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


# -- la pagina di cura del faceset ------------------------------------------
FACESET_SRC = "src"
FACESET_DST = "dst"
FACESET_INDEX_NOW = "Index now"
FACESET_UPDATE_INDEX = "Update"


def faceset_index_state(stato, mancanti):
    if stato == STATO_ASSENTE:
        return "Index this folder to enable pose distribution and masks."
    if stato == STATO_PARZIALE:
        return "%d faces not indexed." % mancanti
    return "Indexed."


def progress_bar_format(descrizione):
    """Il testo dentro la barra: la fase, poi i numeri che Qt sostituisce."""
    return "%s  %%v / %%m" % descrizione if descrizione else "%v / %m"


HEATMAP_TITLE = "Pose distribution"
HEATMAP_BINS_TIP = "How finely the yaw/pitch grid is divided."
HEATMAP_COLLAPSE_TIP = ("Hide the pose distribution and give the room to the "
                        "faces. Any active pose filter stays on.")


def heatmap_bins_label(n):
    """«8×8 bins»: la griglia e' quadrata, e un «8» solo non direbbe di che
    cosa -- otto colonne di imbardata, otto righe di beccheggio."""
    return "%d×%d bins" % (n, n)


def heatmap_legend(minimo, massimo, senza_posa):
    """La legenda dichiara i conteggi VERI agli estremi -- la scala e'
    logaritmica, e un colore senza numeri mentirebbe su quanto vale."""
    testo = "%d to %d faces per cell" % (minimo, massimo)
    if senza_posa:
        testo += " · %d faces without pose data are not shown" % senza_posa
    return testo


HEATMAP_FILTER_CLEAR = "Clear"

# L'href non e' un testo: e' la chiave che `linkActivated` consegna, e non
# la legge nessuno all'infuori del collegamento stesso. Sta qui perche' qui
# si compone il markup, non perche' sia un testo da tradurre.
_HEATMAP_FILTER_HREF = "clear"


def heatmap_filter_pill(mostrati, totali, quanti_bin):
    return "Showing %d of %d · %d pose bins" % (mostrati, totali, quanti_bin)


def heatmap_filter_pill_html(mostrati, totali, quanti_bin):
    """La stessa pastiglia col comando che la spegne DENTRO
    («Showing 412 of 1 619 · 2 pose bins · [Clear]»).

    Una funzione sola sopra l'altra e non due testi paralleli: con dei bin
    accesi la pastiglia e' l'unica cosa a schermo che dice che la griglia
    e' una fetta, e il modo di tornare indietro deve stare li' e non
    altrove.
    """
    return '%s · <a href="%s">%s</a>' % (
        heatmap_filter_pill(mostrati, totali, quanti_bin),
        _HEATMAP_FILTER_HREF, HEATMAP_FILTER_CLEAR)


def faceset_frame_filter_pill(mostrati, totali, nome_frame):
    return "Showing %d of %d · frame %s" % (mostrati, totali, nome_frame)


def faceset_frame_filter_pill_html(mostrati, totali, nome_frame):
    """La stessa pastiglia dei bin, con dentro lo stesso comando che la
    spegne: un solo posto dove si legge che la griglia e' una fetta, e un
    solo posto da cui si torna indietro."""
    return '%s · <a href="%s">%s</a>' % (
        faceset_frame_filter_pill(mostrati, totali, nome_frame),
        _HEATMAP_FILTER_HREF, HEATMAP_FILTER_CLEAR)


FACESET_SIZE_TIP = "How large the faces are drawn in the grid."
FACESET_MASK_OFF = "No mask"
FACESET_MASK_OVERLAY = "Mask overlay"
FACESET_MASK_ONLY = "Mask only"
FACESET_MASK_TIP = ("Show the XSeg masks the index holds: tinted over the "
                    "face, or on their own.")
FACESET_NO_MASKS = ("No face in this folder has an XSeg mask in the index. "
                    "Index the folder, or apply XSeg first.")
FACESET_LANDMARKS = "Landmarks"
FACESET_LANDMARKS_TIP = "Draw the 68 alignment points on top of the face."
FACESET_NO_LANDMARKS = "This face carries no landmarks: there is nothing to draw."
FACESET_SIBLING_PREV = "◀"
FACESET_SIBLING_NEXT = "▶"
FACESET_SIBLING_TIP = ("The other faces extracted from the same frame. The strip "
                       "beside the grid marks where they are in the current order.")
FACESET_NO_INDEX_FOR_SIBLINGS = ("Index the folder to see which faces come from "
                                 "the same frame.")
FACESET_MENU_SAME_FRAME = "Show aligned from the same frame"
FACESET_MENU_SAME_FRAME_TIP = ("Show only the faces extracted from the same frame "
                               "as this one.")
FACESET_MENU_ORIGINAL_FRAME = "Show original frame"
FACESET_MENU_ORIGINAL_FRAME_TIP = ("Open the Extraction page on the frame this "
                                   "face was extracted from.")


def faceset_sibling_counter(posizione, totale, nome_frame):
    """«2 of 4 from 00042.jpg»: la posizione dentro il gruppo, il gruppo
    intero (il volto corrente compreso) e il frame da cui vengono."""
    return "%d of %d from %s" % (posizione, totale, nome_frame)


def action_not_applicable(etichetta, nome_cartella):
    return "%s works on a folder of aligned faces; “%s” is not one." % (
        etichetta, nome_cartella)


def action_needs_unpack(etichetta):
    """Perche' un'operazione e' grigia su una cartella impacchettata.

    Dice il rimedio per nome ed esteso -- il menu in cui sta e la voce da
    cliccare -- perche' e' la stessa frase che l'utente legge nel
    suggerimento della voce grigia e nel messaggio al centro della griglia:
    chi arriva dall'uno o dall'altro deve trovare la stessa istruzione.
    """
    return ("%s cannot run while the faces are packed into %s. "
            "Unpack them first: Tools ▸ %s." % (etichetta, PACCHETTO_NOME,
                                                FACESET_UNPACK_LABEL))


# Il nome del file e l'etichetta della voce di menu vivono qui perche' due
# testi diversi (il suggerimento e il messaggio della griglia) li nominano
# entrambi: scritti due volte, si sarebbero riformulati una volta sola.
PACCHETTO_NOME = "faceset.pak"
FACESET_UNPACK_LABEL = "Faceset unpack"

FACESET_PACKED = ("The faces in this folder are packed into %s, so there is "
                  "nothing to show.\n\n"
                  "Unpack them to see them again: Tools ▸ %s."
                  % (PACCHETTO_NOME, FACESET_UNPACK_LABEL))


def faceset_cartella_vuota(nome_cartella):
    """L'altro vuoto: la cartella non ha volti, e non e' colpa del
    pacchetto. Due vuoti diversi, due frasi diverse -- la stessa scelta
    gia' fatta da `estrazione_cartella_vuota`."""
    return "No faces in %s yet." % nome_cartella


FACESET_FILTRO_VUOTO = "No faces match this filter."


def job_holds(nome_passo, artefatto):
    """Perche' un'azione della pagina e' grigia: chi sta scrivendo.

    Nomina il passo per intero, come l'utente lo legge nella lista: dire
    solo "busy" lascerebbe cercare la corsa che tiene la cartella fra
    tutte quelle aperte.
    """
    return "“%s” is running and holds %s." % (nome_passo, artifact_label(artefatto))


def action_src_only(etichetta):
    """Perche' un'operazione senza gemello dst non c'e' sul dataset dst.

    Il passo src esiste e girerebbe, ma dichiarerebbe di modificare il
    faceset src mentre riscrive quello dst: la pagina resterebbe verde
    sopra la cartella che si sta riscrivendo.
    """
    return "%s exists for the source dataset only." % etichetta


FACESET_TOOLS = "Tools"
FACESET_DELETE = "Delete"
FACESET_UNDO_DELETE = "Undo delete"
FACESET_OPEN_FOLDER = "Open in file manager"
FACESET_TOOLS_TIP = "Operations on the folder shown, not on the one the step normally points at."
FACESET_DELETE_TIP = "Move the selected faces to the folder's trash."
FACESET_UNDO_DELETE_TIP = "Put the faces of the last delete back where they were."
FACESET_OPEN_FOLDER_TIP = "Open the folder shown in the system file manager."
FACESET_NO_FOLDER = "This dataset has no folder to work on yet."
TITLE_FACESET_ONE_AT_A_TIME = "One at a time"


def faceset_one_job_at_a_time(nome_passo):
    """Perche' questa pagina lancia un job alla volta.

    Non e' solo la barra di avanzamento condivisa: indicizzare mentre si
    ordina e' sbagliato comunque, perche' il sort rinomina i file sotto
    l'indice.
    """
    return ("“%s” is still running on this page. Wait for it to finish: "
            "this page runs one job at a time." % nome_passo)


FACESET_DETAIL_TITLE = "Face detail"


# -- la pagina di estrazione -------------------------------------------------
ESTRAZIONE_AVVIA = "Start"
ESTRAZIONE_AVVIA_TIP = "Run the selected automatic operation as a job, with a dialog for its parameters."
ESTRAZIONE_MANUALE = "Manual session"
ESTRAZIONE_MANUALE_ESCI = "Exit manual session"
ESTRAZIONE_MANUALE_TIP = ("Start (or stop) the native manual extractor: draw the "
                          "face rectangle on the canvas above instead of the old "
                          "cv2 window.")
ESTRAZIONE_MANUALE_OCCUPA = "A manual session is running on this dataset. Exit it first."
# La barra pulsante fra l'ingresso in sessione (o il cambio di motore) e la
# prima risposta del figlio -- il primo `rileva` costruisce i modelli veri
# e li porta in VRAM, ed e' un'attesa senza nessun altro segnale a schermo.
ESTRAZIONE_CARICAMENTO_MOTORI = "Loading the detector and the landmarker…"
# La barra pulsante durante la lettura della mappa fotogramma -> volti
# (`PaginaEstrazione._ricostruisci_mappa`), fuori dal thread di Qt: su una
# cartella grande l'enumerazione costa secondi, e senza questa riga
# l'apertura sembrerebbe congelata. Non costruisce nessun indice -- quello
# lo scrive un passo a parte -- si limita a leggerlo e ad abbinarlo a cio'
# che c'e' sul disco: il testo non deve promettere di piu' di questo.
ESTRAZIONE_MAPPA_IN_COSTRUZIONE = "Reading the aligned faces…"
# I tre controlli dei motori, visibili solo durante la sessione manuale.
# Le voci delle tendine e i loro aiuti NON stanno qui: sono `label` e `help`
# di mainscripts/MotoriCatalog.py, che ne e' la sorgente unica.
ESTRAZIONE_RILEVATORE = "Detector:"
ESTRAZIONE_RILEVATORE_TIP = "Which model finds the face in the frame. Changing it detects this frame again."
ESTRAZIONE_ALLINEATORE = "Landmarks:"
ESTRAZIONE_ALLINEATORE_TIP = ("Which model places the 68 landmarks on the face found. "
                              "Changing it re-runs it on the rectangle you are on.")
ESTRAZIONE_MEMORIA = "Keep models in memory"
ESTRAZIONE_MEMORIA_TIP = ("Keep every model used in this session alive in VRAM, so "
                          "switching back and forth is instant. Unchecked, only the "
                          "two current ones stay and the others are freed right away. "
                          "It lowers what is held between switches, not the peak: "
                          "while a switch happens the old model is still loaded when "
                          "the new one is built.")
ESTRAZIONE_RIESTRAI = "Re-extract selection"
ESTRAZIONE_RIESTRAI_TIP = ("Clear the debug image of the selected frames and "
                           "re-run manual extraction for them only.")
ESTRAZIONE_ANNULLA_RIESTRAI = "Undo"
ESTRAZIONE_ANNULLA_RIESTRAI_TIP = "Put the debug images of the last re-extract selection back where they were."
ESTRAZIONE_INDICIZZA = "Rebuild report"
ESTRAZIONE_INDICIZZA_TIP = ("For a folder extracted before this page existed: rebuilds "
                            "the per-frame report from the aligned faces already on "
                            "disk, without re-running detection.")
ESTRAZIONE_VOLTI_DEL_FRAME = "Show faces from this frame"
ESTRAZIONE_VOLTI_DEL_FRAME_TIP = ("Open Faceset curation showing only the aligned "
                                  "faces extracted from the selected frame.")

ESTRAZIONE_SOVRAPP_RECT = "Rect"
ESTRAZIONE_SOVRAPP_RECT_TIP = ("Draw the face rectangles the report recorded "
                               "for this frame.")
ESTRAZIONE_SOVRAPP_LANDMARKS = "Landmarks"
ESTRAZIONE_SOVRAPP_LANDMARKS_TIP = ("Draw the landmarks stored inside the "
                                    "aligned faces of this frame. Reads them "
                                    "from disk the first time.")
ESTRAZIONE_SOVRAPP_MASCHERA = "XSeg mask"
ESTRAZIONE_SOVRAPP_MASCHERA_TIP = ("Overlay the XSeg mask of each aligned "
                                   "face back onto the frame, where one "
                                   "exists.")
ESTRAZIONE_SOVRAPP_MANUALE_TIP = ("Overlays apply while browsing frames. The "
                                  "manual session always shows the live "
                                  "detection, which comes from the engine "
                                  "and not from disk.")


def estrazione_frame_sparito(nome):
    return "%s is no longer in this folder." % nome


def estrazione_nessun_volto_indicizzato(nome):
    return ("No indexed face comes from %s. Index the aligned folder first." % nome)


def estrazione_stato(totale_frame):
    if totale_frame == 0:
        return "No frames in this folder yet."
    return "%d frame(s)." % totale_frame


def estrazione_cartella_vuota(nome_cartella):
    """Al centro dell'area vuota, non solo in fondo alla barra: dice anche
    cosa fare prima, perche' «frames» qui sono le immagini che i passi 2 e 3
    estraggono dal video, non i volti."""
    return ("No frames in %s yet.\n\n"
            "Extract images from a video first (steps 2 and 3), then come back."
            % nome_cartella)


ESTRAZIONE_FILTRO_VUOTO = "No frames match this filter."


def estrazione_volto_salvato(nome_file):
    return "Saved %s." % nome_file


def estrazione_salvataggio_fallito(motivo):
    """Senza un testo dedicato, un salvataggio
    fallito (servizio riavviato, disco pieno, permessi) restava invisibile
    -- `_su_confermato` avanzava comunque al frame successivo, e l'unico
    posto che leggeva `Servizio.ultimo_errore` era il ramo `rileva`, mai
    quello di `salva`."""
    return "Save failed: %s" % motivo


# Il ripiego quando _salva_corrente non
# ha un `ultimo_errore` da mostrare -- "Save failed: None" e' un testo
# peggiore di uno generico.
ESTRAZIONE_MOTIVO_SALVATAGGIO_IGNOTO = "unknown reason"


def estrazione_avvio_in_corso(nome_passo):
    return "Starting %s…" % nome_passo


def estrazione_job_in_corso(nome_passo):
    """M5 della revisione finale: `estrazione_avvio_in_corso` resta vera
    solo fra il click e la prima riga del figlio -- senza un testo
    dedicato per DOPO, la riga di stato mente per tutta la durata del job
    (quaranta minuti, su un'estrazione vera)."""
    return "Running %s…" % nome_passo


def estrazione_sessione_interrotta_dal_ricaricamento(totale_frame):
    """M4 della revisione finale: F5 (e il cambio di progetto) chiudono
    una sessione manuale aperta passando da `apri()` -> `ferma_servizio()`
    senza chiedere -- resta cosi' apposta (un ricaricamento esplicito deve
    poter uscire da ogni stato), ma l'utente merita di sapere perche'
    l'interfaccia e' tornata alla revisione invece di scoprirlo da un
    rettangolo sparito."""
    return "Manual session closed by the reload. %s" % estrazione_stato(totale_frame)


def estrazione_correzione_avviata(n_mancati):
    """Senza questa riga, "Extract and
    fix the misses" faceva cambiare interfaccia sotto le mani dell'utente
    -- da estrazione automatica a sessione manuale -- senza dirlo, la
    lamentela esatta da cui nasce questo ciclo. Consumata una volta sola da
    `_aggiorna_stato_manuale` via `_nota_salvataggio`, come "Saved ..."."""
    if n_mancati == 1:
        return "Extraction done: 1 frame had no face. Fixing it here."
    return "Extraction done: %d frames had no face. Fixing them here." % n_mancati


def estrazione_frame_scelto(nome, n_volti):
    if n_volti == 1:
        return "%s · 1 face" % nome
    return "%s · %d faces" % (nome, n_volti)


ESTRAZIONE_NESSUN_VOLTO = "No face detected on this frame."
ESTRAZIONE_VOLTO_TROVATO = "Face detected."


def estrazione_servizio_guasto(motivo):
    """Distingue un guasto vero (pesi mancanti, memoria esaurita) da
    "nessun volto", che e' la normalita' di questa pagina -- senza questo
    testo i due casi sembrerebbero identici (206 frame senza volti su 983
    nel materiale dell'utente), e un guasto sistemico sparirebbe dentro una
    sessione che sembra solo un video senza facce."""
    return "Detection failed: %s" % motivo


# Quante righe di stderr mostrare nel tooltip del guasto (righe piu' in
# fondo all'anello, le piu' vicine all'errore): un traceback Python tipico
# finisce in poche righe -- l'ultima e' il messaggio dell'eccezione -- e le
# 200 dell'anello intero (TrasportoAsincrono._MAX_RIGHE_STDERR) sarebbero
# un riquadro illeggibile quanto nessun tooltip affatto.
_RIGHE_TOOLTIP_STDERR = 20


def estrazione_servizio_guasto_tooltip(righe_stderr):
    """Il dettaglio del guasto sotto al puntatore della stessa etichetta di
    stato che gia' mostra `estrazione_servizio_guasto` in breve -- niente
    finestra nuova: la console di MainWindow e' indicizzata per Job e il
    servizio di estrazione non e' un Job.

    Vuoto se non c'e' niente da mostrare, cosi' il tooltip sparisce invece
    di restare un riquadro bianco."""
    if not righe_stderr:
        return ""
    ultime = list(righe_stderr)[-_RIGHE_TOOLTIP_STDERR:]
    return "Child process traceback:\n" + "\n".join(ultime)


def estrazione_componi_stato(nota, testo):
    """Una nota che deve sopravvivere al prossimo aggiornamento della riga
    di stato invece di lasciarsene sovrascrivere -- lo stesso principio di
    `estrazione_stato_manuale`, qui per l'avviso di sessione interrotta che
    `mostra_frame` scrive un attimo prima di mostrare il fotogramma
    successivo."""
    return " · ".join(p for p in (nota, testo) if p)


def estrazione_stato_manuale(nota_salvataggio, nome_frame, stato_rilevamento):
    """La riga di stato della sessione manuale
    (gui/estrazione/pagina.py::PaginaEstrazione._aggiorna_stato_manuale):
    la nota di salvataggio, se c'e', precede il nome del fotogramma e lo
    stato del rilevamento -- senza questa composizione un salvataggio
    confermato spariva dallo schermo prima che l'utente potesse leggerlo,
    cancellato dalla riscrittura dello sfogliamento successivo."""
    pezzi = [p for p in (nota_salvataggio, nome_frame, stato_rilevamento) if p]
    return " · ".join(pezzi)


def estrazione_pesi_mancanti_tip(motore):
    """Perche' una voce della tendina rilevatore/allineatore e' disabilitata:
    almeno uno dei file di pesi di questo motore
    (mainscripts.MotoriCatalog.Motore.pesi, una tupla -- "fan-2d" ne porta
    due perche' il face type puo' alzarlo a 3DFAN.npy anche se scelto
    "2DFAN") non e' sotto facelib/ su questa installazione -- una release
    vecchia, o un download mai fatto. Concatenato all'aiuto normale del
    motore, non lo sostituisce: l'utente deve leggere sia cosa fa il
    motore sia perche' non puo' sceglierlo ora."""
    elenco = ", ".join("facelib/%s" % nome for nome in motore.pesi)
    return "%s\n\nUnavailable: weights not found (%s). Update or reinstall to download them." % (
        motore.help, elenco)


def estrazione_motore_tooltip(motore):
    """Il motore che ha prodotto la voce. «not recorded» e non «unknown»:
    il campo manca per una ragione precisa -- la voce e' stata ricostruita
    da `extracttool index`, che non deduce mai un motore, oppure e' piu'
    vecchia del campo -- e «unknown» la faceva leggere come un guasto."""
    return "Engine: %s" % (motore if motore else "not recorded")


ESTRAZIONE_FRAME_NON_NEL_RAPPORTO = "Not in the extraction report: this frame has not been extracted yet."


# -- la colonna dei comandi (gui/estrazione/comandi.py) ----------------------
# `localization/localization.py::StringsDB['S_HOT_KEY']` esiste gia', ma e'
# localizzata (en/ru/zh dal locale di sistema) mentre questo file dichiara in
# testa di scrivere sempre in inglese, qualunque sia il sistema: usarla qui
# romperebbe quell'invariante, quindi la parola resta una costante propria.
HOT_KEY = "hot key"


def estrazione_comando_tip(etichetta, tasto):
    return "%s (%s: %s)" % (etichetta, HOT_KEY, tasto)


def estrazione_comando_etichetta(etichetta, tasto):
    """Il tasto fra parentesi quadre, staccato da due spazi. La tabulazione
    di prima non e' un separatore visibile in un QPushButton: il tasto
    finiva dentro l'etichetta e non si distingueva."""
    return "%s  [%s]" % (etichetta, tasto)


def estrazione_rapporto_piu_vecchio(nome_frame):
    """Il click cade su un rettangolo del rapporto, ma il disco non ha
    piu' nessun volto allineato li' sotto: il rapporto e' rimasto
    indietro rispetto alla cartella `aligned`."""
    return ("No aligned face on disk matches that rectangle in %s. The report "
            "is older than the aligned folder — rebuild it to refresh it."
            % nome_frame)


ESTRAZIONE_NESSUN_VOLTO_SOTTO_IL_PUNTO = (
    "No face here. Click on a face to open it.")
ESTRAZIONE_INDICE_IN_CORSO = (
    "Still indexing the aligned faces — the landmarks will appear when it "
    "is done.")
ESTRAZIONE_VOLTI_IN_ARRIVO = (
    "Loading the faces for this frame — click again in a moment.")


def estrazione_dettaglio_non_risponde(motivo):
    """Il guasto del servizio, detto sul CLICK e non sullo scorrimento: un
    avviso a ogni fotogramma scorso sarebbe rumore, ma un click e' un gesto
    esplicito e merita una risposta."""
    return "Could not read the aligned faces: %s" % motivo


# -- la finestra del volto allineato (gui/dettaglio/) ------------------------
DETTAGLIO_AREE_TITOLO = "Areas"
DETTAGLIO_AREE_TIP = ("Turn an area off to keep its points out of the "
                      "selection. Click the swatch to change its colour.")
DETTAGLIO_AREE_NOMI = {
    "mascella":              "Jaw",
    "sopracciglio-destro":   "Right brow",
    "sopracciglio-sinistro": "Left brow",
    "naso":                  "Nose",
    "occhio-destro":         "Right eye",
    "occhio-sinistro":       "Left eye",
    "bocca":                 "Mouth",
}
DETTAGLIO_COLORE_TITOLO = "Pick a colour for this area"
DETTAGLIO_FRATELLI_TIP = "The other faces extracted from the same frame"
DETTAGLIO_SALVA = "Save"
DETTAGLIO_SALVA_TIP = "Write the edited landmarks back into this face"
DETTAGLIO_REVERT = "Revert"
DETTAGLIO_REVERT_TIP = "Go back to the last saved landmarks"
DETTAGLIO_DISFA = "Undo"
DETTAGLIO_RIFA = "Redo"
DETTAGLIO_ABBANDONA = "This face has unsaved landmark edits. Discard them?"
DETTAGLIO_SOLA_LETTURA_SENZA_FRAME = (
    "Read-only: the source frame is not on disk, so the face cannot be "
    "re-cut from it.")
DETTAGLIO_SOLA_LETTURA_MATRICE = (
    "Read-only: this face has no usable alignment matrix — either it carries "
    "none, or the one it carries cannot be reversed to put the edits back "
    "onto the source frame.")
DETTAGLIO_SOLA_LETTURA_PUNTI = (
    "Read-only: this face does not carry a 68-point landmark set in frame "
    "coordinates. The points drawn here come from the aligned crop, and an "
    "edit would have nowhere to go back to.")
DETTAGLIO_MASCHERA_DEGRADA = (
    "The XSeg mask was re-cut with the face. Hand-drawn polygons keep their "
    "exact shape; the painted mask loses a little sharpness each time.")
DETTAGLIO_RILEVA_LANDMARKS = "Re-detect landmarks"
DETTAGLIO_RILEVA_LANDMARKS_TIP = (
    "Run the chosen landmarker on the face box this image already has. "
    "Nothing is written until you save.")
DETTAGLIO_RILEVA_VOLTO = "Re-detect face"
DETTAGLIO_RILEVA_VOLTO_TIP = (
    "Run the detector on the whole source frame. It may find a different "
    "box, several boxes, or none. Nothing is written until you save.")
DETTAGLIO_MOTORI_IN_MEMORIA = (
    "The detection models stay loaded for five minutes. On a machine that "
    "is also training, that is video memory the training cannot use.")
DETTAGLIO_ATTESA = "Waiting for the face service..."
DETTAGLIO_NESSUNA_PROPOSTA = "No face found in the source frame."
DETTAGLIO_NESSUN_LANDMARK = (
    "The landmarker returned no usable points for the face box this "
    "image carries.")
DETTAGLIO_MOTORI_SPENTI = "Re-detection needs a face that can be edited."
DETTAGLIO_GUASTO_SENZA_MOTIVO = (
    "The face service failed without saying why. The next request restarts "
    "it.")


def dettaglio_titolo(nome, modificata):
    """Il titolo della finestra: il nome del file, con un segno se ci sono
    modifiche non salvate."""
    return ("%s *" % nome) if modificata else nome


def dettaglio_proposte(quante):
    """Quante proposte hanno risposto, quando sono piu' di una: se ne
    applica una al posto di chi guarda, e il numero non si nasconde."""
    return "%d faces found: showing the first." % quante


# Il codice del guasto -> la frase che si legge. Le chiavi vengono dal
# catalogo, non da letterali: un codice ribattezzato da una parte sola
# spegnerebbe la mappa in silenzio, e un guasto tornerebbe a mostrare il
# testo d'implementazione del servizio.
DETTAGLIO_GUASTI = {
    DettaglioGuasti.FILE_ILLEGGIBILE: (
        "This file cannot be read: it is missing, damaged, or not a JPEG."),
    DettaglioGuasti.FRAME_ASSENTE: (
        "The source frame is not on disk, so this face cannot be re-cut "
        "from it."),
    DettaglioGuasti.SENZA_MATRICE: (
        "This face carries no alignment matrix, so edits have no way back "
        "onto the source frame."),
    DettaglioGuasti.SENZA_RETTANGOLO: (
        "This face carries no source face box, so there is nothing to run "
        "the landmarker on."),
    DettaglioGuasti.ALLINEAMENTO_NON_VALIDO: (
        "Those landmarks do not produce a valid alignment. Move them apart "
        "and try again."),
    DettaglioGuasti.SERVIZIO_MUTO: (
        "The face service did not answer in time. Try again: the next "
        "request restarts it."),
    DettaglioGuasti.RISPOSTA_FUORI_SEQUENZA: (
        "A late answer from an earlier request was discarded. Try again."),
}


def dettaglio_guasto(codice, motivo):
    """Il guasto del servizio, detto da qui e non da lui.

    Il servizio parla italiano d'implementazione («non e' un JPEG DFL»), e
    quel testo non e' una frase da leggere a schermo: si mostra la frase
    del catalogo, scelta dal CODICE che il guasto porta con se'. Il testo
    non si guarda mai per capire quale guasto sia -- si romperebbe in
    silenzio alla prima riformulazione.

    Il ripiego e' il motivo grezzo dietro un prefisso, e vale per ogni
    guasto senza codice: un errore di libreria, un `.npy` che manca, il
    servizio morto per inattivita'. Meglio una riga tecnica che un guasto
    sparito. Il prefisso non nomina nessuna operazione di
    proposito: `fallito` e' un segnale condiviso, e ci arrivano anche i
    guasti di richieste che appartengono alla pagina.

    `motivo` puo' essere un'eccezione, una stringa o None -- il client
    emette tutte e tre le forme -- e nessuna deve sollevare qui dentro:
    chi chiama e' uno slot Qt.
    """
    if codice in DETTAGLIO_GUASTI:
        return DETTAGLIO_GUASTI[codice]
    motivo = "" if motivo is None else str(motivo).strip()
    if not motivo:
        return DETTAGLIO_GUASTO_SENZA_MOTIVO
    return "The face service reported: %s" % motivo
