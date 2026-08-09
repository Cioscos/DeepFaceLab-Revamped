"""Job execution: subprocesses, invocation sequences, conflicts, tree stop.

A Job runs a step's invocations one after another, in the temporary
directory created for it, and reports merged stdout/stderr line by line.
The JobManager arbitrates which jobs may run together, using the conflict
matrix, and knows how to kill a job's entire process tree -- not just the
process it started directly -- because some steps spawn worker processes
of their own.
"""
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from gui.console_buffer import ConsoleBuffer
from gui.console_stream import LineAssembler
from gui.execution.conflicts import conflict
from gui.model_lock_status import busy_holder

_KILL_GRACE_MS = 5000


class StepConflict(Exception):
    def __init__(self, artifact, running_step_name):
        super().__init__("'%s' is busy: %s is using it" % (artifact, running_step_name))
        self.artifact = artifact
        self.running_step_name = running_step_name


def _model_name_from_args(extra_args):
    """The value that follows '--force-model-name' in extra_args, or None.

    This is how a needs_model_name step's chosen model name reaches
    try_start today (see main_window.py's start handler) -- reading it back
    out here avoids widening try_start's own signature just for the lock
    check.
    """
    args = list(extra_args)
    try:
        i = args.index("--force-model-name")
    except ValueError:
        return None
    return args[i + 1] if i + 1 < len(args) else None


_TRAIN_VERB = ("train",)


def _is_training_step(step):
    """True when running this step invokes `main.py train`.

    That is the only verb that ever takes models/model_lock.py's lock (see
    ModelBase.__init__, gated on is_training) -- merging and exporting carry
    needs_model_name too but never touch the lock, on the CLI or from here.
    Checked on the invocation's own verb, not on needs_model_name or
    family, because XSeg's training step ("5.XSeg) train") has neither: it
    trains a fixed model name the user never chooses (Model_XSeg/Model.py
    passes force_model_class_name='XSeg' to ModelBase, which then never
    prompts for a name), so it carries no --force-model-name at all.
    """
    return any(inv.verb == _TRAIN_VERB for inv in step.invocations)


def _model_class_from_step(step):
    """The value following '--model' in the step's training invocation, or None.

    Used only to name the busy model in StepConflict's message when there
    is no user-chosen bare name to show it instead (XSeg's fixed-name
    training step, see _is_training_step).
    """
    for inv in step.invocations:
        if inv.verb != _TRAIN_VERB:
            continue
        args = list(inv.args)
        try:
            i = args.index("--model")
        except ValueError:
            return None
        return args[i + 1] if i + 1 < len(args) else None
    return None


def _resolve(text, workspace, dfl_root):
    return (text.replace("{WORKSPACE}", str(workspace))
                .replace("{DFL_ROOT}", str(dfl_root))
                .replace("{INTERNAL}", str(dfl_root.parent)))


class Job(QObject):
    output = pyqtSignal(str)         # a new merged stdout/stderr line
    output_update = pyqtSignal(str)  # the last line, rewritten in place
    finished = pyqtSignal(int)       # overall exit code (0 = every invocation ok)

    def __init__(self, step, workdir, events_path, commands_path, python_exe, dfl_root,
                 invocation_args, env, parent=None):
        super().__init__(parent)
        self.step = step
        self.workdir = workdir
        self.events_path = events_path
        self.commands_path = commands_path
        self.running = True
        self.buffer = ConsoleBuffer()
        # One assembler for the whole job, not one per invocation: a step
        # with several invocations is one console, and a line cut by the end
        # of a read is reassembled the same way wherever it came from.
        self._assembler = LineAssembler()
        self._python_exe = python_exe
        self._dfl_root = dfl_root
        self._invocation_args = invocation_args  # list, one resolved-args list per invocation
        self._env = env
        self._index = 0
        self.process = None
        self._current_program = None
        self._start_invocation(0)

    @property
    def captured_lines(self):
        """The buffered output, oldest first. A real list: callers slice it."""
        return self.buffer.lines()

    def _command(self, args):
        verb = list(self.step.invocations[self._index].verb)
        program_args = [str(self._dfl_root / "main.py")] + verb + args
        if sys.platform == "win32":
            return self._python_exe, program_args
        # POSIX: start the child as its own session leader so stop() can
        # kill the whole process tree, not just this one process.
        setsid_src = "import os,sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])"
        wrapper_args = ["-c", setsid_src, self._python_exe] + program_args
        return self._python_exe, wrapper_args

    def _start_invocation(self, index):
        self._index = index
        program, args = self._command(self._invocation_args[index])
        self._current_program = program
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(str(self._dfl_root))
        self.process.setProcessEnvironment(self._env)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._on_invocation_finished)
        self.process.errorOccurred.connect(self._on_error_occurred)
        self.process.start(program, args)

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        for kind, line in self._assembler.feed(text):
            if kind == "update":
                self.buffer.replace_last(line)
                self.output_update.emit(line)
            else:
                self.buffer.append(line)
                self.output.emit(line)

    def _on_invocation_finished(self, code, status):
        # A killed invocation (SIGKILL, a crash) can still report exit code
        # 0 on some platforms/paths -- QProcess.NormalExit is the only
        # signal that the process actually ran to completion under its own
        # control, so the sequence advances on both, not on code alone.
        if code == 0 and status == QProcess.NormalExit and self._index + 1 < len(self.step.invocations):
            self._start_invocation(self._index + 1)
            return
        self._finish(code)

    def _on_error_occurred(self, error):
        # A process that never started (bad executable, missing file) never
        # emits `finished` -- left unhandled, the job would stay `running`
        # forever, jamming try_start's conflict check for every future step
        # touching the same artifacts. Report it as a distinguishable
        # failure (-1) instead.
        if error != QProcess.FailedToStart:
            return
        line = "failed to start: %s" % self._current_program
        self.buffer.append(line)
        self.output.emit(line)
        self._finish(-1)

    def _finish(self, code):
        # Idempotent: guards against errorOccurred and finished both firing
        # for the same invocation, which would otherwise emit `finished`
        # twice for one job.
        if not self.running:
            return
        self.running = False
        self.finished.emit(code)

    def stop(self):
        if not self.running or self.process is None:
            return
        pid = self.process.processId()
        if pid == 0:
            # No live process to signal -- notably, this guards against
            # os.killpg(0, SIGTERM), which under POSIX semantics targets the
            # CALLER's own process group (this GUI process), not a no-op.
            return
        if sys.platform == "win32":
            QProcess.startDetached("taskkill", ["/PID", str(pid), "/T", "/F"])
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        def _escalate():
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        QTimer.singleShot(_KILL_GRACE_MS, _escalate)

    def send_command(self, op):
        """Append one command for the child to read. Never blocks, never raises.

        The child polls this file; a command written after it exited is
        simply never read, which is why a failure to write is not worth
        reporting to the user.
        """
        try:
            with open(self.commands_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"op": op}) + "\n")
        except OSError:
            pass


class JobManager(QObject):
    job_started = pyqtSignal(object)          # Job
    job_finished = pyqtSignal(object, int)    # Job, exit code

    def __init__(self, python_exe: str, dfl_root, parent=None):
        super().__init__(parent)
        self._python_exe = python_exe
        self._dfl_root = Path(dfl_root)
        self._jobs = []

    def try_start(self, step, answers: dict, workspace, extra_args=()) -> Job:
        for job in self.active_jobs():
            artifact = conflict(step, job.step)
            if artifact is not None:
                raise StepConflict(artifact, job.step.name)

        if _is_training_step(step):
            # Only a training step actually contends for the lock -- merging
            # and exporting carry needs_model_name too but are readers,
            # never refused by it even from a plain CLI terminal
            # (models/model_lock.py), so checking them here too would make
            # the GUI stricter than the CLI for no reason. A live lock
            # belongs to a process this JobManager did not start -- another
            # GUI, or a CLI terminal -- so it cannot show up in
            # active_jobs() above; reading the lock file is the only way to
            # catch it before wasting a subprocess launch on a training run
            # that would fail anyway.
            #
            # bare_name is the user-chosen part of the model name
            # (--force-model-name, when the step prompts for one at all) --
            # '' for XSeg's training step, the one training step with no
            # such prompt, matching the same '' ModelBase.__init__ uses for
            # that branch (see models/ModelBase.py).
            bare_name = _model_name_from_args(extra_args) if step.needs_model_name else ""
            if bare_name or not step.needs_model_name:
                holder = busy_holder(Path(workspace) / "model", bare_name)
                if holder is not None:
                    label = bare_name or _model_class_from_step(step) or step.name
                    raise StepConflict(
                        label, "another process (pid %s)" % holder.get("pid"))

        workdir = Path(tempfile.mkdtemp(prefix="dfl-gui-"))
        answers_path = workdir / "answers.json"
        answers_path.write_text(json.dumps(answers), encoding="utf-8")
        events_path = workdir / "events.jsonl"
        commands_path = workdir / "commands.jsonl"

        for pattern in step.mkdirs:
            Path(_resolve(pattern, workspace, self._dfl_root)).mkdir(
                parents=True, exist_ok=True)

        invocation_args = []
        for invocation in step.invocations:
            resolved = [_resolve(a, workspace, self._dfl_root) for a in invocation.args]
            resolved.extend(extra_args)
            invocation_args.append(resolved)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("WORKSPACE", str(workspace))
        env.insert("DFL_ANSWERS_FILE", str(answers_path))
        env.insert("DFL_EVENTS_FILE", str(events_path))
        env.insert("DFL_COMMANDS_FILE", str(commands_path))

        job = Job(step, workdir, events_path, commands_path, self._python_exe, self._dfl_root,
                  invocation_args, env, parent=self)
        job.finished.connect(lambda code, job=job: self._on_job_finished(job, code))
        self._jobs.append(job)
        self.job_started.emit(job)
        return job

    def _on_job_finished(self, job, code):
        self.job_finished.emit(job, code)

    def active_jobs(self) -> list:
        return [job for job in self._jobs if job.running]

    def stop(self, job):
        job.stop()

    def stop_all(self):
        for job in self.active_jobs():
            self.stop(job)
