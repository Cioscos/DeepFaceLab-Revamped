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

from gui.execution.conflicts import conflict

_KILL_GRACE_MS = 5000


class StepConflict(Exception):
    def __init__(self, artifact, running_step_name):
        super().__init__("'%s' is busy: %s is using it" % (artifact, running_step_name))
        self.artifact = artifact
        self.running_step_name = running_step_name


def _resolve(text, workspace, dfl_root):
    return (text.replace("{WORKSPACE}", str(workspace))
                .replace("{DFL_ROOT}", str(dfl_root))
                .replace("{INTERNAL}", str(dfl_root.parent)))


class Job(QObject):
    output = pyqtSignal(str)      # one merged stdout/stderr line
    finished = pyqtSignal(int)    # overall exit code (0 = every invocation ok)

    def __init__(self, step, workdir, events_path, python_exe, dfl_root,
                 invocation_args, env, parent=None):
        super().__init__(parent)
        self.step = step
        self.workdir = workdir
        self.events_path = events_path
        self.running = True
        self.captured_lines = []
        self._python_exe = python_exe
        self._dfl_root = dfl_root
        self._invocation_args = invocation_args  # list, one resolved-args list per invocation
        self._env = env
        self._index = 0
        self.process = None
        self._current_program = None
        self._start_invocation(0)

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
        for line in text.splitlines():
            self.captured_lines.append(line)
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
        self.captured_lines.append(line)
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

        workdir = Path(tempfile.mkdtemp(prefix="dfl-gui-"))
        answers_path = workdir / "answers.json"
        answers_path.write_text(json.dumps(answers), encoding="utf-8")
        events_path = workdir / "events.jsonl"

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

        job = Job(step, workdir, events_path, self._python_exe, self._dfl_root,
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
