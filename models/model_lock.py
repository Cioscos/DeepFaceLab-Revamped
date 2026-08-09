"""One writer at a time on a model's files.

Nothing used to stop two processes from training the same model: two
terminals, or a graphical launcher beside a terminal, both open the same
directory and both save into it, and the last save wins. The lock lives
here, next to the model, rather than in whoever launches the run, so it
protects every caller equally.

Only a writer takes it. Opening a model to read its weights -- merging,
exporting -- disturbs nobody and is never refused.
"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

_WIN32_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN32_ERROR_ACCESS_DENIED = 5


class ModelInUse(Exception):
    pass


def _win_kernel32():
    """Seam for tests: the real kernel32 on Windows, monkeypatched elsewhere.

    `ctypes.WinDLL` does not exist off Windows, so calling it directly would
    make the Windows branches below untestable from Linux. Every Windows
    call goes through this one function so a test can replace it with a
    fake exposing OpenProcess/CloseHandle/GetProcessTimes, without needing
    a real Windows process to point at.

    Explicit argtypes/restype matter here, not just style: left to ctypes'
    default int marshalling, a HANDLE -- 64 bits wide on 64-bit Windows --
    would come back through OpenProcess truncated to a C int, silently
    corrupting a handle whose numeric value happens to exceed 32 bits.
    """
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    return kernel32


def _win_last_error():
    """Seam for tests, same reasoning as _win_kernel32()."""
    return ctypes.get_last_error()


def _windows_start_ticks(pid):
    """Windows analog of the /proc starttime below: `pid`'s creation time.

    None when a handle cannot be opened at all, or when GetProcessTimes
    itself fails -- in which case the reused-pid defence quietly falls back
    to "the pid exists" alone (see _is_alive_windows). That degradation is
    documented, not silent: GetProcessTimes' CreationTime is set once by the
    kernel when the process is created and never repeated by a different
    process reusing the same pid number, the same role /proc/pid/stat's
    starttime plays on Linux, so losing it is a real loss of precision, not
    a formality.
    """
    kernel32 = _win_kernel32()
    handle = kernel32.OpenProcess(_WIN32_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        from ctypes import wintypes
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(handle, ctypes.pointer(creation), ctypes.pointer(exit_time),
                                       ctypes.pointer(kernel_time), ctypes.pointer(user_time))
        if not ok:
            return None
        return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def _start_ticks(pid):
    """The kernel-assigned start time of `pid`, or None when it cannot be read.

    On Linux, field 22 of /proc/pid/stat -- clock ticks since boot, set once
    when the kernel allocates the pid and never repeated for another live
    process. Unlike the command line (which a spawner is free to reuse,
    rewrite, or -- as pytest-xdist's own worker bootstrap shows -- never
    expose in a form related to what the running code passed as sys.argv),
    this cannot coincidentally match between two different processes that
    happen to share a recycled pid number. On Windows, _windows_start_ticks
    plays the same role via GetProcessTimes' CreationTime.
    """
    if sys.platform == "win32":
        return _windows_start_ticks(pid)
    try:
        stat = Path("/proc/%d/stat" % pid).read_text()
        # comm (field 2) is parenthesized and may itself contain ')' or
        # spaces, so split past its last ')' rather than just whitespace.
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _is_alive_windows(pid, recorded_start_ticks):
    """Windows counterpart of the POSIX branch in _is_alive.

    `os.kill(pid, 0)` is not a liveness probe on this platform: signal 0
    collides with CTRL_C_EVENT, so calling it on another process's pid
    either delivers a real Ctrl+C (if the OS lets the call through) or fails
    with ERROR_INVALID_PARAMETER, which the POSIX code would misread as
    "not alive" -- reporting a live trainer as dead and letting a second
    process take its lock. OpenProcess with a query-only access right never
    signals anything, which is the whole reason to use it here instead.
    """
    kernel32 = _win_kernel32()
    handle = kernel32.OpenProcess(_WIN32_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means "alive, but not ours to inspect" -- a lock
        # must not be stolen from a process we merely lack permission to
        # query (e.g. started by another user). Any other failure (an
        # invalid pid, no such process) means gone.
        return _win_last_error() == _WIN32_ERROR_ACCESS_DENIED
    kernel32.CloseHandle(handle)
    if recorded_start_ticks is None:
        return True  # never recorded (an old lock file, or the read above failed)
    live_start_ticks = _windows_start_ticks(pid)
    if live_start_ticks is None:
        return True
    return live_start_ticks == recorded_start_ticks


def _is_alive(pid, recorded_start_ticks):
    """True when `pid` names a live process that is plausibly the same one.

    A bare liveness probe succeeding only proves *some* process owns that
    number now, not that it is the one that wrote the lock -- pids get
    reused. Comparing the live start time against the one recorded at
    acquire time closes that gap, on both platforms this function branches
    on. The pid check is all there is only when neither side can supply a
    start time (see _start_ticks/_windows_start_ticks), and the message
    tells the user how to clear a lock by hand.

    `PermissionError` is an `OSError` subclass and must not fall into the
    same bucket as `ProcessLookupError`: the kernel raises it for a pid that
    exists but is owned by another user, which os.kill(pid, 0) cannot
    signal either way -- the same "alive, not ours to inspect" case the
    Windows branch above already treats as alive via ACCESS_DENIED. Reading
    it as dead would let a second process steal the lock from a live
    trainer it merely lacks permission to signal.
    """
    if sys.platform == "win32":
        return _is_alive_windows(pid, recorded_start_ticks)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    if recorded_start_ticks is None:
        return True  # never recorded (not on Linux, or an old lock file)
    live_start_ticks = _start_ticks(pid)
    if live_start_ticks is None:
        return True
    return live_start_ticks == recorded_start_ticks


class ModelLock(object):
    def __init__(self, model_dir, model_name, model_class):
        self.path = Path(model_dir) / ("%s_%s.lock" % (model_name, model_class))

    def holder(self):
        """The recorded holder, or None when the lock is absent or unreadable."""
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def acquire(self):
        """Take the lock, or raise ModelInUse naming who holds it.

        No reentrancy: a live holder blocks a new acquire even when the
        recorded pid happens to be this very process's, because nothing in
        this codebase legitimately acquires the same lock twice -- treating
        "it's me" as a free pass would just be a second, quieter way for a
        bug to corrupt the same files this exists to protect.
        """
        holder = self.holder()
        if holder is not None:
            pid = holder.get("pid")
            if isinstance(pid, int) and _is_alive(pid, holder.get("start_ticks")):
                raise ModelInUse(
                    "the model is already open for training by process %d "
                    "(started %s, command: %s). Close it, or delete %s if you "
                    "are sure nothing is running."
                    % (pid, time.ctime(holder.get("started", 0)),
                       holder.get("command", "?"), self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "pid": os.getpid(),
            "started": time.time(),
            "command": " ".join(sys.argv),
            "start_ticks": _start_ticks(os.getpid()),
        }), encoding="utf-8")

    def release(self):
        """Drop the lock. Never raises: a lock already gone is fine."""
        try:
            self.path.unlink()
        except OSError:
            pass
