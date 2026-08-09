"""Read-only check of whether a model is locked by a live training process.

Mirrors the read side of models/model_lock.py without importing it: that
module lives under models/, which pulls in torch through ModelBase on
import, exactly what the GUI must never depend on. Only the lock's shape
(a JSON file next to the model, `{pid, start_ticks, ...}`) and the
liveness check are duplicated here; the write side (acquire/release) stays
in models/, and this file never takes or removes a lock, only reads one
someone else may be holding.
"""
import ctypes
import json
import os
import sys
from pathlib import Path

_WIN32_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN32_ERROR_ACCESS_DENIED = 5


def _win_kernel32():
    """Seam for tests: the real kernel32 on Windows, monkeypatched elsewhere.
    Same reasoning (including the explicit argtypes/restype) as
    models/model_lock.py's own copy of this function."""
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
    """Windows analog of the /proc starttime below -- see
    models/model_lock.py's own copy for the full reasoning, duplicated
    rather than imported for the same reason as the rest of this file."""
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
    See models/model_lock.py's own copy for the full reasoning."""
    if sys.platform == "win32":
        return _windows_start_ticks(pid)
    try:
        stat = Path("/proc/%d/stat" % pid).read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _is_alive_windows(pid, recorded_start_ticks):
    """Windows counterpart of the POSIX branch in _is_alive -- see
    models/model_lock.py's own copy for why os.kill(pid, 0) cannot be used
    as a liveness probe on this platform.
    """
    kernel32 = _win_kernel32()
    handle = kernel32.OpenProcess(_WIN32_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means "alive, but not ours to inspect": a lock must
        # not be reported free just because we lack permission to query the
        # process holding it (e.g. started by another user).
        return _win_last_error() == _WIN32_ERROR_ACCESS_DENIED
    kernel32.CloseHandle(handle)
    if recorded_start_ticks is None:
        return True  # never recorded (an old lock file, or the read above failed)
    live_start_ticks = _windows_start_ticks(pid)
    if live_start_ticks is None:
        return True
    return live_start_ticks == recorded_start_ticks


def _is_alive(pid, recorded_start_ticks):
    """Same recognition as models/model_lock.py's own -- see there for why,
    including why PermissionError (a live process owned by someone else)
    must read as alive, not as gone."""
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


def busy_holder(model_dir, model_name):
    """The holder dict of a live lock on `model_name` in `model_dir`, or None.

    `model_name` is the bare name the user chose (before any model-class
    suffix); every '<model_name>_<class>.lock' in `model_dir` is a
    candidate, since the GUI does not know ahead of time which class the
    step will train. A lock whose recorded pid is no longer alive -- the
    same staleness models/model_lock.py's acquire() reclaims -- is not
    reported as busy, so a crashed process never leaves the GUI permanently
    refusing to start.
    """
    model_dir = Path(model_dir)
    try:
        candidates = sorted(model_dir.glob("%s_*.lock" % model_name))
    except OSError:
        return None
    for path in candidates:
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = holder.get("pid")
        if isinstance(pid, int) and _is_alive(pid, holder.get("start_ticks")):
            return holder
    return None
