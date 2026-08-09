"""Application bootstrap for the PyQt5 shell."""
import os
import sys
from pathlib import Path


def _sanitize_qt_plugin_env(environ, cv2_package_dir=None):
    """Drop Qt plugin path variables that point inside cv2's own Qt build.

    `main.py` imports `core.interact.interact` (and, through it, `cv2`)
    before any verb runs, `gui` included. opencv-python's Linux wheel points
    QT_QPA_PLATFORM_PLUGIN_PATH (and, on some builds, QT_PLUGIN_PATH) at its
    own bundled Qt plugins as a side effect of that import. PyQt5's
    QApplication then tries to load cv2's "xcb" plugin instead of its own
    and never starts. Only entries that resolve inside cv2's package
    directory are removed here, so a value some other component legitimately
    set is left alone.

    `environ` is mutated in place and returned -- a plain dict works in
    tests without touching the real process environment. `cv2_package_dir`
    defaults to the real cv2 install (skipped entirely if cv2 isn't
    importable) but can be overridden, which is how tests exercise this
    without depending on where cv2 actually happens to be installed.
    """
    if cv2_package_dir is None:
        try:
            import cv2
        except ImportError:
            return environ
        cv2_package_dir = Path(cv2.__file__).resolve().parent
    cv2_package_dir = Path(cv2_package_dir).resolve()

    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        value = environ.get(var)
        if not value:
            continue
        try:
            resolved = Path(value).resolve()
        except (OSError, ValueError):
            continue
        if resolved == cv2_package_dir or str(resolved).startswith(str(cv2_package_dir) + os.sep):
            del environ[var]
    return environ


def _warn_about_missing_xcb_libraries():
    """Print a warning if the system libraries the "xcb" platform plugin
    needs are missing, without stopping the application from starting.

    The soname-to-package mapping lives in `setup/prerequisiti_linux.py`
    (a single source, shared with the installer, which runs the same check
    right after installing PyQt5) rather than duplicated here. Imported
    lazily, inside this function, so importing `gui.app` on Windows -- or
    anywhere before a venv exists -- never pulls in a module built around a
    Linux-only concept. Anything going wrong here (module missing, `ldd`
    behaving unexpectedly) is swallowed: the user might still have a
    working Qt backend this check does not know about (wayland, offscreen),
    and a broken diagnostic must never block a GUI that would otherwise
    start.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import sysconfig

        from setup.prerequisiti_linux import diagnosi
        site_packages = Path(sysconfig.get_paths()["purelib"])
        message = diagnosi(site_packages)
    except Exception:
        return
    if message is not None:
        print(message)


def run(argv=None):
    _warn_about_missing_xcb_libraries()
    _sanitize_qt_plugin_env(os.environ)

    from PyQt5.QtWidgets import QApplication

    from gui.main_window import MainWindow
    from gui.theme import apply_dark_theme

    app = QApplication(argv if argv is not None else sys.argv)
    apply_dark_theme(app)
    dfl_root = Path(__file__).resolve().parent.parent
    window = MainWindow(sys.executable, dfl_root)
    window.show()
    return app.exec_()
