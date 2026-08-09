"""The shell's dark theme.

The colours are the mask editor's own (`core/qtex/qtex.py::QDarkPalette`),
duplicated here on purpose: the `gui` package imports nothing from the
application, not even its Qt helpers. A guard compares the two palettes
role by role, so the duplication cannot drift.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette

TEXT = QColor(200, 200, 200)
WINDOW = QColor(53, 53, 53)
BASE = QColor(25, 25, 25)
ACCENT = QColor(42, 130, 218)

CONSOLE_FONT_POINT_SIZE = 9


def dark_palette():
    """The palette applied to the whole application."""
    p = QPalette()
    p.setColor(QPalette.Window, WINDOW)
    p.setColor(QPalette.WindowText, TEXT)
    p.setColor(QPalette.Base, BASE)
    p.setColor(QPalette.AlternateBase, WINDOW)
    p.setColor(QPalette.ToolTipBase, TEXT)
    p.setColor(QPalette.ToolTipText, TEXT)
    p.setColor(QPalette.Text, TEXT)
    p.setColor(QPalette.Button, WINDOW)
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.BrightText, Qt.red)
    p.setColor(QPalette.Link, ACCENT)
    p.setColor(QPalette.Highlight, ACCENT)
    p.setColor(QPalette.HighlightedText, Qt.black)
    return p


def apply_dark_theme(app):
    """Apply style and palette. Same pair the mask editor applies."""
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
