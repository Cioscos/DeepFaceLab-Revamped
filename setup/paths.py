"""Dove sta ogni cosa in un'installazione.

Un solo posto in cui i percorsi si costruiscono. Ogni modulo di setup/ riceve
un InstallPaths e non concatena percorsi per conto proprio: e' cio' che
garantisce che l'installer non scriva mai fuori dalla cartella scelta.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallPaths:
    root: Path
    internal: Path
    repo: Path
    venv: Path
    python: Path
    scripts_dir: Path
    workspace: Path
    log_file: Path
    uv_bin: Path


def resolve(root: Path) -> InstallPaths:
    root = Path(root).resolve()
    internal = root / "_internal"
    venv = internal / ".venv"
    on_windows = sys.platform == "win32"

    # uv_bin di norma vive sotto la root di installazione (internal/uv/uv) --
    # il caso comune, dove --dest coincide con la cartella di install.sh/
    # install.bat. Ma install.sh/install.bat scaricano il
    # binario di uv PRIMA di sapere quale --dest l'utente passera' (bash/
    # cmd.exe girano prima di qualunque parsing Python): lo mettono sempre
    # accanto a se stessi, in "<cartella dello script>/_internal/uv". Se
    # l'utente passa un --dest esplicito diverso da quella cartella (caso
    # supportato), le due sedi divergono e
    # calcolare uv_bin da `root` (cioe' da --dest) punterebbe a un binario
    # che non esiste mai li' -- riprodotto per davvero lanciando `install.sh
    # --dest <altrove>` ("[Errno 2] No such file or directory:
    # '<dest>/_internal/uv/uv'"). install.sh/install.bat esportano
    # percio' DFL_UV_BIN con la sede vera prima di invocare "uv run"; se la
    # variabile manca (uso diretto di setup/__main__.py, o test), si ricade
    # sul calcolo relativo a root, che e' l'unico dato disponibile e resta
    # corretto nel caso comune (--dest assente o uguale alla cartella dello
    # script).
    uv_bin_override = os.environ.get("DFL_UV_BIN")
    uv_bin = Path(uv_bin_override) if uv_bin_override else internal / "uv" / ("uv.exe" if on_windows else "uv")

    return InstallPaths(
        root=root,
        internal=internal,
        repo=internal / "DeepFaceLab",
        venv=venv,
        python=venv / ("Scripts/python.exe" if on_windows else "bin/python"),
        scripts_dir=root / "scripts",
        # WORKSPACE e' fratello di _internal, non figlio: e' cosi' che
        # setenv.bat lo esporta (%INTERNAL%\..\workspace).
        workspace=root / "workspace",
        log_file=internal / "_e" / "install.log",
        uv_bin=uv_bin,
    )
