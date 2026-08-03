"""Lettura di scripts/commands.toml.

commands.toml e' la fonte di verita' dei passi utente: da esso si generano sia
i .bat per Windows sia i .sh per Linux (setup/gen_scripts.py). Descriverli una
volta sola e' cio' che impedisce alle due piattaforme di divergere.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Le quattro scorciatoie che layout.py depone nella root dell'installazione.
# Non duplicano lo script: lo richiamano.
SHORTCUTS = (
    "4) data_src faceset extract",
    "5) data_dst faceset extract",
    "6) train SAEHD",
    "7) merge SAEHD",
)

# I valori ammessi per Command.pause.
PAUSE_ALWAYS = "always"      # `pause` in fondo, sempre
PAUSE_ON_ERROR = "on_error"  # solo se il comando e' uscito con codice != 0
PAUSE_NEVER = "never"        # nessun pause: lo script lancia una GUI e ritorna

# I valori ammessi per Command.kind.
KIND_MAIN = "main"      # una o piu' invocazioni di main.py
KIND_VIEWER = "viewer"  # apre una cartella nel visualizzatore di sistema
KIND_EBSYNTH = "ebsynth"  # lancia EbSynth (solo Windows)
KIND_CLEAR = "clear"    # ricrea la struttura di workspace/ svuotandola


@dataclass(frozen=True)
class Invocation:
    """Una singola chiamata a main.py."""
    verb: tuple[str, ...]   # es. ("videoed", "extract-video") oppure ("train",)
    args: tuple[str, ...]   # gli argomenti gia' in ordine, con i segnaposto


@dataclass(frozen=True)
class Command:
    """Un passo utente: un file .bat e un file .sh."""
    name: str                        # "6) train SAEHD", senza estensione
    kind: str                        # una delle costanti KIND_*
    invocations: tuple[Invocation, ...]
    mkdirs: tuple[str, ...]          # cartelle da creare prima, con i segnaposto
    pause: str                       # una delle costanti PAUSE_*
    pause_before: bool               # `echo` + pause PRIMA di eseguire: conferma
                                      # per un'operazione distruttiva (es. kind=clear).
                                      # Senza, l'utente non ha modo di annullare.
    passthrough: bool                # lo script inoltra il primo argomento (%1 / "$1")
    target: str                      # per KIND_VIEWER: la cartella da aprire
    echo: tuple[str, ...]            # righe da stampare prima di eseguire


def load_commands(path: Path) -> list[Command]:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    commands = [
        Command(
            name=entry["name"],
            kind=entry["kind"],
            invocations=tuple(
                Invocation(verb=tuple(inv["verb"]), args=tuple(inv.get("args", ())))
                for inv in entry.get("invocation", ())
            ),
            mkdirs=tuple(entry.get("mkdirs", ())),
            pause=entry.get("pause", PAUSE_ALWAYS),
            pause_before=entry.get("pause_before", False),
            passthrough=entry.get("passthrough", False),
            target=entry.get("target", ""),
            echo=tuple(entry.get("echo", ())),
        )
        for entry in raw["command"]
    ]
    _validate(commands)
    return commands


def _validate(commands: list[Command]) -> None:
    names = [c.name for c in commands]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"nomi duplicati in commands.toml: {sorted(duplicates)}")
    for c in commands:
        if c.kind not in (KIND_MAIN, KIND_VIEWER, KIND_EBSYNTH, KIND_CLEAR):
            raise ValueError(f"{c.name!r}: kind sconosciuto {c.kind!r}")
        if c.pause not in (PAUSE_ALWAYS, PAUSE_ON_ERROR, PAUSE_NEVER):
            raise ValueError(f"{c.name!r}: pause sconosciuto {c.pause!r}")
        if c.kind == KIND_MAIN and not c.invocations:
            raise ValueError(f"{c.name!r}: kind=main senza invocazioni")
        if c.kind == KIND_VIEWER and not c.target:
            raise ValueError(f"{c.name!r}: kind=viewer senza target")
        if c.pause_before and not c.echo:
            raise ValueError(f"{c.name!r}: pause_before=True senza echo (conferma senza messaggio)")
        if c.kind == KIND_CLEAR and not c.pause_before:
            raise ValueError(f"{c.name!r}: kind=clear senza conferma pre-esecuzione (pause_before)")
    missing = set(SHORTCUTS) - set(names)
    if missing:
        raise ValueError(f"scorciatoie senza comando corrispondente: {sorted(missing)}")
