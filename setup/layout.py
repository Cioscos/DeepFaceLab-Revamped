"""La posa dell'installazione: script utente, scorciatoie, workspace/.

Quattro funzioni, ciascuna idempotente per se stessa (rilanciare `install` e'
l'aggiornamento):

- `place_scripts` rigenera i 55 .bat/.sh in `scripts/` da `commands.toml`,
  svuotando prima solo i file dell'estensione del sistema corrente (cosi' un
  passo tolto dal TOML sparisce davvero, invece di restare a meta' funzionante
  sul disco dell'utente).
- `install_setenv` copia `setenv.bat`/`setenv.sh` dal repo clonato a
  `_internal/`, dove ogni script generato da `place_scripts` e ogni
  scorciatoia da `write_shortcuts` si aspetta di trovarlo (`gen_scripts.py`:
  `call "%~dp0..\\_internal\\setenv.bat"` / `source
  "$(dirname "$0")/../_internal/setenv.sh"'`). Senza, i 55+4 script generati
  falliscono tutti alla primissima riga eseguibile con "No such file or
  directory" -- verificato per davvero lanciandone uno dopo un giro reale di
  place_scripts/write_shortcuts. Non basta che la riga "source
  .../setenv.sh" compaia nel testo generato: il file a cui punta deve
  esistere davvero sul disco.
- `write_shortcuts` scrive in root un solo avvio, quello della GUI: richiama
  lo script in `scripts/`, non ne duplica il contenuto -- due
  copie divergerebbero al primo cambio di `commands.toml`. Rimuove anche, se
  le trova, le quattro scorciatoie per-passo di una versione precedente
  dell'installer (extract src, extract dst, train SAEHD, merge SAEHD): un
  utente che aggiorna non deve ritrovarsele in root a fianco della GUI. La
  rimozione e' prudente -- vedi `LEGACY_SHORTCUTS` e il docstring della
  funzione.
- `create_workspace` crea la struttura di `workspace/` con `mkdir(parents=True,
  exist_ok=True)` e nient'altro: nessun `rmdir`, nessuna scrittura di file.
  E' l'unica funzione di questo file che tocca `workspace/`, ed e' scritta
  per non toccare mai cio' che gia' c'e' -- ci stanno i volti estratti e i
  modelli addestrati dell'utente.
"""
from __future__ import annotations

import sys
from pathlib import Path

from setup.commands import SHORTCUTS, load_commands
from setup.gen_scripts import generate
from setup.paths import InstallPaths

# Le quattro scorciatoie per-passo che una versione precedente dell'installer
# depositava in root, ritirate a favore dell'unica scorciatoia della GUI
# (SHORTCUTS, sopra). Restano qui, separate da SHORTCUTS, solo perche'
# write_shortcuts deve saperle riconoscere per rimuoverle da
# un'installazione preesistente -- non sono piu' "scorciatoie" nel senso di
# commands.py (nessun comando le referenzia, _validate non le vede), sono
# rifiuti di una versione precedente di questo stesso file.
LEGACY_SHORTCUTS = (
    "4) data_src faceset extract",
    "5) data_dst faceset extract",
    "6) train SAEHD",
    "7) merge SAEHD",
)

# La struttura vuota di workspace/: niente video di esempio (non si
# distribuiscono piu').
WORKSPACE_TREE = ("data_src/aligned", "data_dst/aligned", "model")


def _system() -> str:
    return "win" if sys.platform == "win32" else "linux"


def place_scripts(paths: InstallPaths, log) -> list[Path]:
    """Rigenera `scripts/` da commands.toml. Va rilanciata a ogni aggiornamento.

    Svuota prima ogni file con l'estensione del sistema corrente (generato o
    no: `glob(f"*{suffix}")` non distingue un .bat/.sh scritto da questa
    funzione in un giro precedente da un file dell'utente con lo stesso
    suffisso -- entrambi vengono cancellati), cosi' un passo rimosso dal TOML
    non sopravvive al giro successivo. Un eventuale file dell'utente con
    l'estensione DELL'ALTRO sistema resta intatto, perche' non viene mai
    guardato; uno con l'estensione corrente no: `scripts/` e' territorio
    generato, non un posto dove depositare script propri.

    Eccezione: `setenv.bat`/`setenv.sh` non si toccano. Non sono territorio
    generato ma file del repository (gen_scripts non li genera, commands.toml
    non li nomina), e la procedura documentata installa dentro il clone del
    repository stesso -- li' `paths.scripts_dir` E' la directory versionata
    che li contiene, e cancellarli sporca il `git status` dell'utente con
    ' D scripts/setenv.bat' senza spiegazione.

    La fonte e' sempre `paths.repo/scripts/commands.toml`, senza ricadute:
    nel flusso wired (`setup/__main__.py`, `_STEPS`) `step_sync_repo` gira
    sempre prima di `step_layout`, e `sync_repo`/`_check_installer_layout`
    (setup/repo.py) garantiscono che questo file esista dopo ogni clone o
    pull riuscito. Una ricaduta silenziosa su un'altra copia (es. quella
    bundlata insieme a questo stesso `setup/`) genererebbe script da una
    fonte diversa da quella appena clonata, senza dirlo -- esattamente il
    modo in cui un utente si ritroverebbe script che non corrispondono al
    codice che ha. Se il file manca, l'errore nomina il percorso esatto in
    cui si e' cercato, cosi' la causa (un `sync_repo` mai girato, o girato
    su un clone con modifiche locali che ha saltato il pull) e' ovvia subito
    invece che tre passi dopo.
    """
    system = _system()
    suffix = ".bat" if system == "win" else ".sh"

    for stale in paths.scripts_dir.glob(f"*{suffix}"):
        if stale.name in ("setenv.bat", "setenv.sh"):
            continue
        stale.unlink()

    commands_toml = paths.repo / "scripts" / "commands.toml"
    if not commands_toml.is_file():
        raise RuntimeError(
            f"commands.toml non trovato in {commands_toml}: step_sync_repo deve "
            "clonare o aggiornare paths.repo prima che step_layout giri. "
            "Verifica che 'install' abbia eseguito il passo 'repo' senza "
            "errori, oppure che il clone non abbia modifiche locali che hanno "
            "fatto saltare il pull (setup/repo.py::sync_repo)."
        )
    commands = load_commands(commands_toml)
    written = generate(commands, paths.scripts_dir, system)

    if log is not None:
        log.info("scritti %d script (%s) in %s", len(written), suffix, paths.scripts_dir)
    return written


def install_setenv(paths: InstallPaths, log) -> Path:
    """Copia `scripts/setenv.bat` o `setenv.sh` dal repo clonato a
    `_internal/`, del sistema corrente soltanto (stesso principio di
    `place_scripts`: un file dell'altro sistema accanto a quello buono non
    serve a nessuno).

    La fonte e' sempre `paths.repo/scripts/setenv.*`, mai un'altra copia
    (stesso motivo di `place_scripts`, vedi il suo docstring): `step_sync_repo`
    gira sempre prima di `step_layout` nel flusso wired, quindi il file esiste
    dopo ogni clone o pull riusciti. Se manca, l'errore nomina il percorso
    esatto invece di lasciare che ogni script generato fallisca dopo, con un
    "No such file or directory" che non dice perche'.
    """
    system = _system()
    name = "setenv.bat" if system == "win" else "setenv.sh"
    src = paths.repo / "scripts" / name
    if not src.is_file():
        raise RuntimeError(
            f"{src} non trovato: step_sync_repo deve clonare o aggiornare "
            "paths.repo prima che step_layout giri. Verifica che "
            "'install' abbia eseguito il passo 'repo' senza errori."
        )
    dest = paths.internal / name
    dest.write_bytes(src.read_bytes())
    if system != "win":
        dest.chmod(0o755)

    if log is not None:
        log.info("copiato %s in %s", name, dest)
    return dest


def _shortcut_bytes(name: str, system: str) -> bytes:
    """Il contenuto che una scorciatoia con questo nome avrebbe, sul sistema
    dato. Condiviso fra la scrittura (SHORTCUTS) e il riconoscimento delle
    scorciatoie legacy (LEGACY_SHORTCUTS) sotto: e' lo stesso identico
    calcolo, e tenerlo in un solo posto e' cio' che garantisce che
    "contenuto generato da questa funzione" non diverga per i due usi."""
    if system == "win":
        return f'@echo off\r\ncall "%~dp0scripts\\{name}.bat" %*\r\n'.encode("ascii")
    return f'#!/usr/bin/env bash\nexec "$(dirname "$0")/scripts/{name}.sh" "$@"\n'.encode("utf-8")


def write_shortcuts(paths: InstallPaths, log) -> list[Path]:
    """Scrive in root l'unica scorciatoia (SHORTCUTS: la GUI). Richiama lo
    script corrispondente in `scripts/`, non ne ripete il contenuto.

    Rimuove anche, se le trova, le quattro scorciatoie per-passo di una
    versione precedente di questo installer (LEGACY_SHORTCUTS). PERCHE':
    `place_scripts`/`write_shortcuts` sono pensate per essere rilanciate a
    ogni update (rilanciare `install` E' l'aggiornamento), e senza questa
    pulizia chi aggiorna da un'installazione vecchia si ritroverebbe le
    quattro scorciatoie di prima ancora in root, perfettamente funzionanti
    (gli script in `scripts/` che richiamano ci sono ancora, `commands.toml`
    non li ha tolti) ma in contraddizione diretta con la richiesta di un
    solo avvio in evidenza: l'utente le vedrebbe e non capirebbe perche' non
    sono sparite.

    La rimozione e' prudente, non un `glob` per nome: un file di root con
    uno dei quattro nomi legacy viene cancellato solo se il suo contenuto e'
    esattamente quello che questa stessa funzione genererebbe per quel nome
    (`_shortcut_bytes`) -- cioe' solo se e' davvero un residuo generato da
    un giro precedente di `write_shortcuts`, mai un file che l'utente ha
    modificato o creato lui con lo stesso nome. E' lo stesso principio con
    cui `place_scripts` (sopra) tratta `scripts/`: territorio generato si
    puo' pulire, territorio dell'utente no -- qui pero' il nome da solo non
    basta a distinguerli, perche' un utente potrebbe benissimo avere un
    proprio file con uno di quei quattro nomi; il contenuto si'."""
    system = _system()
    suffix = ".bat" if system == "win" else ".sh"

    written = []
    for name in SHORTCUTS:
        target = paths.root / f"{name}{suffix}"
        target.write_bytes(_shortcut_bytes(name, system))
        if system != "win":
            target.chmod(0o755)
        written.append(target)

    removed = 0
    for name in LEGACY_SHORTCUTS:
        target = paths.root / f"{name}{suffix}"
        if target.is_file() and target.read_bytes() == _shortcut_bytes(name, system):
            target.unlink()
            removed += 1

    if log is not None:
        log.info("scritte %d scorciatoie in %s", len(written), paths.root)
        if removed:
            log.info("rimosse %d scorciatoie legacy da %s", removed, paths.root)
    return written


def create_workspace(paths: InstallPaths, log) -> None:
    """Crea la struttura di workspace/, senza mai toccare cio' che c'e' gia'."""
    for rel in WORKSPACE_TREE:
        (paths.workspace / rel).mkdir(parents=True, exist_ok=True)

    if log is not None:
        log.info("workspace pronto in %s", paths.workspace)
