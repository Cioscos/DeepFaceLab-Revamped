"""Clone o aggiornamento di ``_internal/DeepFaceLab``.

``git`` si invoca sempre tramite un ``runner`` iniettabile, con lo stesso
motivo di ``setup/preflight.py``: i test devono valere senza toccare la rete
ne' un repository vero.

Il caso che conta di piu' qui non e' tecnico ma di dati dell'utente: le
modifiche locali vanno rispettate e segnalate. Se qualcuno ha
toccato il proprio clone -- un branch locale, una patch al volo per debug --
rilanciare l'installer non deve sovrascriverlo con un `git pull`: lo
registra a log e prosegue, lasciando all'utente la scelta di fare
commit/stash e rilanciare per aggiornare davvero.
"""
from __future__ import annotations

import subprocess

from setup.paths import InstallPaths

# Il repository da cui l'installer prende il codice. Nessun --branch: `git
# clone` senza flag prende il branch di default del remoto, che e' cio' che
# un'installazione pulita deve seguire invece di un branch fissato a mano nel
# codice dell'installer.
REPO_URL = "https://github.com/Cioscos/DeepFaceLab-Revamped.git"

# Cio' che l'installer si aspetta di trovare in un clone completo. Senza
# questo controllo, un clone parziale fallisce molto piu' tardi, dentro
# step_install_requirements, con un errore di uv che non nomina la causa
# vera.
_REQUIRED_LAYOUT = ("setup", "requirements", "scripts/commands.toml")


def _check_installer_layout(paths: InstallPaths, runner) -> None:
    """Solleva RuntimeError se il clone non contiene cio' che l'installer usa.

    Va chiamata dopo un clone e dopo un pull riusciti (non dopo lo skip per
    modifiche locali, che non cambia nulla sul disco). Il messaggio dice
    cosa manca e in quale branch si trova il clone, non "installa il branch
    X": un nome di branch scritto qui diventerebbe stantio al primo merge,
    quando la causa piu' probabile tornerebbe a essere un'altra (un clone
    corrotto, un fetch parziale).
    """
    missing = [rel for rel in _REQUIRED_LAYOUT if not (paths.repo / rel).exists()]
    if not missing:
        return

    branch = "sconosciuto"
    try:
        result = runner(
            ["git", "-C", str(paths.repo), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            branch = result.stdout.strip()
    except (FileNotFoundError, OSError):
        pass

    raise RuntimeError(
        f"il clone in {paths.repo} (branch corrente: {branch}) non contiene "
        f"{', '.join(missing)}: mancano file che l'installer si aspetta di "
        "trovare subito dopo un clone o un aggiornamento. La causa piu' "
        "probabile e' un clone corrotto o un fetch parziale; un'altra e' che "
        "il branch su cui si trova il clone non includa l'installer. "
        "Verifica quale branch del "
        "repository contiene setup/, requirements/ e scripts/commands.toml, "
        "e riprova."
    )


def sync_repo(paths: InstallPaths, log, runner=subprocess.run) -> None:
    """Clona ``paths.repo`` se assente, altrimenti lo aggiorna.

    - assente -> ``git clone --depth 1 <REPO_URL> <repo>``.
    - presente, pulito -> ``git pull --ff-only``.
    - presente, con modifiche locali (``git status --porcelain`` non vuoto)
      -> nessun pull: solo un log e si prosegue.
    - ``git`` non nel PATH -> errore che dice come installarlo, non solo che
      manca.

    Dopo un clone o un pull riusciti, verifica che il risultato contenga
    davvero ``setup/``, ``requirements/`` e ``scripts/commands.toml``
    (``_check_installer_layout``): un branch senza l'installer deve fallire
    qui, non tre passi dopo dentro ``uv pip install``.
    """
    try:
        runner(["git", "--version"], capture_output=True, text=True, check=False)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            "git non trovato nel PATH: installalo da "
            "https://git-scm.com/downloads (Windows) oppure con il gestore "
            "pacchetti della tua distribuzione, es. `sudo apt install git` "
            "(Debian/Ubuntu) o `sudo dnf install git` (Fedora), poi rilancia "
            "l'installer."
        ) from exc

    git_dir = paths.repo / ".git"
    if not git_dir.exists():
        paths.repo.parent.mkdir(parents=True, exist_ok=True)
        if log is not None:
            log.info("clono %s in %s", REPO_URL, paths.repo)
        runner(["git", "clone", "--depth", "1", REPO_URL, str(paths.repo)], check=True)
        _check_installer_layout(paths, runner)
        return

    status = runner(
        ["git", "-C", str(paths.repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    if status.stdout.strip():
        if log is not None:
            log.warning(
                "modifiche locali non salvate in %s: salto 'git pull' per non "
                "sovrascriverle. Fai commit o stash e rilancia l'installer per "
                "aggiornare.",
                paths.repo,
            )
        return

    if log is not None:
        log.info("aggiorno %s (git pull --ff-only)", paths.repo)
    runner(["git", "-C", str(paths.repo), "pull", "--ff-only"], check=True)
    _check_installer_layout(paths, runner)
