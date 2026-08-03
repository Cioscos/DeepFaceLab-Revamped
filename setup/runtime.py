"""L'ambiente Python: venv e dipendenze, entrambi via ``uv``.

``uv`` si invoca sempre tramite un ``runner`` iniettabile (stesso motivo di
``setup/preflight.py`` e ``setup/repo.py``): i comandi che scaricano gigabyte
da PyPI/pytorch.org non vanno eseguiti in un unit test, si cattura solo cio'
che verrebbe lanciato.

``install_requirements`` fa **due** chiamate separate a ``uv pip install``,
mai una sola con entrambi i file (misura riportata nel commento in testa a
``requirements/cuda.txt``):
``--index-url`` in ``cuda.txt``/``cpu.txt`` diventa l'indice *esclusivo*
della risoluzione una volta che i due file vengono passati alla stessa
invocazione di ``uv pip install``, e pacchetti come ``numexpr`` (solo su
PyPI, non sull'indice torch) spariscono dalla risoluzione. Con due comandi
separati ciascuno vede un solo indice per volta e risolve correttamente.
"""
from __future__ import annotations

import os
import subprocess

from setup.paths import InstallPaths

# La versione pinnata per il venv: lo stesso interprete su
# cui il bootstrap lancia gia' setup/__main__.py con `uv run --python 3.11`.
PYTHON_VERSION = "3.11"


def uv_env(paths: InstallPaths) -> dict[str, str]:
    """L'ambiente con cui ogni invocazione di ``uv`` va lanciata.

    Le cinque variabili confinano ``uv`` dentro la cartella
    d'installazione: nessuna scrittura nel profilo utente, nessuna modifica
    al PATH di sistema -- la stessa filosofia che
    ``setenv.bat`` applica da sempre con ``_e``. Si parte da una copia
    dell'ambiente del processo (serve per trovare ``uv``/``git`` stessi sul
    PATH) e si sovrascrivono solo queste cinque chiavi, non tutto
    l'ambiente.

    ``UV_MANAGED_PYTHON`` e' l'unica che non riguarda *dove* uv scrive ma
    *quale* interprete sceglie, e senza di essa ``UV_PYTHON_INSTALL_DIR``
    non garantisce niente: dice dove mettere un Python scaricato, non
    obbliga a scaricarlo. La prima installazione fatta su Windows lo ha
    mostrato -- la macchina aveva un 3.11.9 sotto ``pyenv-win``, uv lo ha
    riusato, ``_internal/python`` non e' mai nata e ``pyvenv.cfg`` puntava
    fuori dal pacchetto, a un interprete che l'installer non aggiorna e che
    l'utente puo' disinstallare. Le stesse variabili valgono anche per
    l'``uv run`` del bootstrap, dove ``install.bat``/``install.sh`` le
    esportano per conto proprio: qui servono per le invocazioni che vengono
    dopo, e i due elenchi vanno tenuti allineati.
    """
    env = dict(os.environ)
    env.update({
        "UV_INSTALL_DIR": str(paths.internal / "uv"),
        "UV_PYTHON_INSTALL_DIR": str(paths.internal / "python"),
        "UV_CACHE_DIR": str(paths.internal / "_e" / "uv-cache"),
        "UV_NO_MODIFY_PATH": "1",
        "UV_MANAGED_PYTHON": "1",
    })
    return env


def create_venv(paths: InstallPaths, log, runner=subprocess.run) -> None:
    """``uv venv <venv> --python 3.11``.

    Salta la creazione se il venv esiste gia' (un rilancio e' un
    aggiornamento): senza ``--clear``, ``uv venv`` si rifiuta di
    sovrascrivere una directory gia' presente ("Failed to create virtual
    environment... A virtual environment already exists") -- verificato per
    davvero rilanciando l'installer su un'installazione gia' fatta: il
    secondo giro falliva qui, prima ancora di arrivare a
    ``install_requirements``, che e' gia' idempotente per conto suo (`uv pip
    install` risolve e non fa nulla se tutto e' gia' soddisfatto). Nessun
    test in questo file lanciava mai due volte di seguito lo stesso venv
    reale -- ``_Recorder`` cattura il comando e ritorna sempre successo,
    quindi il rifiuto di ``uv`` non era visibile finche' non si e' rilanciato
    l'installer per davvero.

    Il controllo guarda ``paths.python`` (l'eseguibile dentro il venv), non
    solo la cartella: un venv a meta' creato da un giro interrotto proprio
    qui non deve passare per "gia' pronto".
    """
    if paths.python.exists():
        if log is not None:
            log.info("venv gia' presente in %s: salto la creazione", paths.venv)
        return
    if log is not None:
        log.info("creo il venv in %s (Python %s)", paths.venv, PYTHON_VERSION)
    runner(
        [str(paths.uv_bin), "venv", str(paths.venv), "--python", PYTHON_VERSION],
        env=uv_env(paths),
        check=True,
    )


def install_requirements(paths: InstallPaths, wheel_set: str, log, runner=subprocess.run) -> None:
    """Installa ``base.txt`` e poi ``cuda.txt``/``cpu.txt`` nel venv.

    Due chiamate a ``uv pip install --python <paths.python> -r <file>``,
    mai una sola con entrambi i ``-r``: vedi il docstring del modulo per il
    perche'.
    """
    if wheel_set not in ("cuda", "cpu"):
        raise ValueError(f"wheel_set sconosciuto: {wheel_set!r} (atteso 'cuda' o 'cpu')")

    env = uv_env(paths)
    requirements_dir = paths.repo / "requirements"
    for req_file in (requirements_dir / "base.txt", requirements_dir / f"{wheel_set}.txt"):
        if log is not None:
            log.info("installo %s nel venv", req_file)
        runner(
            [
                str(paths.uv_bin), "pip", "install",
                "--python", str(paths.python),
                "-r", str(req_file),
            ],
            env=env,
            check=True,
        )
