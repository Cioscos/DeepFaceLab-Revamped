"""Rilevamento GPU e scelta fra la wheel CUDA e quella CPU.

nvidia-smi si invoca sempre tramite un ``runner`` iniettabile: e' cosi' che i
test valgono sia sulla macchina di sviluppo (che una GPU NVIDIA ce l'ha) sia
in CI (che non ce l'ha), senza dipendere da cosa gira davvero sulla macchina
che esegue la suite. ``choose_wheel_set`` non stampa nulla: restituisce solo
la scelta. E' ``setup/__main__.py`` (che ha il logger) a comporre il
messaggio per l'utente, in particolare quello per il caso "GPU presente ma
driver troppo vecchio".
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from setup.paths import InstallPaths

# MIN_DRIVER_WINDOWS / MIN_DRIVER_LINUX: il floor del driver NVIDIA per la
# build CUDA gia' decisa da requirements/cuda.txt (torch==2.13.0 dall'indice
# cu130 -- vedi il commento in testa a quel file per la lettura che l'ha
# fissata il 2026-08-02/03). Il pacchetto punta le RTX 3000 (sm_86),
# supportate da tutte le build correnti: il vincolo qui e' il driver, non
# l'architettura.
#
# Letti da fonte primaria il 2026-08-03 (UTC), non a memoria: i due numeri
# devono venire dallo stesso rilascio NVIDIA, non da una fascia di famiglia.
# Fonte
# unica per la coppia, verificata con `curl` sull'HTML grezzo:
#   https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-65-06/index.html
#   Titolo della pagina: "Version 580.65.06(Linux)/580.88(Windows)". Il corpo
#   elenca esplicitamente, sotto "CUDA Toolkit 13: 13.x": "NVIDIA Data Center
#   GPU Driver: 580.65.06 (Linux) / 580.88 (Windows)" -- lo stesso rilascio
#   R580 di cui 580.65.06 e' gia' il numero Linux usato qui sotto, quindi la
#   coppia e' coerente (stesso punto di rilascio) invece di un numero preciso
#   accostato a una fascia. pytorch.org stesso non pubblica una propria
#   tabella driver -> CUDA (verificato su get-started/locally e sulla
#   RELEASE.md del progetto): per questo numero rimanda alla documentazione
#   di compatibilita' NVIDIA sopra citata.
MIN_DRIVER_WINDOWS = "580.88"
MIN_DRIVER_LINUX = "580.65.06"

# Spazio disco minimo: 15 GB liberi.
NEEDED_BYTES = 15 * 1024 ** 3

# "OS e architettura" e' il primo dei tre controlli del passo preflight.
# uv, il CPython
# standalone e le wheel torch che l'installer scarica sono tutte x86_64;
# platform.machine() riporta "x86_64" su Linux e "AMD64" su Windows a 64 bit
# (verificato su questa macchina di sviluppo per Linux; per Windows e'
# comportamento stdlib documentato, non un dato che cambia nel tempo come una
# versione driver).
SUPPORTED_PLATFORMS = ("win32", "linux")
SUPPORTED_MACHINES = ("x86_64", "amd64")


@dataclass(frozen=True)
class GpuInfo:
    present: bool
    name: str
    driver_version: str


def check_platform() -> None:
    """Solleva RuntimeError se OS o architettura non sono quelli supportati.

    Va chiamato per primo in step_preflight, prima di disco e GPU: e' il
    controllo piu' economico e, senza, l'installer scarica CPython/uv/wheel
    x86_64 su un'architettura che non li puo' eseguire, per poi fallire
    *dopo* aver scaricato gigabyte, con un errore di pip che non nomina la
    causa vera."""
    if sys.platform not in SUPPORTED_PLATFORMS:
        raise RuntimeError(
            f"sistema operativo non supportato: {sys.platform!r} "
            f"(serve uno fra {SUPPORTED_PLATFORMS})"
        )
    machine = platform.machine()
    if machine.lower() not in SUPPORTED_MACHINES:
        raise RuntimeError(
            f"architettura non supportata: {machine!r} (serve x86_64/AMD64)"
        )


def detect_gpu(runner=subprocess.run) -> GpuInfo:
    """Interroga nvidia-smi tramite ``runner``.

    Nessuna GPU, nvidia-smi assente, o nvidia-smi che fallisce sono tutti lo
    stesso caso legittimo (non un guasto): l'installer prosegue sulla wheel
    CPU. Con piu' GPU, la prima riga vince.
    """
    try:
        result = runner(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return GpuInfo(False, "", "")

    if result.returncode != 0:
        return GpuInfo(False, "", "")

    first_line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return GpuInfo(False, "", "")

    name, _, driver = first_line.partition(",")
    return GpuInfo(True, name.strip(), driver.strip())


def _parse_driver_version(version: str) -> tuple[int, ...]:
    """'552.22' -> (552, 22); '552.22.01' -> (552, 22, 1).

    Confronto numerico, non come stringa: '9.0' > '10.0' se le due si
    confrontano come stringhe. Il formato varia anche fra Windows (due
    componenti) e Linux (tre)."""
    return tuple(int(part) for part in version.strip().split(".") if part.isdigit())


def _driver_at_least(actual: str, floor: str) -> bool:
    """Confronta due versioni driver a componenti diversi senza penalizzare
    quella con meno componenti.

    Python confronta le tuple a prefisso: (580,) < (580, 0) e' True anche se
    "580" e "580.0" sono la stessa versione -- una tupla piu' corta che e'
    prefisso dell'altra risulta sempre "minore". Si riempiono le due tuple
    con zeri fino alla stessa lunghezza prima di confrontarle."""
    a = _parse_driver_version(actual)
    b = _parse_driver_version(floor)
    length = max(len(a), len(b))
    a = a + (0,) * (length - len(a))
    b = b + (0,) * (length - len(b))
    return a >= b


def choose_wheel_set(gpu: GpuInfo, force_cpu: bool) -> str:
    """Restituisce "cuda" o "cpu". Non stampa (vedi il docstring del modulo)."""
    if force_cpu or not gpu.present:
        return "cpu"
    floor = MIN_DRIVER_WINDOWS if sys.platform == "win32" else MIN_DRIVER_LINUX
    if _driver_at_least(gpu.driver_version, floor):
        return "cuda"
    return "cpu"


def check_disk_space(paths: InstallPaths, needed_bytes: int) -> None:
    """Solleva RuntimeError se lo spazio libero e' insufficiente.

    ``paths.root`` puo' non esistere ancora (e' la cartella di destinazione
    dell'installazione, che i passi successivi creeranno): si risale ai
    genitori fino al primo che esiste gia', che sta comunque sullo stesso
    filesystem su cui l'installazione finira'."""
    target = Path(paths.root)
    while not target.exists():
        parent = target.parent
        if parent == target:
            break
        target = parent

    free = shutil.disk_usage(target).free
    if free < needed_bytes:
        raise RuntimeError(
            f"spazio insufficiente in {target}: {free / 1024 ** 3:.1f} GB liberi, "
            f"ne servono almeno {needed_bytes / 1024 ** 3:.1f} GB"
        )
