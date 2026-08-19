"""Il codice dell'applicazione: scaricato come archivio, non clonato.

Un'installazione non ha bisogno di `git`. L'archivio del ramo pubblicato pesa
poco piu' di due megabyte e si prende con gli stessi `curl`/`tar` che il
bootstrap usa gia' per procurarsi `uv`: chiedere a chi installa di procurarsi
un sistema di controllo di versione per copiare trecento file era il
prerequisito piu' costoso dell'intera procedura.

Come si riconosce che non e' cambiato niente: **si riscarica e si confronta lo
sha256 dell'archivio** con quello registrato. Non l'interfaccia HTTP di
GitHub, che e' limitata a sessanta richieste all'ora per indirizzo IP -- un
limite condiviso da tutti quelli dietro lo stesso indirizzo, e una modalita'
di fallimento in piu' da spiegare a chi installa; e non l'ETag, che non e'
documentato come stabile. Due megabyte e mezzo per rilancio, contro un
rilancio che comunque interroga la GPU, ricontrolla lo sha256 di quasi
quattrocento megabyte di pesi e fa girare `uv pip install`: non e' li' che si
spende il tempo.

La cancellazione di cio' che sparisce a monte e' guidata dall'elenco dei file
che l'archivio PRECEDENTE aveva portato, mai da un `rmtree` della cartella. I
pesi delle reti (`facelib/*.npy`, quasi quattrocento megabyte) vivono dentro
questa stessa cartella ma arrivano dal manifest, non dall'archivio: non
comparendo in nessun elenco sono fuori dalla portata della cancellazione per
costruzione, non per un'eccezione che qualcuno puo' dimenticare di scrivere.
Vale identico per `__pycache__/` e per qualunque cosa un processo abbia
scritto li'.

Se la cartella contiene un `.git`, questo modulo non fa nulla -- e non lancia
nessun comando git per decidere. Due storie diverse portano allo stesso
`.git`, e si distinguono da un dato che e' gia' li': se un'estrazione c'e'
gia' stata (`codice.json` esiste) e' l'albero di lavoro di qualcuno, messo
apposta; se non c'e' mai stata, e' il clone della procedura di installazione
precedente. In nessuno dei due casi si tocca, ma cio' che si dice a chi
installa e' diverso -- consigliare `git pull` a chi ha il secondo e' un
consiglio che non funziona.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from setup.paths import InstallPaths

# L'archivio del ramo pubblicato. Nessun riferimento fissato a una versione:
# un'installazione pulita segue l'ultima pubblicazione, che e' il
# comportamento che aveva anche prima.
URL_CODICE = "https://codeload.github.com/Cioscos/DeepFaceLab-Revamped/tar.gz/refs/heads/main"

# Cio' che deve esistere dopo ogni estrazione riuscita: un archivio che non li
# contiene non e' quello dell'applicazione, e va rifiutato qui e non tre passi
# piu' avanti dentro `uv pip install`, dove l'errore non nominerebbe la causa.
_LAYOUT_RICHIESTO = ("setup", "requirements", "scripts/commands.toml")

# L'archivio si legge a blocchi di un mebibyte, come `setup/assets.py`: e' la
# stessa ragione, non tenere in memoria in un colpo solo cio' che si scarica.
BLOCCO_BYTE = 1024 * 1024


def percorso_stato(paths: InstallPaths) -> Path:
    return paths.internal / "_e" / "codice.json"


def leggi_stato(paths: InstallPaths) -> dict:
    """Lo stato dell'ultima estrazione, o un dizionario vuoto.

    Un file illeggibile o troncato (un'interruzione a meta' scrittura) vale
    quanto un file assente: si riscarica e si riestrae. E' il ripiego piu'
    costoso possibile, e costa due megabyte e mezzo.
    """
    percorso = percorso_stato(paths)
    if not percorso.is_file():
        return {}
    try:
        dati = json.loads(percorso.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return dati if isinstance(dati, dict) else {}


def scrivi_stato(paths: InstallPaths, sha256: str, file: list[str]) -> None:
    percorso = percorso_stato(paths)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(
        json.dumps(
            {
                "sha256":    sha256,
                "url":       URL_CODICE,
                "scaricato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "file":      sorted(file),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def scarica_archivio(destinazione: Path, opener=urllib.request.urlopen) -> str:
    """Scarica l'archivio su `destinazione` e ne torna lo sha256.

    Si scrive su `<destinazione>.part` e si rinomina solo a trasferimento
    finito, come fa `setup/assets.py`: un'interruzione lascia o un file intero
    o un file dichiaratamente parziale, mai un troncato spacciato per
    completo. Nessuna ripresa con `Range` invece: la ripresa esiste per i
    download da un gigabyte e mezzo, qui si riparte da capo e costa due
    megabyte e mezzo.
    """
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    parziale = destinazione.parent / (destinazione.name + ".part")
    digest = hashlib.sha256()
    with opener(URL_CODICE) as risposta, parziale.open("wb") as uscita:
        while True:
            blocco = risposta.read(BLOCCO_BYTE)
            if not blocco:
                break
            digest.update(blocco)
            uscita.write(blocco)
    parziale.replace(destinazione)
    return digest.hexdigest()


def _estrai(archivio: Path, staging: Path) -> Path:
    """Estrae in una cartella di appoggio e torna l'unica radice trovata.

    `filter="data"` e' cio' che impedisce a un archivio malevolo di scrivere
    fuori dalla destinazione con percorsi assoluti o `..`.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archivio) as tar:
        tar.extractall(staging, filter="data")

    voci = sorted(staging.iterdir())
    if len(voci) != 1 or not voci[0].is_dir():
        raise RuntimeError(
            f"l'archivio scaricato da {URL_CODICE} non ha una sola cartella di "
            f"primo livello ma {len(voci)} voci ({', '.join(v.name for v in voci[:5])}): "
            "non e' l'archivio atteso. Riprova; se il problema resta, scarica "
            f"a mano {URL_CODICE} e verifica cosa contiene."
        )
    return voci[0]


def _sovrapponi(radice: Path, repo: Path) -> list[str]:
    """Copia l'albero estratto sopra la cartella del codice e torna i
    percorsi relativi di cio' che ha portato."""
    portati = []
    for sorgente in sorted(radice.rglob("*")):
        if sorgente.is_dir():
            continue
        rel = sorgente.relative_to(radice).as_posix()
        destinazione = repo / rel
        destinazione.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sorgente, destinazione)
        shutil.copymode(sorgente, destinazione)
        portati.append(rel)
    return portati


def _togli_spariti(repo: Path, precedenti: list[str], nuovi: list[str]) -> int:
    """Cancella i file che l'archivio precedente portava e questo non porta.

    Solo quelli: un file che non compare in `precedenti` non e' mai stato
    nostro e non si tocca, quale che sia il suo nome.
    """
    tolti = 0
    rimasti = set(nuovi)
    for rel in precedenti:
        if rel in rimasti:
            continue
        vecchio = repo / rel
        if vecchio.is_file():
            vecchio.unlink()
            tolti += 1
            # Rimuovere gli antenati rimasti vuoti.
            cartella = vecchio.parent
            while cartella != repo and cartella.is_dir() and not any(cartella.iterdir()):
                genitore = cartella.parent
                cartella.rmdir()
                cartella = genitore
    return tolti


def _verifica_layout(percorso: Path) -> None:
    mancanti = [rel for rel in _LAYOUT_RICHIESTO if not (percorso / rel).exists()]
    if not mancanti:
        return
    raise RuntimeError(
        f"l'archivio scaricato da {URL_CODICE} non contiene {', '.join(mancanti)}: "
        "mancano file che l'installazione si aspetta di trovare subito dopo "
        "lo scaricamento. La causa piu' probabile e' un archivio incompleto o "
        "un trasferimento interrotto. Riprova; se il problema resta, scarica "
        f"a mano {URL_CODICE} e verifica cosa contiene."
    )


def sincronizza_codice(paths: InstallPaths, log, opener=urllib.request.urlopen) -> bool:
    """Porta `paths.repo` all'ultima pubblicazione. Torna True se ha estratto.

    - `.git` presente -> non si tocca niente e non si scarica niente;
    - sha256 identico a quello registrato e tutti i file al loro posto ->
      niente da fare;
    - altrimenti -> estrazione in appoggio, sovrapposizione, rimozione di cio'
      che e' sparito a monte, stato riscritto.
    """
    if (paths.repo / ".git").exists():
        if log is not None:
            if percorso_stato(paths).is_file():
                # C'e' gia' stata un'estrazione qui dentro: il `.git` lo ha
                # messo qualcuno dopo, deliberatamente. E' un albero di
                # lavoro, e chi lo tiene sa come aggiornarlo.
                log.info(
                    "%s e' gestito con git: non lo aggiorno e non lo sovrascrivo. "
                    "Aggiornalo tu quando vuoi (per esempio con `git pull`).",
                    paths.repo,
                )
            else:
                # Nessuna estrazione e' mai avvenuta qui: questo `.git` e'
                # quello che ci ha messo la procedura di installazione
                # precedente, che clonava. Dirgli "aggiornalo tu con git
                # pull" sarebbe un consiglio che non funziona -- e' un clone
                # shallow di un repository ripubblicato con force-push -- e
                # lascerebbe l'installazione ferma per sempre al codice del
                # giorno in cui e' stata fatta.
                log.info(
                    "%s e' un clone git, quello che ci ha messo la procedura "
                    "di installazione precedente: non lo tocco, ma da qui in "
                    "avanti quel codice NON viene piu' aggiornato "
                    "automaticamente. Per tornare agli aggiornamenti "
                    "automatici: cancella quella cartella e rilancia "
                    "l'installazione, che scarichera' il codice come archivio "
                    "(due megabyte e mezzo). Nient'altro dell'installazione va "
                    "rifatto.",
                    paths.repo,
                )
        return False

    lavoro = paths.internal / "_e"
    archivio = lavoro / "codice.tar.gz"
    if log is not None:
        log.info("scarico il codice da %s", URL_CODICE)
    sha256 = scarica_archivio(archivio, opener=opener)

    stato = leggi_stato(paths)
    precedenti = list(stato.get("file", []))
    if (
        stato.get("sha256") == sha256
        and precedenti
        and all((paths.repo / rel).exists() for rel in precedenti)
    ):
        if log is not None:
            log.info("codice gia' aggiornato (%s): niente da estrarre", sha256[:12])
        archivio.unlink(missing_ok=True)
        return False

    staging = lavoro / "codice-nuovo"
    radice = _estrai(archivio, staging)
    _verifica_layout(radice)
    paths.repo.mkdir(parents=True, exist_ok=True)
    portati = _sovrapponi(radice, paths.repo)
    tolti = _togli_spariti(paths.repo, precedenti, portati)
    scrivi_stato(paths, sha256, portati)

    shutil.rmtree(staging, ignore_errors=True)
    archivio.unlink(missing_ok=True)
    if log is not None:
        log.info(
            "codice aggiornato in %s: %d file, %d rimossi (%s)",
            paths.repo, len(portati), tolti, sha256[:12],
        )
    return True
