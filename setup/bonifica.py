"""La copia inerte del repository nella cartella di installazione.

Chi ha installato clonando (l'unica procedura documentata fino a oggi) si
ritrova il codice due volte: in cima, dove ha clonato, e in
_internal/DeepFaceLab, dove l'installazione lo mette. Solo la seconda viene
eseguita -- `setenv` fissa la radice li' e non altrove -- quindi la prima e'
solo ingombro, e per chi apre la cartella per capire cosa modificare e'
ingombro che inganna.

Toglierla e' l'unica operazione di questa procedura che cancella roba
dell'utente, quindi la regola e' stretta e si legge in una riga: **si
propone di togliere solo cio' che e' identico, byte per byte, alla copia che
resta**. Da qui discende tutto il resto senza bisogno di eccezioni: un file
modificato a mano non e' identico, uno script generato non ha omonimo nel
codice, e nessuno di loro e' candidabile.

Dove guardare lo dice l'elenco dell'archivio quando c'e', e altrimenti il
contenuto di `_internal/DeepFaceLab` -- che e' il caso di chi ha clonato,
cioe' proprio quello per cui questa procedura esiste: vedi
`elenco_di_riferimento`.

E si chiede sempre conferma, una volta sola: un no si registra e non si
ridomanda, perche' rilanciare questa procedura E' l'aggiornamento, e una
domanda che torna a ogni aggiornamento e' una domanda a cui si finisce per
rispondere senza leggerla.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from setup.codice import leggi_stato
from setup.commands import SHORTCUTS
from setup.paths import InstallPaths


def percorso_risposta(paths: InstallPaths) -> Path:
    return paths.internal / "_e" / "bonifica.json"


def leggi_risposta(paths: InstallPaths) -> bool | None:
    """La risposta gia' data, o None se non e' mai stata data."""
    percorso = percorso_risposta(paths)
    if not percorso.is_file():
        return None
    try:
        dati = json.loads(percorso.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    risposta = dati.get("rimuovere") if isinstance(dati, dict) else None
    return risposta if isinstance(risposta, bool) else None


def scrivi_risposta(paths: InstallPaths, risposta: bool) -> None:
    percorso = percorso_risposta(paths)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(
        json.dumps({"rimuovere": risposta,
                    "quando":    datetime.now(timezone.utc).isoformat(timespec="seconds")},
                   indent=2),
        encoding="utf-8",
    )


def lista_bianca() -> set[str]:
    """I file di root che vengono dall'archivio e devono restarci.

    I due bootstrap perche' sono l'avvio dell'aggiornamento; le scorciatoie
    perche' sono l'avvio dell'applicazione. Le scorciatoie in realta'
    nell'archivio non ci sono nemmeno -- le scrive `setup/layout.py` -- ma
    elencarle qui costa nulla e rende la regola leggibile senza dover sapere
    cosa l'archivio contiene.
    """
    suffisso = ".bat" if sys.platform == "win32" else ".sh"
    return {"install.bat", "install.sh"} | {f"{nome}{suffisso}" for nome in SHORTCUTS}


def elenco_di_riferimento(paths: InstallPaths) -> list[str]:
    """I percorsi con cui confrontare la root: quelli dell'archivio, se si
    sanno, altrimenti quelli che stanno in `paths.repo`.

    Il ripiego non e' una comodita', e' il caso che questa procedura serve
    piu' spesso: **chi ha installato con la procedura precedente ha clonato**,
    quindi `paths.repo` contiene un `.git`, quindi `sincronizza_codice` non
    scarica niente e non scrive `codice.json` -- e senza ripiego l'elenco
    sarebbe vuoto, la bonifica un nulla di fatto silenzioso, e la copia
    inerte in cima resterebbe li' per sempre proprio nelle installazioni per
    cui e' stata scritta.

    Ed e' sicuro perche' la condizione che porta la sicurezza non e'
    l'elenco: e' l'identita' byte per byte con la copia che resta (piu' la
    lista bianca). Un file dell'utente non e' identico a niente, e un file
    identico a quello che l'installazione esegue e' per definizione una
    copia, non un originale. L'elenco decide solo *dove guardare*.

    `.git` esce dal ripiego, e non e' un dettaglio: sia la root sia
    `paths.repo` sono cloni dello stesso repository, quindi file come
    `.git/description` o `.git/hooks/*.sample` sarebbero byte-identici e
    diventerebbero candidati -- si cancellerebbero pezzi del repository
    dell'utente credendo di togliere una copia inerte.
    """
    dallo_stato = leggi_stato(paths).get("file", [])
    if dallo_stato:
        return list(dallo_stato)
    if not paths.repo.is_dir():
        return []
    return [
        percorso.relative_to(paths.repo).as_posix()
        for percorso in paths.repo.rglob("*")
        if percorso.is_file() and ".git" not in percorso.relative_to(paths.repo).parts
    ]


def candidati(paths: InstallPaths) -> tuple[list[str], list[str]]:
    """(da togliere, che restano). Percorsi relativi alla cartella di
    installazione."""
    bianca = lista_bianca()
    da_togliere, restano = [], []
    for rel in elenco_di_riferimento(paths):
        if rel in bianca:
            continue
        in_root = paths.root / rel
        if not in_root.is_file():
            continue
        nel_codice = paths.repo / rel
        if nel_codice.is_file() and in_root.read_bytes() == nel_codice.read_bytes():
            da_togliere.append(rel)
        else:
            restano.append(rel)
    if (paths.root / ".git").exists():
        restano.append(".git")
    return sorted(da_togliere), sorted(restano)


def rimuovi(paths: InstallPaths, da_togliere: list[str]) -> int:
    """Cancella i candidati e le sole cartelle che restano vuote per causa loro.

    Le cartelle si potano risalendo dai file cancellati, mai camminando tutta
    la cartella di installazione: sotto ci sono `_internal` (il venv, i pesi)
    e `workspace` (i dati dell'utente), e una cartella vuota li' dentro puo'
    essere qualcosa che qualcuno ha creato apposta.
    """
    tolti = 0
    for rel in da_togliere:
        percorso = paths.root / rel
        if not percorso.is_file():
            continue
        percorso.unlink()
        tolti += 1
        cartella = percorso.parent
        while cartella != paths.root and cartella.is_dir() and not any(cartella.iterdir()):
            genitore = cartella.parent
            cartella.rmdir()
            cartella = genitore
    return tolti


def step_bonifica_root(paths: InstallPaths, args, log, chiedi=input) -> None:
    """Propone di togliere la copia inerte, una volta sola."""
    da_togliere, restano = candidati(paths)
    if not da_togliere:
        return

    if leggi_risposta(paths) is False:
        log.info(
            "in %s restano %d file del repository che avevi scelto di tenere: "
            "non vengono eseguiti, il codice che gira sta in %s.",
            paths.root, len(da_togliere), paths.repo,
        )
        return

    log.info(
        "in %s ci sono %d file identici a quelli di %s: sono una copia che non "
        "viene mai eseguita, lasciata dalla procedura di installazione "
        "precedente.",
        paths.root, len(da_togliere), paths.repo,
    )
    for rel in da_togliere[:10]:
        log.info("  - %s", rel)
    if len(da_togliere) > 10:
        log.info("  ... e altri %d", len(da_togliere) - 10)
    if restano:
        log.info("non toccati in nessun caso: %s", ", ".join(restano))

    if not args.yes:
        risposta = chiedi("Rimuoverli? [s/N] ")
        if risposta.strip().lower() not in ("s", "si", "sì", "y", "yes"):
            scrivi_risposta(paths, False)
            log.info("non rimuovo niente: non te lo chiedero' piu'.")
            return

    tolti = rimuovi(paths, da_togliere)
    scrivi_risposta(paths, True)
    log.info("rimossi %d file da %s", tolti, paths.root)
