"""Spostare nel cestino, annullare, eliminare davvero.

La cartella e' <cartella>_trash accanto -- la stessa costruzione dei venti
metodi di ordinamento (mainscripts/SorterCatalog.risolvi_artefatti), non
una seconda convenzione: un utente che alterna sort e cancellazioni a mano
deve trovare un cestino solo.

Dentro un cestino la cancellazione e' definitiva: li' il cestino e' gia'
il cestino, e un secondo livello sarebbe solo un posto in piu' in cui
dimenticare dei file.
"""
from collections import namedtuple
from pathlib import Path

Mossa = namedtuple("Mossa", ["coppie"])   # [(origine, destinazione), ...]

SUFFISSO = "_trash"


def cartella_cestino(cartella):
    cartella = Path(cartella)
    return cartella.parent / (cartella.stem + SUFFISSO)


def e_un_cestino(cartella):
    return Path(cartella).name.endswith(SUFFISSO)


def _destinazione_libera(trash, nome):
    candidata = trash / nome
    if not candidata.exists():
        return candidata
    stelo, punto, estensione = nome.partition(".")
    i = 2
    while True:
        candidata = trash / ("%s_%d%s%s" % (stelo, i, punto, estensione))
        if not candidata.exists():
            return candidata
        i += 1


def sposta_nel_cestino(percorsi, cartella):
    trash = cartella_cestino(cartella)
    try:
        trash.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Mossa([])
    coppie = []
    for p in percorsi:
        p = Path(p)
        try:
            destinazione = _destinazione_libera(trash, p.name)
            p.rename(destinazione)
        except OSError:
            continue
        coppie.append((p, destinazione))
    return Mossa(coppie)


def annulla(mossa):
    riportati = 0
    for origine, destinazione in mossa.coppie:
        if origine.exists():
            # lo slot originale e' di nuovo occupato -- una nuova
            # estrazione o un sort ci ha rimesso qualcosa nel frattempo.
            # rename() lo sovrascriverebbe in silenzio: il file resta nel
            # cestino invece di sparire per farne posto.
            continue
        try:
            destinazione.rename(origine)
        except OSError:
            continue
        riportati += 1
    return riportati


def elimina_definitivamente(percorsi):
    eliminati = 0
    for p in percorsi:
        try:
            Path(p).unlink()
        except OSError:
            continue
        eliminati += 1
    return eliminati
