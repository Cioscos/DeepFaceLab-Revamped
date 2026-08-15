"""Dove vive la cache di una cartella di volti.

Fuori dal progetto, in _internal/_e/faceset-cache/<id>: e' una decisione
dell'utente, e ha una conseguenza che va pagata qui -- cancellare un
progetto lascia una cache orfana, quindi qualcuno deve potarla.

Questo e' l'unico posto che calcola quel nome. L'indicizzatore
(mainscripts/FacesetIndex.py) se lo fa dire con --cache-dir e non lo
ricalcola: due implementazioni potrebbero divergere, e una cache che
esiste ma che nessuno trova e' peggio di una cache assente.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

RADICE = "faceset-cache"


def id_cartella(path):
    """Un nome di cartella sicuro e stabile per un percorso qualsiasi."""
    normalizzato = os.path.normcase(os.path.abspath(str(path)))
    return hashlib.sha1(normalizzato.encode("utf-8")).hexdigest()


def percorso_cache(radice_e, path):
    return Path(radice_e) / RADICE / id_cartella(path)


def pota_orfane(radice_e):
    """Elimina le cache la cui cartella d'origine non esiste piu'. Ritorna quante."""
    base = Path(radice_e) / RADICE
    if not base.is_dir():
        return 0
    potate = 0
    try:
        voci = list(base.iterdir())
    except OSError:
        return 0
    for c in voci:
        meta = c / "meta.json"
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            origine = payload.get("origine")
            if not origine:
                continue
            if Path(origine).exists():
                continue
        except (OSError, ValueError):
            continue
        try:
            shutil.rmtree(c)
        except OSError:
            continue
        potate += 1
    return potate
