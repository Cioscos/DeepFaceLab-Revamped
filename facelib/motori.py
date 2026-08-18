"""Da una chiave del catalogo a un estrattore costruito.

Qui si paga torch, e va bene: questo modulo lo importa chi estrae. Il
catalogo che elenca le chiavi sta invece in mainscripts/MotoriCatalog.py,
resta dati puri e lo legge anche la GUI.

Le chiavi -- e non gli oggetti -- sono cio' che attraversa il confine dei
processi: lo start method e' `spawn`, ogni figlio costruisce il proprio
estrattore da capo, e una classe passata per pickle sarebbe un modo
complicato di ottenere la stessa cosa piu' fragile.
"""
from mainscripts import MotoriCatalog

from .FaceType import FaceType
from .FANExtractor import FANExtractor
from .S3FDExtractor import S3FDExtractor


def costruisci_rilevatore(chiave, place_model_on_cpu=False):
    motore = MotoriCatalog.rilevatore(chiave)
    if motore.classe != "S3FDExtractor":
        raise KeyError(f"rilevatore sconosciuto: {motore.classe}")
    return S3FDExtractor(place_model_on_cpu=place_model_on_cpu,
                         **motore.parametri)


def landmarks_3D_per(chiave, face_type):
    """Vero se questa coppia (chiave, face_type) risolve in landmark 3D.

    `face_type` puo' ALZARE la scelta da 2D a 3D, mai abbassarla. E' il
    comportamento storico di Extractor -- 'head' ha sempre usato i
    landmark 3D -- conservato come pavimento invece che come regola: la
    scelta esplicita di fan-3d vale su ogni face type, ma nessuna scelta
    riporta 'head' a 2D.

    Unico posto dove questa regola vive: `costruisci_allineatore` la legge
    per il parametro vero passato a `FANExtractor`, e chi ha bisogno di
    una CHIAVE di cache per lo stesso allineatore (senza costruirlo)
    -- oggi `mainscripts/ExtractManual.py::_allineatore` -- la legge da
    qui invece di riscrivere la formula. Due copie della stessa soglia
    sono innocue finche' coincidono, ma il giorno che una cambia sola la
    chiave smette di corrispondere al comportamento reale: il servizio
    tornerebbe l'allineatore sbagliato in silenzio, senza errore ne'
    log -- la peggiore classe di difetto che questo modulo possa avere.
    """
    motore = MotoriCatalog.allineatore(chiave)
    return bool(motore.parametri.get("landmarks_3D")) or face_type >= FaceType.HEAD


def costruisci_allineatore(chiave, face_type, place_model_on_cpu=False):
    motore = MotoriCatalog.allineatore(chiave)
    if motore.classe != "FANExtractor":
        raise KeyError(f"allineatore sconosciuto: {motore.classe}")
    parametri = dict(motore.parametri)
    parametri["landmarks_3D"] = landmarks_3D_per(chiave, face_type)
    return FANExtractor(place_model_on_cpu=place_model_on_cpu, **parametri)
