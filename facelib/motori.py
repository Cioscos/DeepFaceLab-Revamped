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


def costruisci_allineatore(chiave, face_type, place_model_on_cpu=False):
    """`face_type` puo' ALZARE la scelta da 2D a 3D, mai abbassarla.

    E' il comportamento storico di Extractor -- 'head' ha sempre usato i
    landmark 3D -- conservato come pavimento invece che come regola: la
    scelta esplicita di fan-3d vale su ogni face type, ma nessuna scelta
    riporta 'head' a 2D.
    """
    motore = MotoriCatalog.allineatore(chiave)
    if motore.classe != "FANExtractor":
        raise KeyError(f"allineatore sconosciuto: {motore.classe}")
    parametri = dict(motore.parametri)
    parametri["landmarks_3D"] = bool(parametri.get("landmarks_3D")) \
        or face_type >= FaceType.HEAD
    return FANExtractor(place_model_on_cpu=place_model_on_cpu, **parametri)
