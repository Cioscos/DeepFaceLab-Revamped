"""Da una chiave del catalogo a un estrattore costruito.

Qui si paga torch, e va bene: questo modulo lo importa chi estrae. Il
catalogo che elenca le chiavi sta invece in mainscripts/MotoriCatalog.py,
resta dati puri e lo legge anche la GUI.

Le chiavi -- e non gli oggetti -- sono cio' che attraversa il confine dei
processi: lo start method e' `spawn`, ogni figlio costruisce il proprio
estrattore da capo, e una classe passata per pickle sarebbe un modo
complicato di ottenere la stessa cosa piu' fragile.
"""
from collections import namedtuple

from mainscripts import MotoriCatalog

from .FaceType import FaceType
from .FANExtractor import FANExtractor
from .S3FDExtractor import S3FDExtractor
from .RetinaFaceExtractor import RetinaFaceExtractor
from .PipNetExtractor import PipNetExtractor

_RILEVATORI = {
    "S3FDExtractor": lambda **kw: S3FDExtractor(**kw),
    "RetinaFaceExtractor": lambda **kw: RetinaFaceExtractor(**kw),
}


def costruisci_rilevatore(chiave, place_model_on_cpu=False):
    motore = MotoriCatalog.rilevatore(chiave)
    costruttore = _RILEVATORI.get(motore.classe)
    if costruttore is None:
        raise KeyError(f"rilevatore sconosciuto: {motore.classe}")
    return costruttore(place_model_on_cpu=place_model_on_cpu, **motore.parametri)


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


# Il costruttore di ogni classe di allineatore E il fatto che consumi o no
# landmarks_3D, in UNA voce sola. I due dati stanno insieme perche' devono
# cambiare insieme: chi aggiunge un allineatore che legge il pavimento 2D/3D
# scrive `True` sulla stessa riga in cui scrive la lambda che lo usa. Un
# elenco di nomi tenuto a parte -- qui o, peggio, in ExtractManual -- e' la
# copia che diverge in silenzio, cioe' il difetto di partenza di questo
# lavoro ripetuto un livello piu' in la'.
_Allineatore = namedtuple("_Allineatore", ["costruisci", "consuma_landmarks_3D"])

_ALLINEATORI = {
    # Solo FANExtractor conosce landmarks_3D: e' la regola del pavimento di
    # landmarks_3D_per, che vale unicamente per lui. PipNetExtractor non ha
    # una variante 3D, quindi non deve nemmeno vedere il parametro.
    "FANExtractor": _Allineatore(
        costruisci=lambda face_type, chiave, **kw: FANExtractor(
            landmarks_3D=landmarks_3D_per(chiave, face_type), **kw),
        consuma_landmarks_3D=True),
    "PipNetExtractor": _Allineatore(
        costruisci=lambda face_type, chiave, **kw: PipNetExtractor(**kw),
        consuma_landmarks_3D=False),
}


def consuma_landmarks_3D(chiave):
    """Vero se l'allineatore di questa chiave guarda il pavimento 2D/3D.

    Chi tiene una CACHE di allineatori la chiave anche sul booleano
    landmarks_3D, e per un motore che non lo consuma quel booleano
    duplicherebbe la voce: `('pipnet-68', False)` e `('pipnet-68', True)`
    sarebbero due costruzioni dello stesso identico oggetto -- la stessa
    classe di duplicazione (in scala minore) che la cache di
    mainscripts/ExtractManual.py esiste per non avere. La risposta si legge
    da `_ALLINEATORI`, la stessa tabella che fornisce il costruttore, mai da
    un elenco di nomi scritto altrove.
    """
    motore = MotoriCatalog.allineatore(chiave)
    voce = _ALLINEATORI.get(motore.classe)
    if voce is None:
        raise KeyError(f"allineatore sconosciuto: {motore.classe}")
    return voce.consuma_landmarks_3D


def costruisci_allineatore(chiave, face_type, place_model_on_cpu=False):
    motore = MotoriCatalog.allineatore(chiave)
    voce = _ALLINEATORI.get(motore.classe)
    if voce is None:
        raise KeyError(f"allineatore sconosciuto: {motore.classe}")
    # `motore.parametri` non attraversa questo confine: per FANExtractor
    # l'unico parametro che porta (landmarks_3D) e' gia' ricalcolato da
    # landmarks_3D_per dentro la lambda, che rilegge la stessa chiave dal
    # catalogo; PipNetExtractor non ha parametri extra oggi. Rigirarlo qui
    # come **kw darebbe a FANExtractor due valori per landmarks_3D.
    return voce.costruisci(face_type, chiave, place_model_on_cpu=place_model_on_cpu)
