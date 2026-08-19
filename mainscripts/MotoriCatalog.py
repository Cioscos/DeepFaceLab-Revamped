"""I descrittori dei motori di rilevamento e di allineamento.

Sorgente unica dell'elenco, dell'ordine del selettore e delle choices di
argparse -- lo stesso ruolo che SorterCatalog ha per il sorting.

Dati puri, e deve restare tale: l'interfaccia grafica e la sua suite leggera
importano questo modulo, e non devono pagare torch per leggere una tabella.
Import consentiti: collections, niente altro. E' anche la ragione per cui
sta qui e non in facelib/, il cui __init__ importa S3FDExtractor e
FANExtractor: `mainscripts` non ha __init__.py, quindi importare questo
modulo non esegue nient'altro.

L'ordine e' APPEND-ONLY. La preferenza di progetto salva la CHIAVE e non
l'indice -- che e' proprio l'errore che il sorting si porta dietro -- ma il
selettore resta ordinato per posizione, e qui una voce spostata cambierebbe
il motore con cui si estrae un faceset da decine di migliaia di volti.

`manual` non e' in questa tabella e non e' una svista: non e' un rilevatore
ma un ramo diverso di Extractor.main, che invoca ExtractSubprocessor con
'landmarks-manual' invece di 'all' e non ha nulla che assomigli a
`extract(immagine) -> rects`. Resta una voce di --detector accanto al
catalogo, non dentro.
"""
from collections import namedtuple

Motore = namedtuple("Motore", [
    "key",         # chiave CLI e chiave salvata, stabile per sempre
    "label",       # nome breve in inglese: e' cio' che il selettore mostra
    "help",        # una frase, con il costo dove c'e' un costo
    "classe",      # nome della classe in facelib, risolto dalla fabbrica
    "parametri",   # kwargs aggiuntivi del costruttore, {} per il default
    "pesi",        # tupla dei file .npy sotto facelib/ che il motore puo'
                   # caricare -- di norma uno solo, ma "fan-2d" ne porta
                   # due: il face type e' un pavimento, non una regola
                   # (facelib/motori.py::landmarks_3D_per alza
                   # landmarks_3D a True per 'head' e oltre anche quando
                   # e' stato scelto "fan-2d"), quindi FANExtractor puo'
                   # caricare 3DFAN.npy pur essendo stato selezionato
                   # "2DFAN". Un motore e' selezionabile solo se TUTTI i
                   # file della sua tupla sono sul disco.
])

RILEVATORI = (
    Motore(
        key="s3fd",
        label="S3FD",
        help="The detector DeepFaceLab has always used. Scales the frame "
             "down to 640 px on the long side before the network sees it, "
             "which also filters out sensor noise and compression artifacts.",
        classe="S3FDExtractor",
        parametri={},
        pesi=("S3FD.npy",),
    ),
    Motore(
        key="s3fd-alta-risoluzione",
        label="S3FD, full resolution",
        help="The same detector without the 640 px cap. Three times slower, "
             "and measurably worse on noisy or heavily compressed footage. "
             "Only worth it for small faces in clean material.",
        classe="S3FDExtractor",
        parametri={"lato_rete": None},
        pesi=("S3FD.npy",),
    ),
    Motore(
        key="retinaface-r50",
        label="RetinaFace-R50",
        help="Finds profile faces and extreme poses that S3FD misses "
             "entirely. Slower than S3FD on easy footage, and it does not "
             "replace it: the two are offered side by side.",
        classe="RetinaFaceExtractor",
        parametri={},
        pesi=("RetinaFaceR50.npy",),
    ),
)

ALLINEATORI = (
    Motore(
        key="fan-2d",
        label="2DFAN",
        help="The default landmark model. Overridden by 3DFAN automatically "
             "when the face type is 'head', as it has always been.",
        classe="FANExtractor",
        parametri={"landmarks_3D": False},
        pesi=("2DFAN.npy", "3DFAN.npy"),
    ),
    Motore(
        key="fan-3d",
        label="3DFAN",
        help="The 3D landmark model already shipped in facelib, until now "
             "reachable only through the 'head' face type. Meant for poses "
             "2DFAN does not hold.",
        classe="FANExtractor",
        parametri={"landmarks_3D": True},
        pesi=("3DFAN.npy",),
    ),
    Motore(
        key="pipnet-68",
        label="PIPNet",
        help="Faster than 2DFAN and steadier on extreme yaw, with the same "
             "68 ibug points. It does not run the second refinement pass "
             "2DFAN does.",
        classe="PipNetExtractor",
        parametri={},
        pesi=("PIPNet68.npy",),
    ),
)

DEFAULT_RILEVATORE = "s3fd"
DEFAULT_ALLINEATORE = "fan-2d"

# Il lato minimo del volto accettato, in pixel del fotogramma d'origine.
# Non e' un attributo del Motore ma della corsa (--min-face-size), e sta qui
# per una ragione precisa: e' l'UNICO posto in cui il numero e' scritto.
# `S3FDExtractor.__init__` lo prende da qui, il prompt di `Extractor.main` lo
# mostra come default e il campo del catalogo GUI lo legge -- tre punti che
# altrimenti sarebbero tre copie libere di scostarsi, con l'utente che legge
# un numero nel form e ne ottiene un altro. Il valore e' quello storico,
# invariato: 40.
LATO_MIN_PREDEFINITO = 40

CHIAVI_RILEVATORI = tuple(m.key for m in RILEVATORI)
CHIAVI_ALLINEATORI = tuple(m.key for m in ALLINEATORI)

_PER_CHIAVE_R = dict((m.key, m) for m in RILEVATORI)
_PER_CHIAVE_A = dict((m.key, m) for m in ALLINEATORI)


def rilevatore(chiave):
    """Solleva KeyError su una chiave sconosciuta: un ripiego silenzioso sul
    default produrrebbe un faceset misto senza dirlo."""
    return _PER_CHIAVE_R[chiave]


def allineatore(chiave):
    return _PER_CHIAVE_A[chiave]
