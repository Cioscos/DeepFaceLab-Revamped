"""Il servizio di dettaglio: due operazioni, a piena fedelta'.

Persistente perche' l'import costa 6,11 s e servire un volto 5,80 ms: un
processo per doppio click renderebbe la finestra inutilizzabile.

Il protocollo non trasporta byte. I vettori (landmark, poligoni) vanno in
JSON, il raster (la maschera non ridotta) passa da un file annunciato DOPO
essere stato scritto -- lo stesso schema di DFL_PREVIEW_DIR
(PreviewWriter, mainscripts/TrainerLib.py): si scrive su un file
temporaneo, si rinomina in modo atomico (os.replace), e SOLO ALLORA si
annuncia il nome nella risposta. L'immagine del volto non passa affatto:
e' un JPEG ordinario, e chi la vuole la apre da se'.

La maschera dentro il JPEG e' gia' compressa: come in
mainscripts/FacesetIndex.py, la scriviamo verbatim, senza decodificarla
per ricodificarla (costerebbe una decodifica in cambio di niente). Il
formato pero' NON e' garantito PNG: DFLIMG.DFLJPG.set_xseg_mask ripiega
su JPEG oltre una soglia di dimensione (50000 byte, DFLIMG/DFLJPG.py). Il
nome del file scritto qui riflette il formato vero, riconosciuto dai
primi byte, cosi' il client non deve indovinare dall'estensione.

Un guasto risponde `{"op": "error", "motivo": ..., "codice": ...}`. Il
motivo e' il testo dell'eccezione, italiano e d'implementazione: serve a
chi legge un registro, non a chi guarda una finestra. Il codice viene da
DettaglioGuasti, e c'e' solo per i guasti che i dati o la macchina di chi
usa il programma possono davvero produrre -- la GUI ci mappa sopra una
frase sua. Vale None per tutti gli altri, ed e' voluto: chi legge ha un
ripiego generico, e nessun guasto sparisce perche' non e' catalogato.

`DFLJPG.load` cattura le proprie eccezioni e, prima di
tornare None, chiama `io.log_err(...)` -- che stampa su `sys.stdout` vero
(`core/interact/interact.py`), lo stesso canale del protocollo JSON riga
per riga. Un file che non e' un JPEG DFL, o un percorso inesistente,
avrebbe scritto un traceback multi-riga PRIMA della risposta di errore,
desincronizzando il parser del client su una pipe vera. `rispondi()` gira
quindi sotto `contextlib.redirect_stdout(sys.stderr)`: qualunque stampa
generata durante l'elaborazione (non solo da `load`, da qualunque cosa
domani) finisce su stderr, mai sul canale del protocollo. `servi()` scrive
la riga di risposta FUORI da quel blocco, quindi il protocollo stesso non
si sposta.
"""
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from DFLIMG import DFLJPG
from mainscripts import CanaleComandi, DettaglioGuasti

FIRMA_PNG = b"\x89PNG\r\n\x1a\n"


class Guasto(ValueError):
    """Un guasto con attaccato un codice di DettaglioGuasti.

    Il codice viaggia con l'eccezione fino a `_errore`, che lo mette nel
    dizionario di risposta accanto al motivo: chi legge il protocollo non
    deve riconoscere il guasto dal testo italiano del messaggio, che e'
    d'implementazione e puo' cambiare parola in qualunque momento.
    """

    def __init__(self, codice, motivo):
        super().__init__(motivo)
        self.codice = codice


def _errore(ident, e):
    """La risposta di errore: il motivo grezzo e, se l'eccezione lo porta,
    il codice.

    `codice` None e' la norma e non un buco -- un'eccezione qualunque (una
    libreria, il disco, il protocollo violato) non ha nessuna diagnosi da
    promettere, e chi legge ha un ripiego generico che la mostra comunque.
    """
    return {"op": "error", "id": ident, "motivo": str(e),
            "codice": getattr(e, "codice", None)}


def _estensione_maschera(byte_grezzi):
    return "png" if byte_grezzi[:8] == FIRMA_PNG else "jpg"


def _polys_serializzabili(polys):
    """SegIEPolys.dump() torna 'pts' come array numpy dentro ogni
    poligono: json.dumps ci si strozza sopra. I volti di prova di questo
    servizio non hanno poligoni, quindi la suite non lo eserciterebbe da
    sola -- ma un volto reale che li ha non deve poter fermare il ciclo."""
    for poligono in polys.get("polys", []):
        pts = poligono.get("pts")
        if hasattr(pts, "tolist"):
            poligono["pts"] = pts.tolist()
    return polys


def _scrivi_maschera_e_annuncia(byte_grezzi, ident, workdir):
    """Scrive la maschera su un file temporaneo, la rinomina in modo
    atomico e ritorna il nome finale -- solo a scrittura conclusa, mai
    prima, cosi' un file annunciato esiste sempre."""
    nome = "%s_mask.%s" % (ident, _estensione_maschera(byte_grezzi))
    finale = Path(workdir) / nome
    provvisorio = Path(workdir) / (nome + ".tmp")
    provvisorio.write_bytes(byte_grezzi)
    os.replace(str(provvisorio), str(finale))
    return nome


def _landmarks_di_frame(dfl):
    """`source_landmarks` come lista di coppie, o None.

    `DFLJPG.get_source_landmarks()` e' `np.array(dict.get(..., None))`: su
    un file che non ha il campo NON torna None, torna un array numpy
    0-dimensionale di tipo object. `is None` non lo intercetta, e
    `.tolist()` ci mette dentro un None al posto della lista di coppie che
    il client si aspetta. Il controllo giusto e' sulla FORMA."""
    lmrks = np.asarray(dfl.get_source_landmarks())
    if lmrks.ndim != 2 or lmrks.shape[1] != 2:
        return None
    return lmrks.astype(float).tolist()


def _posa_e_lato(dfl, rect):
    """I due campi che `ExtractReport.voce` vuole per ogni volto.

    La canonicalizzazione e' la STESSA di ExtractIndex._volto_da_dfl, e non
    per gusto: `get_landmarks()` torna i punti nella dimensione con cui il
    volto e' stato salvato (512, o 768 per head), mentre
    estimate_pitch_yaw_roll assume col suo default una camera tarata per
    256. Passarle i punti cosi' come sono da' una posa sbagliata in
    silenzio -- misurato altrove: ~14 gradi di pitch, ~28 di yaw. Il lato
    invece viene dal rettangolo nello spazio del FOTOGRAMMA.

    Un guasto di facelib qui degrada alla posa di ripiego -- la stessa
    dei landmark non 68 -- invece di sollevare: questa funzione gira
    dentro il ciclo di `_rispondi_frame`, che protegge un file rotto
    perdendo SOLO lui. Lasciarla sollevare perderebbe l'intera voce
    (rettangolo, landmark, maschera compresi) per un guasto che riguarda
    solo la posa.
    """
    from facelib import FaceType, LandmarksProcessor
    posa = [0.0, 0.0, 0.0]
    lmrks = np.asarray(dfl.get_landmarks())
    if lmrks.ndim == 2 and lmrks.shape[0] == 68:
        try:
            mat = LandmarksProcessor.get_transform_mat(lmrks, 256, FaceType.FULL)
            allineati = LandmarksProcessor.transform_points(lmrks, mat)
            posa = [float(p) for p in
                    LandmarksProcessor.estimate_pitch_yaw_roll(allineati)]
        except Exception:
            posa = [0.0, 0.0, 0.0]
    lato = 0
    if rect is not None:
        l, t, r, b = rect
        lato = max(r - l, b - t)
    return posa, int(lato)


def _volto_per_il_protocollo(percorso, ident, indice, workdir):
    """Una voce di `volti`, o None se il file non e' un JPEG DFL.

    Ogni campo e' None per conto suo: un allineato puo' avere i landmark e
    non la maschera, o la matrice e non il rettangolo (un file scritto da
    una versione piu' vecchia). Nessun campo mancante fa saltare la voce.
    """
    dfl = DFLJPG.load(str(percorso))
    if dfl is None:
        return None
    rect = dfl.get_source_rect()
    mat = dfl.get_image_to_face_mat()
    voce = {
        "path": str(percorso),
        "rect": None if rect is None
                else [int(v) for v in np.asarray(rect).reshape(-1)[:4]],
        "source_landmarks": _landmarks_di_frame(dfl),
        "mat": None if mat is None else np.asarray(mat).astype(float).tolist(),
        "shape": list(dfl.get_shape()),
        "mask": None,
    }
    voce["posa"], voce["lato"] = _posa_e_lato(dfl, voce["rect"])
    if dfl.has_xseg_mask():
        byte_grezzi = bytes(np.asarray(dfl.get_xseg_mask_compressed()).tobytes())
        # L'INDICE oltre all'id: due volti dello stesso fotogramma stanno
        # nella stessa risposta, e col solo id il secondo sovrascriverebbe
        # il file del primo in silenzio.
        voce["mask"] = _scrivi_maschera_e_annuncia(
            byte_grezzi, "%s_%s" % (ident, indice), workdir)
    return voce


def _rispondi_open(comando, ident, workdir):
    percorso = comando.get("path")
    try:
        dfl = DFLJPG.load(str(percorso))
        if dfl is None:
            raise Guasto(DettaglioGuasti.FILE_ILLEGGIBILE, "non e' un JPEG DFL")
        forma = list(dfl.get_shape())
        landmarks = np.asarray(dfl.get_landmarks()).tolist()
        polys = None
        if dfl.has_seg_ie_polys():
            polys = _polys_serializzabili(dfl.get_seg_ie_polys().dump())
        risposta = {"op": "opened", "id": ident, "shape": forma,
                    "face_type": dfl.get_face_type(),
                    "landmarks": landmarks, "polys": polys, "mask": None}
        rect = dfl.get_source_rect()
        mat = dfl.get_image_to_face_mat()
        risposta.update({
            # `source_landmarks` e `mat` sono gli stessi campi che
            # `_volto_per_il_protocollo` gia' consegna per ogni volto in
            # `frame` (li' `mat` si chiama cosi', il rettangolo invece
            # `rect` -- qui e' `source_rect` per non confonderlo con
            # `shape`). `source_filename` e' nuovo: non attraversa `frame`
            # perche' li' e' gia' un argomento del comando, non una
            # risposta. Sono aggiunte, non cambiamenti -- chi legge solo i
            # campi vecchi continua a funzionare.
            "source_filename": dfl.get_source_filename(),
            "source_landmarks": _landmarks_di_frame(dfl),
            "source_rect": None if rect is None
                           else [int(v) for v in np.asarray(rect).reshape(-1)[:4]],
            "mat": None if mat is None else np.asarray(mat).astype(float).tolist(),
        })
        if dfl.has_xseg_mask():
            byte_grezzi = bytes(np.asarray(dfl.get_xseg_mask_compressed()).tobytes())
            risposta["mask"] = _scrivi_maschera_e_annuncia(byte_grezzi, ident, workdir)
        return risposta
    except Exception as e:
        return _errore(ident, e)


def _rispondi_frame(comando, ident, workdir):
    """I volti gia' su disco per un fotogramma.

    I percorsi li decide il CHIAMANTE. Questo servizio non sa piu' quale
    file venga da quale fotogramma, e non deve saperlo: la regola vive in
    gui/faceset/indice.py::mappa_per_fotogramma, che la ricava dall'indice
    invece che dal nome del file. Un elenco vuoto -- un progetto su cui non
    si e' ancora estratto niente -- non e' un errore.
    """
    percorsi = comando.get("percorsi")
    if not isinstance(percorsi, list):
        percorsi = []
    volti = []
    for indice, percorso in enumerate(percorsi):
        try:
            voce = _volto_per_il_protocollo(percorso, ident, indice, workdir)
        except Exception:
            # Un file rotto in mezzo a due buoni perde se stesso, non gli
            # altri: e' cio' che uno Stop a meta' estrazione lascia dietro.
            continue
        if voce is not None:
            volti.append(voce)
    return {"op": "framed", "id": ident, "volti": volti}


# Due nomi alternati, non uno solo e non uno nuovo per richiesta. Uno solo
# espone a una lettura stantia (Qt puo' avere ancora in mano il file mentre
# lo si riscrive); uno nuovo per richiesta fa crescere la workdir a OGNI
# trascinamento, e quella cartella non la cancella gia' nessuno.
NOMI_ANTEPRIMA = ("anteprima_a.png", "anteprima_b.png")
NOMI_MASCHERA_ANTEPRIMA = ("maschera_a.png", "maschera_b.png")

# `rispondi` e' una funzione e non ha stato: l'alternanza vive qui, chiavata
# sulla workdir, perche' due finestre aperte insieme ne hanno una per una.
_alternanza = {}


def _prossimo_indice(workdir):
    """L'indice che TOCCHEREBBE alla prossima scrittura, senza consumarlo:
    lo stato si aggiorna solo con `_consuma_indice`, chiamato a scrittura
    riuscita. Se la scrittura sollevasse fra le due chiamate, la richiesta
    dopo ripeterebbe lo stesso indice invece di saltarne uno -- l'ordine
    naturale e' scrivere e poi avanzare, non il contrario."""
    return 1 - _alternanza.get(str(workdir), 1)


def _consuma_indice(workdir, indice):
    _alternanza[str(workdir)] = indice


def _scrivi_raster_e_annuncia(immagine, nome, workdir):
    """Come _scrivi_maschera_e_annuncia, ma per un raster in memoria:
    si codifica, si scrive su .tmp, si rinomina, e SOLO ALLORA si torna
    il nome."""
    import cv2
    ok, buf = cv2.imencode(".png", immagine)
    if not ok:
        raise ValueError("non si riesce a codificare l'anteprima")
    finale = Path(workdir) / nome
    provvisorio = Path(workdir) / (nome + ".tmp")
    provvisorio.write_bytes(bytes(buf))
    os.replace(str(provvisorio), str(finale))
    return nome


def _rispondi_riallinea(comando, ident, workdir):
    """L'anteprima del riallineamento. NON tocca il file allineato: legge
    i suoi metadati, warpa, e scrive solo nella workdir temporanea."""
    from core.cv2ex import cv2_imread
    from mainscripts import ExtractorLib
    try:
        dfl = DFLJPG.load(str(comando.get("path")))
        if dfl is None:
            raise Guasto(DettaglioGuasti.FILE_ILLEGGIBILE, "non e' un JPEG DFL")
        mat_vecchia = dfl.get_image_to_face_mat()
        if mat_vecchia is None:
            raise Guasto(DettaglioGuasti.SENZA_MATRICE,
                         "questo volto non ha una matrice di allineamento")
        nome_frame = dfl.get_source_filename()
        frame = cv2_imread(str(Path(comando.get("frame_dir")) / str(nome_frame)))
        if frame is None:
            raise Guasto(DettaglioGuasti.FRAME_ASSENTE,
                         "il fotogramma non e' su disco: %s" % (nome_frame,))
        lato = int(dfl.get_shape()[0])
        punti = np.array(comando.get("source_landmarks"), dtype=np.float32)
        # 68, non solo "coppie": get_transform_mat legge solo [17:49]+[54:55]
        # e non solleva su un array piu' corto ma ancora fatto di coppie --
        # 60 punti, per dire, produce un allineamento plausibile e SBAGLIATO
        # in silenzio, indistinguibile da uno buono. La lunghezza e' cio'
        # che conta davvero, non solo la forma.
        if punti.ndim != 2 or punti.shape[1] != 2 or punti.shape[0] != 68:
            raise ValueError("servono 68 coppie di landmark")

        raster, mat_nuova, lmrks = ExtractorLib.riallinea_volto(
            frame, punti, dfl.get_face_type(), lato)
        if not np.all(np.isfinite(mat_nuova)):
            # Landmark tutti coincidenti (l'utente trascina ogni punto sullo
            # stesso pixel) fanno tornare a get_transform_mat una matrice
            # tutta NaN senza sollevare -- stesso controllo di
            # riallinea_e_salva, e per la stessa ragione: annunciare
            # "riallineato" con un raster nero e una maschera azzerata e'
            # il risultato sbagliato in silenzio, la classe peggiore. Va
            # fatto PRIMA di scrivere qualunque cosa nella workdir.
            raise Guasto(DettaglioGuasti.ALLINEAMENTO_NON_VALIDO,
                         "i landmark forniti non producono un allineamento valido")
        indice = _prossimo_indice(workdir)
        risposta = {
            "op": "riallineato", "id": ident, "lato": lato,
            "mat": np.asarray(mat_nuova).astype(float).tolist(),
            "landmarks": np.asarray(lmrks).astype(float).tolist(),
            "raster": _scrivi_raster_e_annuncia(
                raster, NOMI_ANTEPRIMA[indice], workdir),
            "maschera": None,
        }
        # La maschera va trasportata anche nell'ANTEPRIMA, non solo al
        # salvataggio: senza, si vedrebbe il raster nuovo con sopra la
        # maschera del VECCHIO allineamento -- una maschera fuori posto,
        # che a schermo si legge come una segmentazione sbagliata invece
        # che come un'anteprima incompleta.
        if dfl.has_xseg_mask():
            composta = ExtractorLib.composta_fra_allineamenti(mat_vecchia, mat_nuova)
            ExtractorLib.trasporta_maschera(dfl, composta, lato)
            maschera = np.clip(np.asarray(dfl.get_xseg_mask()) * 255, 0, 255).astype(np.uint8)
            risposta["maschera"] = _scrivi_raster_e_annuncia(
                maschera, NOMI_MASCHERA_ANTEPRIMA[indice], workdir)
        _consuma_indice(workdir, indice)
        return risposta
    except Exception as e:
        return _errore(ident, e)


def _rispondi_salva(comando, ident, _workdir):
    """L'unica operazione di questo servizio che scrive nel PROGETTO.

    Il lavoro sta tutto in ExtractorLib.riallinea_e_salva, che si porta
    dietro maschera e poligoni; qui si carica il fotogramma e si
    impacchetta. La workdir non serve: non si annuncia nessun raster --
    chi ha chiesto il salvataggio ha gia' l'anteprima.
    """
    from core.cv2ex import cv2_imread
    from mainscripts import ExtractorLib
    try:
        dfl = DFLJPG.load(str(comando.get("path")))
        if dfl is None:
            raise Guasto(DettaglioGuasti.FILE_ILLEGGIBILE, "non e' un JPEG DFL")
        nome_frame = dfl.get_source_filename()
        frame = cv2_imread(str(Path(comando.get("frame_dir")) / str(nome_frame)))
        if frame is None:
            raise Guasto(DettaglioGuasti.FRAME_ASSENTE,
                         "il fotogramma non e' su disco: %s" % (nome_frame,))
        punti = np.array(comando.get("source_landmarks"), dtype=np.float32)
        # 68, non solo "coppie": get_transform_mat legge solo [17:49]+[54:55]
        # e non solleva su un array piu' corto ma ancora fatto di coppie --
        # 60 punti, per dire, produce un allineamento plausibile e SBAGLIATO
        # in silenzio, indistinguibile da uno buono. La lunghezza e' cio'
        # che conta davvero, non solo la forma -- stesso controllo di
        # _rispondi_riallinea, e per la stessa ragione.
        if punti.ndim != 2 or punti.shape[1] != 2 or punti.shape[0] != 68:
            raise ValueError("servono 68 coppie di landmark")
        esito = ExtractorLib.riallinea_e_salva(comando.get("path"), frame, punti)
        return dict(esito, op="salvato", id=ident)
    except Exception as e:
        return _errore(ident, e)


# I motori sono pesi in VRAM su una macchina che puo' stare addestrando:
# si costruiscono al primo uso e si tengono, e il sorvegliante
# d'inattivita' li porta via col processo. Chiavati su (tipo, chiave,
# face_type): lo stesso allineatore su un face type diverso e' un motore
# diverso, per la regola-pavimento di facelib/motori.py::landmarks_3D_per.
_MOTORI = {}


def _costruisci_allineatore(chiave, face_type):
    from facelib import motori
    return motori.costruisci_allineatore(chiave, face_type)


def _costruisci_rilevatore(chiave):
    from facelib import motori
    return motori.costruisci_rilevatore(chiave)


def _allineatore_per(chiave, face_type):
    voce = ("allineatore", chiave, str(face_type))
    if voce not in _MOTORI:
        _MOTORI[voce] = _costruisci_allineatore(chiave, face_type)
    return _MOTORI[voce]


def _rilevatore_per(chiave):
    voce = ("rilevatore", chiave)
    if voce not in _MOTORI:
        _MOTORI[voce] = _costruisci_rilevatore(chiave)
    return _MOTORI[voce]


def _rispondi_rileva(comando, ident, _workdir):
    """Proposte, mai una scrittura.

    Due modi. `landmarks` costruisce SOLO l'allineatore e lo fa girare sul
    rettangolo che il volto ha gia': e' cio' che ExtractManual._op_rileva
    fa quando il rect arriva esplicito (`serve_rilevatore = rect is None or
    accurato`). `volto` costruisce anche il rilevatore e gira sul
    fotogramma intero, e puo' tornare zero, una o piu' proposte -- nessuna
    delle quali e' necessariamente il volto che si stava guardando.
    """
    from core.cv2ex import cv2_imread
    from facelib import FaceType
    from mainscripts import MotoriCatalog
    try:
        modo = comando.get("modo")
        if modo not in ("landmarks", "volto"):
            raise ValueError("modo di rilevamento sconosciuto: %r" % (modo,))
        chiave_all = comando.get("allineatore")
        if chiave_all not in MotoriCatalog.CHIAVI_ALLINEATORI:
            raise ValueError("allineatore sconosciuto: %r" % (chiave_all,))

        dfl = DFLJPG.load(str(comando.get("path")))
        if dfl is None:
            raise Guasto(DettaglioGuasti.FILE_ILLEGGIBILE, "non e' un JPEG DFL")
        nome_frame = dfl.get_source_filename()
        frame = cv2_imread(str(Path(comando.get("frame_dir")) / str(nome_frame)))
        if frame is None:
            raise Guasto(DettaglioGuasti.FRAME_ASSENTE,
                         "il fotogramma non e' su disco: %s" % (nome_frame,))
        face_type = FaceType.fromString(dfl.get_face_type())

        if modo == "landmarks":
            rect = dfl.get_source_rect()
            if rect is None:
                raise Guasto(DettaglioGuasti.SENZA_RETTANGOLO,
                             "questo volto non ha un rettangolo di partenza")
            rects = [[int(v) for v in np.asarray(rect).reshape(-1)[:4]]]
        else:
            chiave_ril = comando.get("rilevatore")
            if chiave_ril not in MotoriCatalog.CHIAVI_RILEVATORI:
                raise ValueError("rilevatore sconosciuto: %r" % (chiave_ril,))
            rects = [[int(v) for v in np.asarray(r).reshape(-1)[:4]]
                     for r in _rilevatore_per(chiave_ril).extract(frame, is_bgr=True)]

        proposte = []
        if rects:
            allineatore = _allineatore_per(chiave_all, face_type)
            for rect, lmrks in zip(rects, allineatore.extract(frame, rects, None,
                                                              is_bgr=True)):
                punti = np.asarray(lmrks)
                # 68, non solo "coppie" -- stesso controllo di
                # _rispondi_riallinea/_rispondi_salva, ma qui una proposta
                # fuori forma non ferma l'intera richiesta: FANExtractor
                # puo' tornare None per un rect (ExtractManual._op_rileva
                # lo scarta con "if lmrks is None: continue"), e
                # np.asarray(None) e' 0-dimensionale -- senza guardia
                # cadrebbe silenziosa in .astype(float).tolist() producendo
                # un NaN al posto di 68 coppie. Si salta SOLO la proposta,
                # non tutta la risposta: le altre possono essere buone.
                if punti.ndim != 2 or punti.shape[1] != 2 or punti.shape[0] != 68:
                    continue
                proposte.append({"rect": rect,
                                 "source_landmarks": punti.astype(float).tolist()})
        return {"op": "rilevato", "id": ident, "proposte": proposte}
    except Exception as e:
        return _errore(ident, e)


def rispondi(comando, workdir):
    """Un comando -> una risposta. Non solleva mai: qualunque motivo di
    fallimento (file assente, non un JPEG DFL, comando malformato,
    operazione sconosciuta) diventa op='error', mai un'eccezione che
    fermerebbe chi chiama. Gira sotto redirect_stdout(stderr): niente di
    quello che succede qui dentro -- neanche una stampa di libreria che
    non ci aspettiamo -- puo' finire sul canale del protocollo."""
    with contextlib.redirect_stdout(sys.stderr):
        if not isinstance(comando, dict):
            return {"op": "error", "id": None, "codice": None,
                    "motivo": "comando non valido"}
        ident = comando.get("id")
        op = comando.get("op")
        if op == "open":
            return _rispondi_open(comando, ident, workdir)
        if op == "frame":
            return _rispondi_frame(comando, ident, workdir)
        if op == "riallinea":
            return _rispondi_riallinea(comando, ident, workdir)
        if op == "salva":
            return _rispondi_salva(comando, ident, workdir)
        if op == "rileva":
            return _rispondi_rileva(comando, ident, workdir)
        return {"op": "error", "id": ident, "codice": None,
                "motivo": "operazione sconosciuta"}


# Cinque minuti, la STESSA soglia di mainscripts/ExtractManual.py e per la
# stessa ragione: questo servizio importa DFLIMG, e con lui torch, e resta
# residente per l'intera sessione della GUI se nessuno lo ferma. La VRAM e'
# un vincolo di prima classe qui dentro -- le schede recenti ne hanno in
# media MENO di quelle che sostituiscono -- e un servizio dimenticato
# aperto e' precisamente la voce lasciata aperta dal ciclo faceset.
#
# La duplicazione da ExtractManual e' DELIBERATA e non un'estrazione
# mancata: i test di quel modulo inchiodano la forma interna del suo
# `_avvia_sorvegliante` (`args == (stato,)`), un modulo comune con un
# parametro in piu' li romperebbe, e CODEGUIDELINES chiede di non
# collassare il codice in astrazioni per risparmiare righe. La guardia
# contro la divergenza e'
# `test_le_due_copie_del_sorvegliante_non_divergono`.
TIMEOUT_INATTIVITA_S = 300.0
INTERVALLO_SORVEGLIANZA_S = 5.0


class Attivita:
    """Quando si e' visto l'ultimo comando, e se ne stiamo servendo uno.

    `ultimo` e' seminato con l'orologio vero nel costruttore, non con 0.0:
    un'Attivita passata a `sorveglia` senza un `tocca()` esplicito verrebbe
    altrimenti uccisa al primo risveglio -- INTERVALLO_SORVEGLIANZA_S dopo
    l'avvio -- invece che dopo il timeout vero."""

    def __init__(self):
        self.ultimo = time.time()
        self.occupato = False

    def tocca(self, adesso):
        self.ultimo = adesso


def sorveglia(stato, timeout_s=TIMEOUT_INATTIVITA_S, orologio=None,
              dormi=None, esci=None):
    """Esce dal processo dopo `timeout_s` senza comandi.

    Orologio, attesa e uscita sono iniettati: il test non avvia nessun
    thread e non aspetta nessun secondo vero. In produzione gira su un
    thread demone e `esci` e' `os._exit` -- **mai** `sys.exit`, che
    solleverebbe un SystemExit su un thread che non e' il principale e
    verrebbe inghiottito senza fermare niente: il processo resterebbe vivo
    e il difetto sarebbe indistinguibile dal non aver scritto il
    sorvegliante.

    Da qui non si stampa MAI: gira su un thread separato da quello che
    esegue `rispondi`, e il `redirect_stdout(sys.stderr)` di li' dentro e'
    un cambio GLOBALE di processo, non locale al thread -- una stampa da
    qui, nella finestra sbagliata, finirebbe sul canale del protocollo.
    """
    orologio = orologio or time.time
    dormi = dormi or time.sleep
    esci = esci or (lambda: os._exit(0))
    while True:
        dormi(INTERVALLO_SORVEGLIANZA_S)
        if not stato.occupato and orologio() - stato.ultimo >= timeout_s:
            esci()
            return


def _avvia_sorvegliante(stato):
    """Il thread del sorvegliante -- demone, o si comporterebbe come
    l'esatto contrario di se stesso: un thread non demone tiene in vita il
    processo Python finche' non esce da solo."""
    threading.Thread(target=sorveglia, args=(stato,), daemon=True).start()


def servi(entrata, uscita, workdir, stato=None):
    """Legge un comando JSON a riga da 'entrata', scrive una risposta JSON
    a riga su 'uscita', con un flush dopo ognuna -- il client legge una
    riga alla volta e aspetta: una risposta ferma nel buffer lo blocca per
    l'intero timeout, a ogni doppio click. Una riga non decodificabile
    come JSON riceve comunque una risposta di errore (id=None, non
    essendoci un comando da cui leggerlo): un client in attesa non deve
    restare appeso perche' la riga che ha mandato non si capiva."""
    for riga in entrata:
        riga = riga.strip()
        if not riga:
            continue
        try:
            comando = json.loads(riga)
        except ValueError as e:
            risposta = {"op": "error", "id": None, "codice": None,
                        "motivo": "riga non decodificabile come JSON: %s" % e}
            uscita.write(json.dumps(risposta) + "\n")
            uscita.flush()
            continue
        if stato is not None:
            stato.occupato = True
            stato.tocca(time.time())
        try:
            risposta = rispondi(comando, workdir)
            uscita.write(json.dumps(risposta) + "\n")
            uscita.flush()
        finally:
            # `finally`, non "dopo": se `rispondi` sollevasse -- non
            # dovrebbe mai, ma "non dovrebbe" non e' "non puo'" --
            # `occupato` resterebbe alto per sempre e il servizio
            # tornerebbe immortale, cioe' esattamente il difetto che
            # questo lavoro rimuove. E la bandiera si abbassa SOLO dopo
            # write+flush: chiude la finestra pericolosa per costruzione,
            # non per coincidenza aritmetica.
            if stato is not None:
                stato.tocca(time.time())
                stato.occupato = False


def main(workdir):
    stato = Attivita()
    stato.tocca(time.time())
    _avvia_sorvegliante(stato)
    servi(CanaleComandi.apri(), sys.stdout, Path(workdir), stato)
