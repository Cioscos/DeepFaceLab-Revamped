"""Il servizio di estrazione manuale: un frame alla volta.

Stessa forma di mainscripts/FacesetDetail.py, e per le stesse ragioni. Due
cautele che quel file ha pagato e che qui valgono identiche:

- `rispondi` gira sotto contextlib.redirect_stdout(sys.stderr), perche' una
  qualunque stampa di libreria sul canale del protocollo desincronizzerebbe
  il parser del client;
- il raster viene annunciato DOPO os.replace, mai prima, cosi' un file
  annunciato esiste sempre.

Nessuno stato attraversa una chiamata all'altra: ogni operazione e'
autosufficiente nel proprio comando. Eccezione: `rileva` costruisce S3FD e
FAN alla prima richiesta e li tiene vivi in `_RILEVATORE`/`_ALLINEATORI`
per tutto il processo, perche' caricarli di nuovo a ogni frame costerebbe
secondi dove serve rispondere in millisecondi -- `landmark` invece resta
geometria pura via `landmarks_da_vettore`, senza toccare un pixel. La tela
Qt possiede l'interazione: qui non c'e' nessun loop di eventi, solo un
comando e una risposta.

Proprio perche' S3FD e FAN restano vivi, il processo non puo' restare in
piedi all'infinito: `sorveglia` lo chiude da solo dopo
TIMEOUT_INATTIVITA_S senza comandi. E' il precedente lasciato aperto su
gui/faceset/dettaglio.py (il cui `ferma()` non ha chiamanti e non ha
nessun timeout) chiuso qui invece che rimandato.
"""
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


def _scrivi_e_annuncia(byte_grezzi, nome, workdir):
    """Scrive su un file temporaneo, rinomina in modo atomico e torna il
    nome finale -- solo a scrittura conclusa, mai prima, cosi' un file
    annunciato esiste sempre."""
    finale = Path(workdir) / nome
    provvisorio = Path(workdir) / (nome + ".tmp")
    provvisorio.write_bytes(byte_grezzi)
    os.replace(str(provvisorio), str(finale))
    return nome


def _op_frame(comando, workdir):
    import cv2
    from core.cv2ex import cv2_imread
    percorso = comando.get("path")
    immagine = cv2_imread(str(percorso))
    if immagine is None:
        raise ValueError("frame non leggibile: %s" % percorso)
    ok, codificato = cv2.imencode(".jpg", immagine,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("codifica del raster fallita")
    nome = _scrivi_e_annuncia(codificato.tobytes(),
                              "%s_frame.jpg" % comando.get("id"), workdir)
    h, w = immagine.shape[:2]
    return {"op": "frame", "id": comando.get("id"), "raster": nome,
            "shape": [int(h), int(w)]}


def _op_landmark(comando):
    from mainscripts import ExtractorLib
    rect, lmrks = ExtractorLib.landmarks_da_vettore(comando.get("centro"),
                                                    comando.get("punta"))
    if lmrks is None:
        raise ValueError("vettore di lunghezza nulla: nessun landmark")
    return {"op": "landmark", "id": comando.get("id"),
            "rect": [int(v) for v in rect],
            "landmarks": np.asarray(lmrks).tolist()}


# I motori si costruiscono UNA volta per processo e restano vivi: sono
# l'unica ragione per cui questo servizio e' persistente invece che un
# comando per frame. Le chiavi vengono dal registro
# (mainscripts/MotoriCatalog.py), mai da una stringa scritta qui.
#
# DUE cache separate, non una chiavata sul face_type grezzo: il rilevatore
# non dipende affatto dal face type (costruisci_rilevatore non lo prende
# nemmeno come parametro) e l'allineatore dipende solo dal PAVIMENTO 2D/3D,
# cioe' da un booleano -- non dalla stringa face_type per intero. Chiavare
# su face_type costruiva fino a otto coppie S3FD+FAN duplicate (un
# FaceType per voce, facelib/FaceType.py), quasi tutte copie esatte:
# misurato, un solo rilevatore costruito due volte per due face type
# diversi raddoppiava da 190 MB a 381 MB allocati. La VRAM qui e' un
# vincolo di prima classe, quindi niente ricostruzioni gratis.
_RILEVATORE = None
_ALLINEATORI = {}  # chiave: bool(landmarks_3D) gia' risolto, MAI face_type


def _rilevatore():
    """L'unico rilevatore del processo: non ha chiave perche' non varia
    mai con l'input di questa pagina."""
    global _RILEVATORE
    if _RILEVATORE is None:
        from facelib import motori
        from mainscripts import MotoriCatalog
        _RILEVATORE = motori.costruisci_rilevatore(MotoriCatalog.DEFAULT_RILEVATORE)
    return _RILEVATORE


def _allineatore(face_type):
    """L'allineatore per questo face type, cache chiavata sul booleano
    landmarks_3D gia' risolto -- NON sul face_type grezzo.

    whole_face e half_face risolvono entrambi in landmarks_3D=False,
    quindi allo stesso allineatore. Il prossimo che legge questa funzione
    e la vede "corta" non deve tornare a chiavare su face_type -- e' la
    causa esatta del raddoppio di VRAM misurato sopra.

    La soglia 2D/3D si legge da `facelib.motori.landmarks_3D_per`, NON si
    riscrive qui: e' la stessa funzione che `costruisci_allineatore` usa
    per il parametro vero passato a FANExtractor. Se questa chiave e quel
    parametro venissero da due formule copiate a mano, il giorno che una
    delle due cambia la chiave smetterebbe di corrispondere al
    comportamento reale -- il servizio tornerebbe l'allineatore sbagliato
    in silenzio, senza errore ne' log.
    """
    from facelib import motori
    from facelib import FaceType
    from mainscripts import MotoriCatalog
    tipo = FaceType.fromString(face_type)
    landmarks_3D = motori.landmarks_3D_per(MotoriCatalog.DEFAULT_ALLINEATORE, tipo)
    if landmarks_3D not in _ALLINEATORI:
        _ALLINEATORI[landmarks_3D] = motori.costruisci_allineatore(
            MotoriCatalog.DEFAULT_ALLINEATORE, tipo)
    return _ALLINEATORI[landmarks_3D]


def _motori(face_type):
    """(rilevatore, allineatore) per questo face type, costruiti alla
    prima richiesta che li tocca -- vedi _rilevatore/_allineatore per il
    perche' sono due cache separate e non una."""
    return _rilevatore(), _allineatore(face_type)


def _op_rileva(comando):
    """I volti del frame, o dell'unico rettangolo dato.

    Due strade, ed e' la stessa distinzione della finestra `cv2`:

    - `rect` assente: il rilevatore cerca da solo. E' cio' che la GUI non
      ha mai avuto, e la ragione per cui la sessione manuale finiva sempre
      nel ripiego a template.
    - `rect` dato: e' il rettangolo che l'utente sta muovendo, e il
      rilevatore si salta. Cercare da capo a ogni pressione di freccia
      butterebbe via proprio il rettangolo che sta scegliendo.

    `accurato` passa il RILEVATORE all'allineatore come secondo passaggio,
    esattamente come `landmarks_accurate` in Extractor.py: e' il tasto `a`
    della finestra `cv2`, piu' preciso e piu' lento.

    Nessun volto NON e' un errore: 206 frame su 983 senza volto e' il caso
    normale di questa pagina.
    """
    from core.cv2ex import cv2_imread

    percorso = comando.get("path")
    immagine = cv2_imread(str(percorso))
    if immagine is None:
        raise ValueError("frame non leggibile: %s" % percorso)
    face_type = str(comando.get("face_type") or "whole_face")
    rilevatore, allineatore = _motori(face_type)

    rect = comando.get("rect")
    if rect is None:
        rects = [tuple(int(v) for v in r) for r in rilevatore.extract(immagine, is_bgr=True)]
    else:
        rects = [tuple(int(v) for v in rect)]
    if not rects:
        return {"op": "rileva", "id": comando.get("id"), "volti": []}

    secondo = rilevatore if comando.get("accurato") else None
    landmarks = allineatore.extract(immagine, rects, secondo, is_bgr=True)
    volti = []
    for r, lmrks in zip(rects, landmarks):
        if lmrks is None:
            continue
        volti.append({"rect": [int(v) for v in r],
                      "landmarks": np.asarray(lmrks).tolist()})
    return {"op": "rileva", "id": comando.get("id"), "volti": volti}


def _op_salva(comando):
    from core.cv2ex import cv2_imread
    from facelib import FaceType
    from mainscripts import ExtractorLib

    percorso = Path(comando.get("path"))
    immagine = cv2_imread(str(percorso))
    if immagine is None:
        raise ValueError("frame non leggibile: %s" % percorso)
    uscita = Path(comando.get("output_dir"))
    uscita.mkdir(parents=True, exist_ok=True)
    nome = "%s_%d.jpg" % (percorso.stem, int(comando.get("face_idx", 0)))
    scritto = ExtractorLib.salva_volto(
        immagine=immagine,
        rect=comando.get("rect"),
        image_landmarks=np.asarray(comando.get("landmarks"), dtype=np.float32),
        face_type=FaceType.fromString(comando.get("face_type")),
        image_size=int(comando.get("image_size")),
        jpeg_quality=int(comando.get("jpeg_quality")),
        output_filepath=uscita / nome,
        source_filename=percorso.name,
        manuale=True)
    if scritto is None:
        # Il rettangolo l'ha tracciato l'utente: salva_volto non lo scarta
        # per l'area (manuale=True lo esclude), ma puo' comunque tornare
        # None -- quel caso non deve mai diventare un AttributeError su
        # Path(None).name.
        raise ValueError("volto scartato: nessun file scritto")
    return {"op": "salvato", "id": comando.get("id"), "file": Path(scritto).name}


def rispondi(comando, workdir):
    """Un comando -> una risposta. Non solleva mai."""
    with contextlib.redirect_stdout(sys.stderr):
        if not isinstance(comando, dict):
            return {"op": "error", "id": None, "motivo": "comando non valido"}
        ident = comando.get("id")
        op = comando.get("op")
        try:
            if op == "frame":
                return _op_frame(comando, workdir)
            if op == "landmark":
                return _op_landmark(comando)
            if op == "rileva":
                return _op_rileva(comando)
            if op == "salva":
                return _op_salva(comando)
            raise ValueError("operazione sconosciuta: %s" % op)
        except Exception as e:
            return {"op": "error", "id": ident, "motivo": str(e)}


# Cinque minuti: abbastanza per studiare un fotogramma senza vedersi morire
# il servizio sotto le mani, abbastanza poco perche' due modelli non restino
# in VRAM per l'intera sessione di lavoro dell'utente. La VRAM qui e' un
# vincolo di prima classe -- le schede recenti ne hanno in media MENO di
# quelle che sostituiscono -- e un servizio dimenticato aperto e' proprio
# la voce di registro lasciata aperta dal ciclo faceset.
TIMEOUT_INATTIVITA_S = 300.0
INTERVALLO_SORVEGLIANZA_S = 5.0


class Attivita:
    """Quando si e' visto l'ultimo comando, e se ne stiamo servendo uno.

    `ultimo` e' seminato con l'orologio vero nel costruttore, non con 0.0:
    un'Attivita passata a `sorveglia` senza un `tocca()` esplicito verrebbe
    altrimenti uccisa al primo risveglio (`INTERVALLO_SORVEGLIANZA_S` dopo
    l'avvio) invece che dopo il timeout vero. `main` chiama comunque
    `tocca` súbito dopo aver costruito lo stato, ma e' il prossimo
    chiamante che non lo facesse a pagare l'innesco a scoppio -- meglio
    non lasciarlo li'."""

    def __init__(self):
        self.ultimo = time.time()
        self.occupato = False

    def tocca(self, adesso):
        self.ultimo = adesso


def sorveglia(stato, timeout_s=TIMEOUT_INATTIVITA_S, orologio=None,
              dormi=None, esci=None):
    """Esce dal processo dopo `timeout_s` senza comandi.

    Funzione pura con orologio, attesa e uscita iniettati: il test non
    avvia nessun thread e non aspetta nessun secondo vero. In produzione
    gira su un thread demone e `esci` e' os._exit -- NON sys.exit, che
    solleverebbe SystemExit su un thread che non e' il principale e
    verrebbe inghiottito senza fermare niente: il processo resterebbe
    vivo e il difetto sarebbe indistinguibile dal non aver scritto il
    sorvegliante.

    `timeout_s=TIMEOUT_INATTIVITA_S` lega il DEFAULT al momento in cui
    questa funzione viene DEFINITA: riassegnare la costante di modulo a
    runtime non cambia il timeout del thread di produzione, gia' avviato
    con quel valore congelato dentro l'`args` del thread. Nessun chiamante
    lo fa oggi.

    Dal sorvegliante non si stampa MAI: gira su un thread separato da
    quello che esegue `rispondi`, e `contextlib.redirect_stdout(sys.stderr)`
    li' dentro e' un cambio GLOBALE di processo, non locale al thread --
    una stampa da qui, nella finestra sbagliata, finirebbe su stdout
    invece che su stderr e desincronizzerebbe il parser del client.
    """
    orologio = orologio or time.time
    dormi = dormi or time.sleep
    esci = esci or (lambda: os._exit(0))
    while True:
        dormi(INTERVALLO_SORVEGLIANZA_S)
        if not stato.occupato and orologio() - stato.ultimo >= timeout_s:
            esci()
            return


def servi(entrata, uscita, workdir, stato=None):
    for riga in entrata:
        riga = riga.strip()
        if not riga:
            continue
        try:
            comando = json.loads(riga)
        except ValueError as e:
            risposta = {"op": "error", "id": None,
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
            # `finally`, non solo "dopo": se `rispondi` sollevasse (non
            # dovrebbe mai, il suo stesso docstring lo garantisce, ma
            # "non dovrebbe" non e' "non puo'") `occupato` resterebbe alto
            # per sempre e il servizio diventerebbe immortale -- esattamente
            # il difetto che questo task rimuove. E la bandiera si abbassa
            # SOLO dopo write+flush, non prima: chiude la finestra pericolosa
            # per costruzione, non per coincidenza aritmetica (stato.tocca()
            # rimetterebbe comunque margine, ma non e' una ragione per
            # lasciarla aperta un istante in piu' del necessario).
            if stato is not None:
                stato.tocca(time.time())
                stato.occupato = False


def _avvia_sorvegliante(stato):
    """Il thread del sorvegliante -- demone, o si comporterebbe come
    l'esatto contrario di se stesso: un thread non demone tiene in vita
    il processo Python finche' non esce da solo, quindi un `sorveglia`
    non demone impedirebbe proprio la chiusura che deve causare.

    Estratta da `main` per essere testabile senza un vero stdin/stdout:
    senza questa funzione un test avrebbe
    dovuto chiamare `main` per intero, bloccandosi sulla lettura di
    `sys.stdin` vero."""
    threading.Thread(target=sorveglia, args=(stato,), daemon=True).start()


def main(workdir):
    stato = Attivita()
    stato.tocca(time.time())
    _avvia_sorvegliante(stato)
    servi(sys.stdin, sys.stdout, Path(workdir), stato)
