"""Il servizio di estrazione manuale: un frame alla volta.

Stessa forma di mainscripts/FacesetDetail.py, e per le stesse ragioni. Due
cautele che quel file ha pagato e che qui valgono identiche:

- `rispondi` gira sotto contextlib.redirect_stdout(sys.stderr), perche' una
  qualunque stampa di libreria sul canale del protocollo desincronizzerebbe
  il parser del client;
- il raster viene annunciato DOPO os.replace, mai prima, cosi' un file
  annunciato esiste sempre.

Nessuno stato attraversa una chiamata all'altra: ogni operazione e'
autosufficiente nel proprio comando. Eccezione: `rileva` costruisce il
rilevatore e l'allineatore che il comando nomina alla prima richiesta e li
tiene vivi in `_RILEVATORI`/`_ALLINEATORI` -- una voce di cache per motore,
non una sola coppia, da quando la barra della pagina lascia sceglierli --
per tutto il processo, perche' caricarli di nuovo a ogni frame costerebbe
secondi dove serve rispondere in millisecondi. `libera` e' il suo
contrario, ed e' un'operazione a se' proprio perche' deve poter agire
anche quando non c'e' nessun fotogramma su cui rilevare. `landmark`
invece resta geometria pura via `landmarks_da_vettore`, senza toccare un
pixel. La tela
Qt possiede l'interazione: qui non c'e' nessun loop di eventi, solo un
comando e una risposta.

Proprio perche' i motori restano vivi, il processo non puo' restare in
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
# nemmeno come parametro) e l'allineatore dipende dal motore E dal
# PAVIMENTO 2D/3D, cioe' da un booleano -- non dalla stringa face_type per
# intero. Chiavare su face_type costruiva fino a otto coppie duplicate (un
# FaceType per voce, facelib/FaceType.py), quasi tutte copie esatte:
# misurato, un solo rilevatore costruito due volte per due face type
# diversi raddoppiava da 190 MB a 381 MB allocati. La VRAM qui e' un
# vincolo di prima classe, quindi niente ricostruzioni gratis.
#
# Il MOTORE sta nella chiave di entrambe, e non e' un dettaglio: da quando
# la pagina lascia scegliere rilevatore e allineatore, una cache senza il
# motore nella chiave tornerebbe in SILENZIO quello gia' costruito --
# nessun errore, nessun log, gli stessi landmark di prima e un utente che
# conclude che il motore nuovo non serve a niente. `fan-2d` e `pipnet-68`
# risolvono entrambi in landmarks_3D=False, quindi con la sola chiave
# booleana di prima sarebbero stati lo stesso oggetto.
_RILEVATORI = {}   # chiave: la chiave del rilevatore nel registro
_ALLINEATORI = {}  # chiave: (chiave dell'allineatore, landmarks_3D o None)


def _rilevatore(chiave):
    """Il rilevatore di questa chiave, costruito alla prima richiesta."""
    if chiave not in _RILEVATORI:
        from facelib import motori
        _RILEVATORI[chiave] = motori.costruisci_rilevatore(chiave)
    return _RILEVATORI[chiave]


def _chiave_allineatore(chiave, face_type):
    """(chiave del motore, landmarks_3D risolto) -- la voce di cache di un
    allineatore, senza costruirlo.

    La soglia 2D/3D si legge da `facelib.motori.landmarks_3D_per`, NON si
    riscrive qui: e' la stessa funzione che `costruisci_allineatore` usa
    per il parametro vero passato a FANExtractor. Se questa chiave e quel
    parametro venissero da due formule copiate a mano, il giorno che una
    delle due cambia la chiave smetterebbe di corrispondere al
    comportamento reale -- il servizio tornerebbe l'allineatore sbagliato
    in silenzio, senza errore ne' log.

    Per un motore che il booleano NON lo consuma (`pipnet-68`) la chiave
    non lo porta affatto -- `None` al suo posto -- o `('pipnet-68', False)`
    e `('pipnet-68', True)` sarebbero due voci e due costruzioni dello
    stesso identico oggetto: la duplicazione di VRAM che questa cache
    esiste per non avere, in scala minore. Anche "lo consuma o no" viene
    da `facelib.motori`, mai da un elenco di nomi scritto qui.
    """
    from facelib import motori
    if not motori.consuma_landmarks_3D(chiave):
        return (chiave, None)
    return (chiave, motori.landmarks_3D_per(chiave, face_type))


def _allineatore(chiave, face_type):
    """L'allineatore di questa chiave per questo face type.

    Il face_type entra nella cache SOLO attraverso il booleano
    landmarks_3D, mai grezzo: whole_face e half_face risolvono entrambi in
    landmarks_3D=False, quindi allo stesso allineatore. Il prossimo che
    legge questa funzione e la vede "corta" non deve tornare a chiavare su
    face_type -- e' la causa esatta del raddoppio di VRAM misurato sopra.
    """
    from facelib import motori
    voce = _chiave_allineatore(chiave, face_type)
    if voce not in _ALLINEATORI:
        _ALLINEATORI[voce] = motori.costruisci_allineatore(chiave, face_type)
    return _ALLINEATORI[voce]


def _libera_tranne(chiave_rilevatore, voce_allineatore):
    """Lascia andare ogni motore che non sia uno dei due correnti.

    Lasciare andare i riferimenti non basta: la memoria resterebbe
    all'allocatore di torch e chi guarda la VRAM non vedrebbe cambiare
    niente, che e' tutto il punto della spunta. Quindi `empty_cache()`, e
    un `gc.collect()` prima -- un modulo torch puo' partecipare a un ciclo
    di riferimenti, e finche' il ciclo non e' raccolto i tensori restano
    allocati.

    Non si tocca nulla quando non c'e' nulla da liberare: `empty_cache()`
    sincronizza col device, e pagarlo a ogni fotogramma di una sessione
    che non ha cambiato motore sarebbe un costo per niente.

    Torna QUANTI motori ha lasciato andare: e' il solo modo che il client
    ha di distinguere "liberato" da "non c'era niente da liberare", e la
    risposta dell'operazione `libera` lo porta.
    """
    da_togliere_r = [k for k in _RILEVATORI if k != chiave_rilevatore]
    da_togliere_a = [k for k in _ALLINEATORI if k != voce_allineatore]
    if not da_togliere_r and not da_togliere_a:
        return 0
    for k in da_togliere_r:
        del _RILEVATORI[k]
    for k in da_togliere_a:
        del _ALLINEATORI[k]
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(da_togliere_r) + len(da_togliere_a)


def _motori(face_type, chiave_rilevatore=None, chiave_allineatore=None,
            tieni_in_memoria=True, serve_rilevatore=True):
    """(rilevatore, allineatore) per questo face type e queste chiavi,
    costruiti alla prima richiesta che li tocca -- vedi _rilevatore e
    _allineatore per il perche' sono due cache separate e non una.

    `serve_rilevatore=False` torna `None` al posto del rilevatore e non lo
    costruisce affatto: e' la richiesta che porta gia' il rettangolo e non
    chiede il secondo passaggio, dove il rilevatore non verrebbe mai usato.
    Costruirlo lo stesso significava che un motore non disponibile (pesi
    assenti, VRAM esaurita) toglieva all'utente anche i landmark che
    l'allineatore gli avrebbe dato da solo.

    Chiavi assenti = i default del registro: un comando `rileva` che non
    nomina i motori si comporta come prima di questa funzione.

    `tieni_in_memoria=False` (la spunta tolta nella barra) lascia vivi
    SOLO i due correnti, e li libera SUBITO -- non alla prossima scelta.
    Non e' "ricostruisci a ogni fotogramma", che renderebbe la sessione
    inusabile: i due correnti restano in cache come sempre.

    Cosa la spunta NON promette, e va detto perche' e' una conseguenza
    diretta dell'ordine qui sotto: si costruisce PRIMA e si libera DOPO,
    quindi nell'ISTANTE di un cambio di motore i due convivono in VRAM.
    La spunta abbassa lo stato stazionario, non il picco -- chi ha la
    memoria al limite proprio in quel momento non e' aiutato. L'ordine
    resta questo di proposito: liberare per primo e poi fallire la
    costruzione (pesi mancanti, memoria esaurita) lascerebbe l'utente
    senza NIENTE in cache e con la sessione ferma, che e' un guasto
    peggiore di un picco. Chi vuole il picco basso toglie la spunta PRIMA
    di cambiare motore: l'operazione `libera` non costruisce nulla.
    """
    from mainscripts import MotoriCatalog
    from facelib import FaceType
    chiave_rilevatore = chiave_rilevatore or MotoriCatalog.DEFAULT_RILEVATORE
    chiave_allineatore = chiave_allineatore or MotoriCatalog.DEFAULT_ALLINEATORE
    tipo = FaceType.fromString(face_type)
    # Prima si costruisce, poi si libera: liberare per primo butterebbe via
    # un motore che la riga dopo potrebbe rimettere in piedi da capo.
    rilevatore = _rilevatore(chiave_rilevatore) if serve_rilevatore else None
    allineatore = _allineatore(chiave_allineatore, tipo)
    if not tieni_in_memoria:
        _libera_tranne(chiave_rilevatore,
                       _chiave_allineatore(chiave_allineatore, tipo))
    return rilevatore, allineatore


def _tieni_in_memoria(comando):
    """La politica di memoria del comando: assente = True, il comportamento
    di sempre.

    "Assente" e "null esplicito" non sono la stessa cosa per un `.get` con
    default -- `{"tieni_in_memoria": null}` lo salta e cadrebbe su
    `bool(None)`, cioe' su "libera", che e' l'opposto del default. Nessun
    client odierno manda quel null, ma il protocollo e' JSON e chi lo
    scrivera' domani non legge questa funzione.
    """
    valore = comando.get("tieni_in_memoria")
    return True if valore is None else bool(valore)


def _op_libera(comando):
    """Libera ogni motore che non sia la coppia nominata, e basta.

    Esiste perche' la promessa della spunta ("gli altri sono liberati
    subito") viaggiava sul comando `rileva`, e `rileva` ha bisogno di un
    fotogramma corrente: senza -- un filtro della pellicola che non lascia
    frame, il momento subito dopo l'ingresso in sessione -- la pagina non
    mandava NIENTE e la VRAM restava occupata senza un segnale. Proprio lo
    stato in cui uno toglie la spunta per fare posto a un training.

    Non costruisce mai niente: e' la sola operazione con cui si puo'
    abbassare anche il PICCO, perche' `_motori` costruisce prima e libera
    dopo. Con le cache gia' vuote non importa nemmeno `facelib` (che
    porterebbe torch) -- non c'e' chiave da normalizzare se non c'e' niente
    da normalizzare.
    """
    liberati = 0
    if _RILEVATORI or _ALLINEATORI:
        from mainscripts import MotoriCatalog
        from facelib import FaceType
        chiave_r = comando.get("rilevatore") or MotoriCatalog.DEFAULT_RILEVATORE
        chiave_a = comando.get("allineatore") or MotoriCatalog.DEFAULT_ALLINEATORE
        tipo = FaceType.fromString(str(comando.get("face_type") or "whole_face"))
        liberati = _libera_tranne(chiave_r, _chiave_allineatore(chiave_a, tipo))
    return {"op": "libera", "id": comando.get("id"), "liberati": int(liberati)}


def _op_rileva(comando):
    """I volti del frame, o dell'unico rettangolo dato.

    Due strade, ed e' la stessa distinzione della finestra `cv2`:

    - `rect` assente: il rilevatore cerca da solo. E' cio' che la GUI non
      ha mai avuto, e la ragione per cui la sessione manuale finiva sempre
      nel ripiego a template.
    - `rect` dato: e' il rettangolo che l'utente sta muovendo, e il
      rilevatore si salta -- non lo si chiama E non lo si costruisce.
      Cercare da capo a ogni pressione di freccia butterebbe via proprio
      il rettangolo che sta scegliendo.

    `accurato` passa il RILEVATORE all'allineatore come secondo passaggio,
    esattamente come `landmarks_accurate` in Extractor.py: e' il tasto `a`
    della finestra `cv2`, piu' preciso e piu' lento. E' anche la ragione
    per cui "salta il rilevatore" non si puo' scrivere come "mai col
    rect": col secondo passaggio acceso quel motore serve davvero.

    Nessun volto NON e' un errore: 206 frame su 983 senza volto e' il caso
    normale di questa pagina.
    """
    from core.cv2ex import cv2_imread

    percorso = comando.get("path")
    immagine = cv2_imread(str(percorso))
    if immagine is None:
        raise ValueError("frame non leggibile: %s" % percorso)
    face_type = str(comando.get("face_type") or "whole_face")
    # Le chiavi dei motori arrivano dal comando come `face_type`: nessun
    # canale nuovo. Assenti, restano i default del registro -- una chiave
    # sconosciuta invece solleva (MotoriCatalog.rilevatore) e `rispondi` la
    # trasforma in una risposta "error", perche' un ripiego silenzioso sul
    # default produrrebbe un faceset misto senza dirlo.
    # `rect` e `accurato` si leggono PRIMA dei motori: sono cio' che decide
    # se il rilevatore serve a questa richiesta. Costruirlo comunque -- come
    # faceva la prima versione, che pure prometteva qui sopra di saltarlo --
    # fa dipendere il rettangolo tracciato a mano da un motore che quella
    # richiesta non tocca mai.
    rect = comando.get("rect")
    accurato = bool(comando.get("accurato"))
    rilevatore, allineatore = _motori(face_type,
                                      comando.get("rilevatore"),
                                      comando.get("allineatore"),
                                      _tieni_in_memoria(comando),
                                      serve_rilevatore=rect is None or accurato)

    if rect is None:
        rects = [tuple(int(v) for v in r) for r in rilevatore.extract(immagine, is_bgr=True)]
    else:
        rects = [tuple(int(v) for v in rect)]
    if not rects:
        return {"op": "rileva", "id": comando.get("id"), "volti": []}

    secondo = rilevatore if accurato else None
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
            if op == "libera":
                return _op_libera(comando)
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
