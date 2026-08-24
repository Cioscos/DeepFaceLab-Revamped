"""Ricostruzione del rapporto per una cartella gia' estratta.

Ripiego, non passo normale: il rapporto lo scrive l'estrazione mentre gira
(mainscripts/ExtractReport.py). Qui si inferisce dai file gia' presenti,
senza rieseguire il rilevamento -- che costerebbe quanto una nuova
estrazione per un dato che si puo' dedurre.

La posa va stimata nello spazio ALLINEATO: DFLJPG.get_landmarks() e' gia'
in quello spazio, non in quello del frame -- get_source_landmarks() darebbe
una posa sbagliata in silenzio. Ma get_landmarks() torna i punti nella
dimensione con cui il volto e' stato effettivamente salvato su disco (512
di default, 768 per head), non 256: LandmarksProcessor.estimate_pitch_yaw_roll
assume col suo default size=256 una camera tarata per quella dimensione, e
chiamarla direttamente sui landmark cosi' come sono da' una posa sbagliata
per un motivo diverso (misurato: ~14 gradi di pitch, ~28 di yaw a parita' di
geometria). La correzione e' la STESSA canonicalizzazione di
ExtractorLib.voce_da_data -- get_transform_mat(lmrks, 256, FaceType.FULL)
poi transform_points, poi estimate_pitch_yaw_roll col default -- applicata
qui ai landmark GIA' allineati invece che a quelli grezzi di frame. E'
scala-invariante (get_transform_mat lavora sulla geometria relativa dei
punti, non su una dimensione presunta altrove), quindi la posa risultante e'
la stessa qualunque sia la dimensione con cui il volto e' stato allineato su
disco, ed e' confrontabile con quella scritta dall'altro produttore dello
stesso rapporto (voce_da_data) -- necessario perche' la posa e' un dato di
filtro della pagina, che deve valere a prescindere da face type e image size
scelti dall'utente. Il rettangolo e il lato restano nello spazio del frame:
quelli vengono giustamente da get_source_rect().

Lo stat() sta DENTRO il ciclo di os.scandir: un list(...) innocuo in cima
costa 3x senza cambiare un solo risultato, misurato nel ciclo faceset. Cio'
che si accumula in una lista sono i soli PERCORSI, e `is_file()` legge il
tipo che il dirent porta gia' -- non e' lo stat() per file di quella
trappola. Serve il totale: senza, non c'e' barra di avanzamento, e questo e'
il ripiego che OGNI progetto preesistente incontra per primo (misurato: 655
frame in 6,58 s monoprocesso, cioe' ~1 minuto a 5 500 frame e ~8 minuti a
50 000 -- con pila di avanzamento vuota, console vuota e pagina che sembra
bloccata).

Le barre passano da `io`, come nel gemello mainscripts/FacesetIndex.py: e'
il canale che la pila di avanzamento della GUI legge (DFL_PROGRESS_FILE), e
un `print()` non lo raggiungerebbe.

La luminanza non si conosce qui (il frame non viene ridecodificato: e'
proprio quello che il ripiego evita di rifare) -- si scrive
luminanza=None, non 0.0, perche' 0.0 e' sotto qualunque soglia di "scuro"
la pagina usera' e classificherebbe come scuro ogni frame ricostruito.
"""
import os
from pathlib import Path

import numpy as np

from core.interact import interact as io
from mainscripts import ExtractReport

ESTENSIONI_FRAME = (".png", ".jpg", ".jpeg")


def _percorsi(cartella, estensioni):
    """I percorsi dei file utili. Serve la LISTA e non l'iteratore perche'
    una barra di avanzamento vuole il totale prima di cominciare."""
    cartella = Path(cartella)
    if not cartella.is_dir():
        return []
    fuori = []
    with os.scandir(str(cartella)) as voci:
        for v in voci:
            if v.is_file() and v.name.lower().endswith(estensioni):
                fuori.append(v.path)
    return fuori


def _volto_da_dfl(percorso):
    """(source_filename, voce-volto) da un JPEG allineato, o None se il file
    non e' un DFLJPG o non dichiara il frame da cui viene.

    Estratta da `_volti_per_frame` perche' la sessione manuale ne ha
    bisogno per un frame solo: due copie di questa aritmetica sono il modo
    in cui una delle due smette di calcolare la stessa posa."""
    from DFLIMG import DFLJPG
    from facelib import FaceType, LandmarksProcessor
    dfl = DFLJPG.load(percorso)
    if dfl is None:
        return None
    sorgente = dfl.get_source_filename()
    if not sorgente:
        return None
    rect = dfl.get_source_rect()
    lmrks = dfl.get_landmarks()
    posa = [0.0, 0.0, 0.0]
    lato = 0
    if rect is not None:
        l, t, r, b = [int(x) for x in np.asarray(rect).reshape(-1)[:4]]
        lato = max(r - l, b - t)
        rect = [l, t, r, b]
    else:
        rect = [0, 0, 0, 0]
    lmrks = np.asarray(lmrks)
    if lmrks.ndim == 2 and lmrks.shape[0] == 68:
        mat = LandmarksProcessor.get_transform_mat(lmrks, 256, FaceType.FULL)
        allineati = LandmarksProcessor.transform_points(lmrks, mat)
        posa = [float(p) for p in
                LandmarksProcessor.estimate_pitch_yaw_roll(allineati)]
    return sorgente, {"rect": rect, "posa": posa, "lato": lato}


def _volti_per_frame(aligned_dir):
    per_frame = {}
    percorsi = _percorsi(aligned_dir, (".jpg",))
    for percorso in io.progress_bar_generator(percorsi, "Reading aligned faces"):
        esito = _volto_da_dfl(percorso)
        if esito is None:
            continue
        sorgente, voce_volto = esito
        per_frame.setdefault(sorgente, []).append(voce_volto)
    return per_frame


def percorsi_di_un_frame(aligned_dir, nome_frame):
    """I percorsi degli allineati che dichiarano `nome_frame` come
    sorgente, ordinati per nome.

    Il glob sullo stelo restringe la lettura ai pochi file plausibili --
    l'estrazione nomina `<stelo>_<indice>.jpg` -- ma non decide:
    «00001_2.png» produce «00001_2_0.jpg», che lo stelo di «00001.png»
    pesca. Chi decide e' il `source_filename` scritto dentro il JPEG.

    E' la sola sede di questa regola: la usano `volti_di_un_frame` qui
    sotto e l'operazione `frame` del servizio di dettaglio
    (mainscripts/FacesetDetail.py). Due copie sono il modo in cui una
    delle due smette di accorgersi dei .png numerati.
    """
    from DFLIMG import DFLJPG
    if not aligned_dir or not nome_frame:
        return []
    cartella = Path(aligned_dir)
    if not cartella.is_dir():
        return []
    fuori = []
    for percorso in sorted(cartella.glob("%s_*.jpg" % Path(nome_frame).stem)):
        dfl = DFLJPG.load(str(percorso))
        if dfl is not None and dfl.get_source_filename() == nome_frame:
            fuori.append(percorso)
    return fuori


def volti_di_un_frame(aligned_dir, nome_frame):
    """I volti gia' su disco per UN fotogramma, nella stessa forma che
    `_volti_per_frame` da' a tutti.

    Legge ogni JPEG due volte -- una in `percorsi_di_un_frame` per il
    `source_filename`, una qui per la posa -- ed e' accettato: sono uno o
    due file, ~5,8 ms l'uno, e la chiamata avviene una volta sola quando
    la sessione manuale entra su un fotogramma. Il prezzo compra la
    regola scritta in un posto solo.
    """
    volti = []
    for percorso in percorsi_di_un_frame(aligned_dir, nome_frame):
        esito = _volto_da_dfl(str(percorso))
        if esito is not None:
            volti.append(esito[1])
    return volti


def ricostruisci(input_dir, aligned_dir, cache_dir):
    per_frame = _volti_per_frame(aligned_dir)
    volti = sum(len(v) for v in per_frame.values())
    scritte = 0
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    percorsi = _percorsi(input_dir, ESTENSIONI_FRAME)
    with ExtractReport.Scrittore(cache_dir) as scrittore:
        for percorso in io.progress_bar_generator(percorsi, "Indexing frames"):
            percorso = Path(percorso)
            scrittore.scrivi(ExtractReport.voce(
                percorso, volti=per_frame.get(percorso.name, []),
                luminanza=None, stato=ExtractReport.STATO_AUTOMATICO))
            scritte += 1
    io.log_info('Report rebuilt: %d frame(s), %d face(s) already extracted.'
                % (scritte, volti))
    return scritte
