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


def _volti_per_frame(aligned_dir):
    from DFLIMG import DFLJPG
    from facelib import FaceType, LandmarksProcessor
    per_frame = {}
    percorsi = _percorsi(aligned_dir, (".jpg",))
    for percorso in io.progress_bar_generator(percorsi, "Reading aligned faces"):
        dfl = DFLJPG.load(percorso)
        if dfl is None:
            continue
        sorgente = dfl.get_source_filename()
        if not sorgente:
            continue
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
        per_frame.setdefault(sorgente, []).append(
            {"rect": rect, "posa": posa, "lato": lato})
    return per_frame


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
