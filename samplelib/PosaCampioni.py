"""La posa (pitch, yaw, roll) dei campioni, calcolata una volta sola.

Prima di questo modulo ogni generatore con `uniform_yaw_distribution`
rifaceva `cv2.solvePnP` per ogni campione a ogni avvio -- e lo faceva
scorrendo una `MPSharedList`, che de-pickla un `Sample` nuovo a ogni
accesso, cosi' il valore calcolato andava perso con l'oggetto. Misurato il
2026-08-29: 8 s su 56 461 volti,
14 s su 98 803, senza nessuna barra a dirlo. Poi il doppio ciclo Python sui
128 bin, con la sua barra "Sort by yaw", altri 0,9-1,6 s.

Qui la posa si stima sulla lista ancora sciolta, prima che venga
impacchettata nella memoria condivisa, cosi' i processi figli la ereditano
gia' fatta; la matrice [N, 3] resta a disposizione del generatore, che
raggruppa per yaw con numpy invece che con il ciclo. Per un faceset
impacchettato la matrice finisce anche in una cache accanto al `.pak`
(`faceset.pose.npz`, con dimensione, mtime e conteggio del `.pak` come
chiave): dal secondo avvio la posa non costa niente.

`solvePnP` non rilascia il GIL -- i thread sono stati misurati piu' lenti
del seriale, 4,2 s contro 2,2 su 20 000 volti -- quindi sopra `SOGLIA_PROCESSI`
il calcolo passa da un `Subprocessor`: 3,6 s contro 8 su 56 461 volti, avvio
dei processi compreso. Sotto la soglia l'avvio costerebbe piu' del lavoro.
"""
import multiprocessing

import cv2
import numpy as np

from core.interact import interact as io
from core.joblib import Subprocessor
from facelib import LandmarksProcessor

SOGLIA_PROCESSI = 4000
#Campioni per messaggio verso i processi: pochi messaggi grandi, perche' il
#costo che resta e' proprio il passaggio dei landmark e dei risultati.
LOTTO = 1000

NOME_CACHE = "faceset.pose.npz"

GRADI = 128
#Non pi/2: il massimo yaw dei landmark 2DFAN e' circa +-1,2 (commento
#ereditato dal ciclo originale).
SPAZIO_YAW = np.linspace(-1.2, 1.2, GRADI)


def gruppi_per_yaw(yaws):
    """Gli indici dei campioni raggruppati per bin di yaw, bin vuoti esclusi.

    Stessi gruppi, nello stesso ordine e con gli indici nello stesso ordine
    del ciclo che sostituisce: il primo bin prende tutto cio' che sta sotto
    il suo bordo destro, l'ultimo tutto cio' che sta sopra il suo bordo
    sinistro, e un yaw non numerico -- che nel ciclo falliva ogni confronto
    -- non entra in nessun gruppo.
    """
    yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
    validi = np.isfinite(yaws)
    bins = np.searchsorted(SPAZIO_YAW[1:], yaws, side='right')
    return [np.flatnonzero((bins == g) & validi).tolist()
            for g in range(GRADI) if np.any((bins == g) & validi)]


def stima_posa(samples, pak=None):
    """La matrice [N, 3] delle pose, scritta anche su ogni campione che non
    l'aveva. `pak` e' il file del faceset impacchettato da cui i campioni
    vengono, se ne hanno uno: abilita la cache accanto a lui.
    """
    n = len(samples)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    posa = _dalla_cache(pak, n) if pak is not None else None
    if posa is None:
        mancanti = [i for i, s in enumerate(samples) if s.pitch_yaw_roll is None]
        if mancanti:
            calcolate = _stima(samples, mancanti)
            for i, valore in zip(mancanti, calcolate):
                samples[i].pitch_yaw_roll = valore
        posa = np.array([s.pitch_yaw_roll for s in samples], dtype=np.float64).reshape(n, 3)
        if pak is not None and mancanti:
            _nella_cache(pak, posa)
    else:
        for s, riga in zip(samples, posa):
            s.pitch_yaw_roll = riga
    return posa


def _stima(samples, indici):
    lavoro = [(samples[i].landmarks, samples[i].shape[1]) for i in indici]
    if len(lavoro) < SOGLIA_PROCESSI:
        return [_una(l, size) for l, size in lavoro]
    lotti = [lavoro[k:k + LOTTO] for k in range(0, len(lavoro), LOTTO)]
    return sum(_PosaSubprocessor(lotti).run(), [])


def _una(landmarks, size):
    """La posa di un volto, o NaN se i landmark non ne danno una.

    `solvePnP` solleva su punti degeneri (tutti uguali, collineari). Prima
    di questo modulo quel volto faceva cadere il training solo con
    `uniform_yaw` acceso; adesso la posa si stima a ogni caricamento, e un
    volto storto non deve fermare un dataset intero: con NaN non entra in
    nessun gruppo di yaw (`gruppi_per_yaw`), come nel ciclo originale.
    """
    try:
        return LandmarksProcessor.estimate_pitch_yaw_roll(landmarks, size=size)
    except cv2.error:
        return NON_STIMABILE


NON_STIMABILE = (np.nan, np.nan, np.nan)


# -- la cache accanto al .pak -----------------------------------------------

def percorso_cache(pak):
    return pak.parent / NOME_CACHE


def _chiave(pak, n):
    st = pak.stat()
    #mtime intero: la stessa chiave letta da Windows e da WSL sullo stesso
    #file, dove i nanosecondi non e' detto che coincidano.
    return np.array([st.st_size, int(st.st_mtime), n], dtype=np.int64)


def _dalla_cache(pak, n):
    percorso = percorso_cache(pak)
    if not percorso.exists():
        return None
    try:
        with np.load(percorso) as f:
            if not np.array_equal(f["chiave"], _chiave(pak, n)):
                return None
            posa = np.asarray(f["posa"], dtype=np.float64)
    except Exception:
        return None
    if posa.shape != (n, 3):
        return None
    return posa


def _nella_cache(pak, posa):
    try:
        np.savez(percorso_cache(pak), chiave=_chiave(pak, len(posa)), posa=posa)
    except Exception:
        #Una cartella di sola lettura non e' un motivo per non allenare:
        #si paga il calcolo al prossimo avvio, come prima di questo modulo.
        pass


def rimuovi_cache(pak):
    percorso = percorso_cache(pak)
    if percorso.exists():
        percorso.unlink()


# -- i processi -------------------------------------------------------------

class _PosaSubprocessor(Subprocessor):
    #override
    def __init__(self, lotti):
        self.lotti = lotti
        self.idxs = [*range(len(lotti))]
        self.result = [None] * len(lotti)
        super().__init__('PosaCampioni', _PosaSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % i, {}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar("Estimating pose", len(self.lotti))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len(self.idxs) > 0:
            idx = self.idxs.pop(0)
            return idx, self.lotti[idx]
        return None

    #override
    def on_data_return(self, host_dict, data):
        self.idxs.insert(0, data[0])

    #override
    def on_result(self, host_dict, data, result):
        idx, pose = result
        self.result[idx] = pose
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, lotto = data
            return idx, [_una(l, size) for l, size in lotto]

        #override
        def get_data_name(self, data):
            return "lotto %d" % data[0]
