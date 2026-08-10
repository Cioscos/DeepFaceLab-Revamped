"""
Support units for the training loop: small, single-purpose classes that
mainscripts/Trainer.py wires together. Kept separate so each can be
exercised with an injected clock instead of real time.
"""
import collections
import json
import os
import time
from pathlib import Path

import numpy as np

from core.cv2ex import cv2_imwrite


class SaveScheduler(object):
    """
    Decides when the periodic save is due. A stall longer than one interval
    (a save that took long, a laptop that slept) collapses into a single
    save, but the schedule stays aligned to fixed interval boundaries: the
    next save is not pushed later by the stall.
    """
    def __init__(self, interval_min, clock=time.time):
        if interval_min < 1:
            raise ValueError(
                "interval_min must be >= 1, got %r: with 0 (or negative) "
                "due() never advances self.last and loops forever" % (interval_min,))
        self.interval_sec = interval_min * 60
        self.clock = clock
        self.last = clock()

    def due(self):
        now = self.clock()
        was_due = False
        while now - self.last >= self.interval_sec:
            self.last += self.interval_sec
            was_due = True
        return was_due


class ProgressLine(object):
    """
    Formats the one-line training status: raw losses, their exponential
    moving average, iteration rate, ETA to the target and device memory.
    Pure formatting over an injected clock and probe, so it can be
    exercised without a GPU or real time.
    """
    EMA_ALPHA = 0.02          # the last hundred iterations dominate, roughly
    RATE_WINDOW = 50          # samples kept for the it/s estimate
    VRAM_EVERY_SEC = 1.0      # driver query cadence, not every iteration

    def __init__(self, clock=time.time, vram=None):
        self.clock = clock
        self.vram = vram
        self.ema = None
        self.window = collections.deque(maxlen=self.RATE_WINDOW)
        self._vram_last = (float("-inf"), None)

    @staticmethod
    def _fmt_eta(sec):
        sec = int(sec)
        if sec >= 3600:
            return "%dh %02dm" % (sec // 3600, (sec % 3600) // 60)
        if sec >= 60:
            return "%dm %02ds" % (sec // 60, sec % 60)
        return "%ds" % sec

    def update(self, iter, iter_time, losses, target_iter=0):
        now = self.clock()
        self.window.append((now, iter))

        if self.ema is None:
            self.ema = list(losses)
        else:
            self.ema = [e + self.EMA_ALPHA * (l - e)
                        for e, l in zip(self.ema, losses)]

        if iter_time >= 10:
            tempo = "[{:.5s}s]".format("{:0.4f}".format(iter_time))
        else:
            tempo = "[{:04d}ms]".format(int(iter_time * 1000))

        riga = time.strftime("[%H:%M:%S]") + "[#%06d]" % iter + tempo
        riga += "[" + " ".join("%.4f" % v for v in losses) + "]"
        riga += "[" + " ".join("~%.4f" % v for v in self.ema) + "]"

        rate = None
        if len(self.window) >= 2:
            (t0, i0), (t1, i1) = self.window[0], self.window[-1]
            if t1 > t0:
                rate = (i1 - i0) / (t1 - t0)
        if rate is not None:
            riga += "[%.1f it/s]" % rate
            if target_iter and target_iter > iter and rate > 0:
                riga += "[ETA %s]" % self._fmt_eta((target_iter - iter) / rate)

        if self.vram is not None:
            t_cache, valore = self._vram_last
            if now - t_cache >= self.VRAM_EVERY_SEC:
                valore = self.vram()
                self._vram_last = (now, valore)
            if valore is not None:
                riga += "[VRAM %.1f/%.1fG]" % valore
        return riga


class LossCsv(object):
    """
    Mirrors the model's loss history into a CSV next to the model files,
    appending only the rows added since the previous save. Row i of the
    history is iteration i+1. The timestamp is the moment the row was
    written (histories predating this file carry no per-iteration time).
    """
    def __init__(self, path):
        self.path = Path(path)
        self.righe_scritte = 0

    def _scrivi_tutto(self, loss_history):
        if not loss_history:
            self.path.write_text("")
            self.righe_scritte = 0
            return
        larghezza = len(loss_history[0])
        intestazione = "iter,timestamp," + ",".join(
            "loss%d" % (i + 1) for i in range(larghezza))
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            f.write(intestazione + "\n")
            self._appendi(f, loss_history, 0)
        self.righe_scritte = len(loss_history)

    def _appendi(self, f, loss_history, da):
        ora = time.strftime("%Y-%m-%dT%H:%M:%S")
        for i in range(da, len(loss_history)):
            f.write("%d,%s,%s\n" % (
                i + 1, ora, ",".join("%.6f" % v for v in loss_history[i])))

    def align(self, loss_history):
        if not self.path.exists():
            self._scrivi_tutto(loss_history)
            return
        with open(self.path, encoding="utf-8") as f:
            righe_dati = max(0, sum(1 for _ in f) - 1)
        if righe_dati > len(loss_history):
            self._scrivi_tutto(loss_history)     # rollback: il file mente
        else:
            self.righe_scritte = righe_dati

    def append_new(self, loss_history):
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._scrivi_tutto(loss_history)
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            self._appendi(f, loss_history, self.righe_scritte)
        self.righe_scritte = len(loss_history)


def _json_nativo(valore):
    """Uno scalare numpy come il tipo Python corrispondente.

    Le opzioni dei modelli passano per np.clip -- l'idioma di upstream --
    quindi una `resolution` riletta dal pickle e' un np.int64, non un int:
    attraversa somme, confronti e min() senza dare segno di se' e muore molto
    piu' tardi qui, dove json.dumps non sa codificarla. Non e' una
    decorazione che si perde: _write vive fuori dal try del thread, quindi
    l'evento si porta via il training intero.

    La conversione sta al passaggio e non su un campo per volta: e' l'unico
    punto per cui passano tutti gli eventi, quindi la prossima opzione numpy
    che entra in un payload non ricade qui. Quello che non e' uno scalare
    numpy resta un errore, e forte: questa e' una conversione che decide, non
    una rete che assorbe.
    """
    if isinstance(valore, np.generic):
        return valore.item()
    raise TypeError("Object of type %s is not JSON serializable"
                    % valore.__class__.__name__)


class EventLog(object):
    """
    Structured training events for an external observer, as JSON lines
    appended to a file. path=None turns every method into a no-op, so a
    run without the observer behaves exactly as before. iter events are
    rate-limited: at most one every ITER_EVERY_SEC, while save/end always
    write.
    """
    VERSION = 1
    ITER_EVERY_SEC = 0.5

    def __init__(self, path, clock=time.time):
        self.path = path
        self.clock = clock
        self._last_iter_emit = float("-inf")

    def _write(self, payload):
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=_json_nativo) + "\n")

    def hello(self, model_name_class, target_iter, model_name=None, model_dir=None):
        #model_name/model_dir sono additivi (VERSION resta 1): senza di loro
        #un osservatore non puo' trovare i file del modello, che portano il
        #nome dell'istanza e non quello della classe.
        self._write({"type": "hello", "version": self.VERSION,
                     "model": model_name_class, "target_iter": target_iter,
                     "model_name": model_name, "model_dir": model_dir})

    def iter(self, iter, iter_time, losses, vram=None):
        #Senza osservatore non si guarda nemmeno l'orologio. Le due ragioni
        #per cui un evento non viene scritto sono due, e la guardia le
        #copriva a meta': la limitazione di frequenza stava qui, l'assenza
        #dell'osservatore soltanto dentro `_write`, cioe' dopo. Il risultato
        #era che una corsa da riga di comando -- dove path e' None e non
        #verra' scritto niente, mai -- interrogava comunque la GPU due volte
        #al secondo per tutta la sua durata, con una chiamata che su quel
        #percorso, prima di questo canale, non esisteva affatto.
        if self.path is None:
            return
        now = self.clock()
        if now - self._last_iter_emit < self.ITER_EVERY_SEC:
            return
        self._last_iter_emit = now
        #vram puo' essere una coppia o una funzione che la produce. La
        #seconda forma esiste perche' misurarla interroga la GPU: valutarla
        #a ogni iterazione per poi scartare l'evento sarebbe un costo per
        #niente, e da qui in giu' l'evento verra' scritto di sicuro.
        misura = vram() if callable(vram) else vram
        usata, totale = misura if misura is not None else (None, None)
        self._write({"type": "iter", "iter": iter, "iter_time": iter_time,
                     "losses": [float(v) for v in losses],
                     "vram_usata_gib": usata, "vram_totale_gib": totale})

    def save(self, iter):
        self._write({"type": "save", "iter": iter})

    def preview(self, iter, immagini, nomi_file=None):
        self._write({"type": "preview", "iter": iter, "immagini": immagini,
                     "nomi_file": nomi_file})

    def end(self):
        self._write({"type": "end"})


class CommandTail(object):
    """
    Commands from an external observer, read as JSON lines appended to a
    file -- the mirror image of EventLog, same format, opposite direction.
    path=None makes nuovi() a no-op, so a run started without the channel
    behaves exactly as before. Only newline-terminated lines are consumed,
    so a half-written command is never acted upon; anything that is not a
    dict carrying a string "op" is ignored.
    """
    def __init__(self, path):
        self.path = path
        self._pos = 0

    def nuovi(self):
        """The ops appended since the last call, oldest first."""
        if self.path is None:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                data = f.read()
        except OSError:
            return []
        cut = data.rfind(b"\n")
        if cut == -1:
            return []
        self._pos += cut + 1
        ops = []
        for line in data[:cut + 1].decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("op"), str):
                ops.append(payload["op"])
        return ops


class PreviewWriter(object):
    """
    Le anteprime per un osservatore esterno: un PNG per anteprima nella
    cartella indicata, piu' una riga sul canale eventi che le annuncia.
    dir_path=None rende scrivi() un no-op, quindi una corsa senza osservatore
    si comporta esattamente come prima.

    Due regole, ed entrambe esistono perche' il lettore guarda la cartella
    mentre questa la riscrive: il file si scrive su '.tmp' e poi si rinomina
    (os.replace e' atomico su entrambi i sistemi), cosi' nessuno vede mai
    mezza immagine; e l'evento si appende DOPO la rinomina, mai prima, cosi'
    non annuncia mai un file che non c'e'.

    I nomi sono l'indice dell'anteprima, non il suo nome: stabili fra una
    scrittura e l'altra -- la cartella non cresce -- e senza il problema dei
    caratteri che un nome di anteprima puo' contenere e un filesystem no.
    """
    def __init__(self, dir_path, events):
        self.dir_path = Path(dir_path) if dir_path is not None else None
        self.events = events

    def scrivi(self, previews, iter, layouts=None, nomi_file=None):
        """Scrive le anteprime e le annuncia. Ritorna le voci annunciate."""
        if self.dir_path is None:
            return []
        self.dir_path.mkdir(parents=True, exist_ok=True)
        immagini = []
        for indice, (nome, immagine) in enumerate(previews):
            nome_file = "%d.png" % indice
            finale = self.dir_path / nome_file
            provvisorio = self.dir_path / ("%d.tmp.png" % indice)
            cv2_imwrite(provvisorio, (np.clip(immagine, 0, 1) * 255).astype(np.uint8))
            if not provvisorio.exists():
                continue    # cv2_imwrite ingoia i suoi errori: qui li vediamo
            os.replace(str(provvisorio), str(finale))
            voce = {"nome": nome, "file": nome_file}
            if layouts is not None and nome in layouts:
                voce.update(layouts[nome])
            immagini.append(voce)
        if immagini:
            self.events.preview(iter, immagini, nomi_file)
        return immagini
