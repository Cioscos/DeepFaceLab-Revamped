"""
Support units for the training loop: small, single-purpose classes that
mainscripts/Trainer.py wires together. Kept separate so each can be
exercised with an injected clock instead of real time.
"""
import collections
import json
import time
from pathlib import Path


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
            f.write(json.dumps(payload) + "\n")

    def hello(self, model_name, target_iter):
        self._write({"type": "hello", "version": self.VERSION,
                     "model": model_name, "target_iter": target_iter})

    def iter(self, iter, iter_time, losses):
        now = self.clock()
        if now - self._last_iter_emit < self.ITER_EVERY_SEC:
            return
        self._last_iter_emit = now
        self._write({"type": "iter", "iter": iter, "iter_time": iter_time,
                     "losses": [float(v) for v in losses]})

    def save(self, iter):
        self._write({"type": "save", "iter": iter})

    def end(self):
        self._write({"type": "end"})
