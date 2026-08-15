"""L'indice di una cartella di volti: metadati e maschere, fuori dal progetto.

Scrive tre file in una cartella che gli viene DETTA -- non calcolata qui.
Il nome della cartella di cache lo decide l'interfaccia grafica
(gui/faceset/cache.py) ed e' l'unica implementazione che esiste: due
implementazioni dello stesso nome potrebbero divergere e la cache
diventerebbe invisibile a chi la cerca.

La chiave di un volto e' (nome, dimensione, mtime). Il fallback su
(dimensione, mtime) -- che rende la cache immune alle rinomine di un sort
-- vive dal lato che legge, non da questo.
"""
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from core.interact import interact as io
from core.joblib import Subprocessor
from DFLIMG import DFLJPG
from facelib import LandmarksProcessor

FORMATO = 1
NOME_INDICE = "index.ndjson"
NOME_BLOB = "masks.bin"
NOME_META = "meta.json"

ESTENSIONI = (".jpg", ".jpeg", ".png")


def chiave_di(path):
    st = Path(path).stat()
    return (Path(path).name, st.st_size, st.st_mtime_ns)


def descrivi(path):
    """(riga dell'indice, byte della maschera o None). Solleva se il file non e' leggibile."""
    path = Path(path)
    dfl = DFLJPG.load(str(path))
    if dfl is None:
        raise ValueError("non e' un JPEG DFL")
    nome, dimensione, mtime = chiave_di(path)
    riga = {"n": nome, "s": dimensione, "m": mtime}

    if dfl.has_data():
        lm = dfl.get_landmarks()
        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(
            lm, size=dfl.get_shape()[1])
        riga.update({"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll),
                     "ft": dfl.get_face_type(), "src": dfl.get_source_filename()})

    maschera = None
    if dfl.has_xseg_mask():
        # Verbatim: e' gia' un PNG compresso, decodificarlo per
        # ricodificarlo costerebbe una decodifica per volto in cambio di
        # niente.
        buf = dfl.get_xseg_mask_compressed()
        maschera = bytes(np.asarray(buf).tobytes())
    return riga, maschera


# Oltre questo numero di figli il carico diventa I/O-bound e aggiungere CPU
# non paga piu' -- lo stesso tetto del caricatore del sorter.
MAX_FIGLI = 8


class IndicizzatoreVolti(Subprocessor):
    """Una passata parallela sulla cartella. Stessa forma del caricatore
    del sorter: otto figli al massimo, thread OpenCV cappati nei figli."""

    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            import cv2
            cv2.setNumThreads(1)

        #override
        def process_data(self, data):
            try:
                riga, maschera = descrivi(data[0])
                return [0, riga, maschera]
            except Exception as e:
                self.log_err(f"{data[0]}: {e}")
                return [1, str(data[0]), None]

        #override
        def get_data_name(self, data):
            return data[0]

    #override
    def __init__(self, paths, consegna=None):
        """`consegna(riga, maschera)` riceve ogni volto APPENA e' pronto.

        Senza, i risultati si accumulano in `self.validi` come prima -- e'
        la forma che serve a chi vuole la passata e basta, senza un indice
        su disco (i test, e chiunque riusi l'indicizzatore). La produzione
        passa il `scrivi` di uno `ScrittoreIndice`: cosi' una corsa
        interrotta lascia sul disco tutto cio' che aveva gia' fatto, e
        `self.validi` non cresce col numero di volti (~69 MB di sole
        maschere a 50 000).
        """
        self.input_data = [[str(p)] for p in paths]
        self.validi = []
        self.scarti = []
        self._consegna = consegna if consegna is not None else (
            lambda riga, maschera: self.validi.append((riga, maschera)))
        super().__init__('IndicizzatoreVolti', IndicizzatoreVolti.Cli, 60)

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), MAX_FIGLI)
        io.log_info(f'Running on {cpu_count} CPUs')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar("Indexing", len(self.input_data))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len(self.input_data) > 0:
            return self.input_data.pop(0)
        return None

    #override
    def on_data_return(self, host_dict, data):
        self.input_data.insert(0, data)

    #override
    def on_result(self, host_dict, data, result):
        # La consegna avviene QUI, un volto alla volta, e non dopo `run()`:
        # lo Stop della pagina uccide l'intero process tree, quindi tutto
        # cio' che non e' gia' sul disco quando arriva e' perso.
        if result[0] == 0:
            self._consegna(result[1], result[2])
        else:
            self.scarti.append(result[1])
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.validi, self.scarti

    def run(self):
        if not self.input_data:
            return [], []
        return super().run()


def _voci_esistenti(cache_dir):
    percorso = Path(cache_dir) / NOME_INDICE
    if not percorso.exists():
        return set()
    viste = set()
    for r in percorso.read_text(encoding="utf-8").splitlines():
        if not r:
            continue
        try:
            d = json.loads(r)
        except ValueError:
            continue
        viste.add((d.get("n"), d.get("s"), d.get("m")))
    return viste


def _file_da_indicizzare(input_dir, only_missing, cache_dir):
    gia = _voci_esistenti(cache_dir) if only_missing else set()
    da_fare, saltati = [], 0
    for e in sorted(Path(input_dir).iterdir(), key=lambda p: p.name):
        if not e.is_file() or e.suffix.lower() not in ESTENSIONI:
            continue
        if chiave_di(e) in gia:
            saltati += 1
            continue
        da_fare.append(e)
    return da_fare, saltati


class ScrittoreIndice:
    """L'indice si scrive MENTRE la corsa va avanti, non alla fine.

    Lo Stop della pagina uccide l'intero process tree, quindi una corsa
    fermata a meta' su 50 000 volti non lascerebbe nessuna riga sul disco e
    `--only-missing` ricomincerebbe da zero: e' l'unica ragione per cui
    questa classe esiste, e per cui ogni voce e' seguita da un `flush()`.
    Il `flush` basta e `fsync` no: quello che si teme e' la morte del
    processo, non quella della macchina, e i byte passati al kernel
    sopravvivono a un SIGKILL.

    **L'ordine dentro `scrivi` non e' negoziabile**: prima i byte della
    maschera, poi la riga che li indica. Al contrario, un arresto fra le due
    scritture lascerebbe una riga che punta a byte che non ci sono -- una
    maschera letta a caso, cioe' un guasto silenzioso. Cosi' invece l'unico
    danno possibile e' qualche byte orfano in coda al blob, che nessuna riga
    nomina e che la corsa successiva si limita a superare (apre in append e
    scrive dopo).

    L'ordine delle righe NON e' quello per nome che `_file_da_indicizzare`
    produce: e' l'ordine di completamento di otto figli in corsa fra loro,
    quindi non specificato e diverso a ogni corsa. Cio' che resta garantito:
    ogni volto indicizzato compare esattamente una volta per corsa, e la
    riconciliazione lato lettura avviene per chiave (nome, dimensione,
    mtime) -- vedi `chiave_di` -- mai per posizione nel file.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.indicizzati = 0
        self._fb = None
        self._fi = None

    def __enter__(self):
        indice = self.cache_dir / NOME_INDICE
        self._chiudi_una_riga_troncata(indice)
        self._fb = open(self.cache_dir / NOME_BLOB, "ab")
        # In append i byte finiscono in coda comunque, ma `tell()` prima
        # della prima scrittura non e' garantito: l'offset della maschera
        # viene da qui, quindi si posiziona esplicitamente.
        self._fb.seek(0, os.SEEK_END)
        self._fi = open(indice, "a", encoding="utf-8")
        return self

    def __exit__(self, *_eccezione):
        for f in (self._fi, self._fb):
            if f is not None:
                f.close()
        self._fi = self._fb = None
        return False

    @staticmethod
    def _chiudi_una_riga_troncata(indice):
        """Un arresto a meta' di una `write` lascia una riga senza `\\n`.

        Senza questa chiusura la prima riga della corsa successiva le si
        incolla dietro e ne diventano illeggibili DUE: quella troncata, che
        era gia' persa, e quella nuova, che non lo era.
        """
        try:
            with open(indice, "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    return
                f.seek(-1, os.SEEK_END)
                intera = f.read(1) == b"\n"
        except OSError:
            return
        if not intera:
            with open(indice, "a", encoding="utf-8") as f:
                f.write("\n")

    def scrivi(self, riga, maschera):
        if maschera:
            riga["mo"] = self._fb.tell()
            riga["ml"] = len(maschera)
            self._fb.write(maschera)
            self._fb.flush()
        else:
            riga["ml"] = 0
        self._fi.write(json.dumps(riga) + "\n")
        self._fi.flush()
        self.indicizzati += 1


def indicizza(input_dir, cache_dir, only_missing=True):
    """Scrive l'indice della cartella. Ritorna i conteggi."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    da_fare, saltati = _file_da_indicizzare(input_dir, only_missing, cache_dir)
    with ScrittoreIndice(cache_dir) as scrittore:
        _validi, scarti_lista = IndicizzatoreVolti(
            da_fare, consegna=scrittore.scrivi).run()
        indicizzati = scrittore.indicizzati
    scarti = len(scarti_lista)

    (cache_dir / NOME_META).write_text(json.dumps({
        "formato": FORMATO,
        "origine": str(Path(input_dir).resolve()),
    }), encoding="utf-8")
    return {"indicizzati": indicizzati, "saltati": saltati, "scarti": scarti}
