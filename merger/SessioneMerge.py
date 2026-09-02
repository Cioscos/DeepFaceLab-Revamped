"""Lo stato della sessione di fusione, senza cv2 e senza io.

Prima stava dentro InteractiveMergerSubprocessor.on_tick, mescolato alla
lettura dei tasti e al disegno. Qui ci sono i frame, il CURSORE (il frame
mostrato), la cfg per frame, la propagazione, il riavvolgimento e il
.dat; il frontend cv2 e il servizio per la GUI (mainscripts/MergeSession.py)
chiamano gli stessi metodi. Il .dat resta quello di sempre: frames_idxs /
frames_done_idxs si DERIVANO dal cursore al salvataggio e lo ridanno al
caricamento; `keyframes` e' la sola chiave nuova, e chi non la conosce la
ignora.

`Frame` e `ProcessingFrame` vivono qui e InteractiveMergerSubprocessor le
riesporta come alias: un .dat vecchio le pickla col qualname annidato
(`InteractiveMergerSubprocessor.Frame`) e pickle lo risolve con getattr
sul modulo, quindi l'alias basta.

Quando `carica_sessione` rifiuta la sessione (RIPRESA_NON_CORRISPONDE) o la
azzera (RIPRESA_AZZERATA), cancellare gli eventuali PNG di output gia'
scritti resta compito di chi chiama: questo modulo non tocca mai il
filesystem oltre al proprio .dat.
"""
import bisect
import copy
import pickle
from pathlib import Path

import numpy as np

from core import pickleex
from .MergerConfig import mode_str_dict, mask_mode_dict, ctm_dict, LIMITI_INTERPOLATI

RIPRESA_NESSUNA = "none"
RIPRESA_OK = "ripresa"
RIPRESA_AZZERATA = "azzerata_per_iter"
RIPRESA_NON_CORRISPONDE = "non_corrisponde"

# I quindici parametri della configurazione di merge (tredici di
# MergerConfigMasked piu' i due ereditati da MergerConfig), divisi per come
# un piano a keyframe li tratta. Il test sotto verifica che l'unione sia
# ESATTAMENTE get_config() meno face_type/default_mode/sharpen_dict: un
# parametro nuovo non puo' restare fuori in silenzio.
CAMPI_INTERPOLATI = ("hist_match_threshold", "erode_mask_modifier",
                     "blur_mask_modifier", "motion_blur_power",
                     "output_face_scale", "super_resolution_power",
                     "blursharpen_amount", "image_denoise_power",
                     "bicubic_degrade_power", "color_degrade_power")
CAMPI_A_SCALINO = ("mode", "mask_mode", "color_transfer_mode",
                   "sharpen_mode", "masked_hist_match")


class Frame(object):
    def __init__(self, prev_temporal_frame_infos=None,
                       frame_info=None,
                       next_temporal_frame_infos=None):
        self.prev_temporal_frame_infos = prev_temporal_frame_infos
        self.frame_info = frame_info
        self.next_temporal_frame_infos = next_temporal_frame_infos
        self.output_filepath = None
        self.output_mask_filepath = None

        self.idx = None
        self.cfg = None
        self.is_done = False
        self.is_processing = False
        self.is_shown = False
        self.image = None


class ProcessingFrame(object):
    def __init__(self, idx=None,
                       cfg=None,
                       prev_temporal_frame_infos=None,
                       frame_info=None,
                       next_temporal_frame_infos=None,
                       output_filepath=None,
                       output_mask_filepath=None,
                       need_return_image = False):
        self.idx = idx
        self.cfg = cfg
        self.prev_temporal_frame_infos = prev_temporal_frame_infos
        self.frame_info = frame_info
        self.next_temporal_frame_infos = next_temporal_frame_infos
        self.output_filepath = output_filepath
        self.output_mask_filepath = output_mask_filepath

        self.need_return_image = need_return_image
        if self.need_return_image:
            self.image = None


class SessioneMerge(object):
    def __init__(self, frames, merger_config, prefetch=4):
        if len(frames) == 0:
            raise ValueError("len (frames) == 0")
        self.frames = frames
        self.merger_config = merger_config
        self.prefetch = prefetch
        self.cursore = 0
        self.keyframes = {}
        self.elabora_restanti = False
        self.in_chiusura = False     # l'Esc della finestra: il frontend lo legge
        self._in_testa = []          # indici da servire prima del prefetch (sonda)
        for i, frame in enumerate(self.frames):
            frame.idx = i
        self.frames[0].cfg = self.merger_config.copy()

    # -- output -----------------------------------------------------------

    def assegna_output(self, output_path, output_mask_path):
        for frame in self.frames:
            stem = frame.frame_info.filepath.stem
            frame.output_filepath = Path(output_path) / (stem + '.png')
            frame.output_mask_filepath = Path(output_mask_path) / (stem + '.png')

    def _riavvolgi_sui_mancanti(self):
        """Un frame senza i suoi due PNG torna da fare, e il cursore
        arretra fino al frame prima del primo mancante (come oggi)."""
        riavvolgi_a = None
        for frame in self.frames:
            if not frame.output_filepath.exists() or \
               not frame.output_mask_filepath.exists():
                frame.is_done = False
                frame.is_shown = False
                riavvolgi_a = frame.idx - 1 if riavvolgi_a is None else min(riavvolgi_a, frame.idx - 1)
        if riavvolgi_a is not None:
            self.cursore = max(0, min(self.cursore, riavvolgi_a + 1))

    # -- lettura ----------------------------------------------------------

    def corrente(self):
        return self.cursore

    def frame_corrente(self):
        return self.frames[self.cursore]

    def senza_volto(self):
        return [f.idx for f in self.frames if len(f.frame_info.landmarks_list) == 0]

    def stato(self):
        # `fatti_idx` accanto al conteggio: chi disegna una timeline ha
        # bisogno di sapere QUALI, e da un numero non si ricava. E' cio'
        # che rende verde una sessione ripresa.
        fatti_idx = [f.idx for f in self.frames if f.is_done]
        return {"frame_totali": len(self.frames), "cursore": self.cursore,
                "fatti": len(fatti_idx), "da_fare": len(self.frames) - len(fatti_idx),
                "fatti_idx": fatti_idx,
                "senza_volto": self.senza_volto(),
                "keyframes": sorted(self.keyframes),
                "batch": self.elabora_restanti}

    # -- navigazione ------------------------------------------------------

    def _segna_da_rifare(self, frame):
        frame.is_done = False
        frame.is_shown = False
        frame.image = None

    def precedente(self, propaga=False, fino_al_primo=False):
        cur = self.frame_corrente()
        if not cur.is_done:
            return False
        cur.image = None
        while self.cursore > 0:
            prev = self.frames[self.cursore - 1]
            prev.is_shown = False
            if propaga and prev.cfg != cur.cfg:
                prev.cfg = cur.cfg.copy()
                self._segna_da_rifare(prev)
            self.cursore -= 1
            if not fino_al_primo:
                break
        return True

    def successivo(self, propaga=False, fino_all_ultimo=False):
        cur = self.frame_corrente()
        if not cur.is_done or self.cursore >= len(self.frames) - 1:
            return False
        cur.image = None
        cur.is_shown = True
        self.cursore += 1
        nxt = self.frames[self.cursore]
        nxt.is_shown = False
        if propaga:
            fine = len(self.frames) if fino_all_ultimo else nxt.idx + 1
            for i in range(nxt.idx, fine):
                self.frames[i].cfg = None
        self._eredita_cfg()
        return True

    def vai(self, idx):
        idx = max(0, min(int(idx), len(self.frames) - 1))
        if idx == self.cursore:
            return
        self.frame_corrente().image = None
        self.cursore = idx
        self.frames[idx].is_shown = False
        self._eredita_cfg()

    def _eredita_cfg(self):
        """I frame dal cursore al prefetch senza cfg la copiano dalla cfg
        piu' vicina a sinistra (di solito il frame prima; un `vai` diretto
        puo' saltare oltre il prefetch e lasciare un vuoto piu' largo)."""
        for i in range(self.cursore, min(len(self.frames), self.cursore + self.prefetch)):
            frame = self.frames[i]
            if frame.cfg is None:
                sorgente = self.merger_config
                for j in range(i - 1, -1, -1):
                    if self.frames[j].cfg is not None:
                        sorgente = self.frames[j].cfg
                        break
                frame.cfg = sorgente.copy()
                self._segna_da_rifare(frame)

    # -- configurazione ---------------------------------------------------

    def modifica_cfg(self, funzione):
        cur = self.frame_corrente()
        if cur.cfg is None:
            self._eredita_cfg()
        prima = cur.cfg.copy()
        funzione(cur.cfg)
        self.elabora_restanti = False
        if prima != cur.cfg:
            self._segna_da_rifare(cur)
            # «la cfg di questo frame l'ha fissata l'utente»: un cambio su un
            # frame che non e' keyframe lo rende tale, su uno che lo e' lo
            # aggiorna. Il piano NON si applica qui: e' un comando a parte.
            self.keyframes[cur.idx] = cur.cfg.copy()
            return True
        return False

    def imposta_cfg(self, campi):
        """Applica solo i campi noti, con la stessa validazione dei metodi
        di MergerConfig: `mode` accetta solo una stringa dell'enumerazione
        (altrimenti la chiave e' ignorata, come farebbe `set_mode` con una
        chiave assente), `mask_mode`/`color_transfer_mode`/`sharpen_mode`
        solo un valore della propria enumerazione, gli interi sono clippati
        con LIMITI_INTERPOLATI (la tabella che usano anche gli add_* di
        MergerConfig) e `masked_hist_match` diventa un booleano."""
        def _applica(cfg):
            for chiave, valore in campi.items():
                if chiave == "mode":
                    if valore in mode_str_dict:
                        cfg.mode = valore
                elif chiave == "mask_mode":
                    if valore in mask_mode_dict:
                        cfg.mask_mode = valore
                elif chiave == "color_transfer_mode":
                    if valore in ctm_dict:
                        cfg.color_transfer_mode = valore
                elif chiave == "sharpen_mode":
                    if valore in cfg.sharpen_dict:
                        cfg.sharpen_mode = valore
                elif chiave == "masked_hist_match":
                    cfg.masked_hist_match = bool(valore)
                elif chiave in LIMITI_INTERPOLATI:
                    lo, hi = LIMITI_INTERPOLATI[chiave]
                    setattr(cfg, chiave, max(lo, min(hi, valore)))
        return self.modifica_cfg(_applica)

    def commuta_batch(self):
        self.elabora_restanti = not self.elabora_restanti
        return self.elabora_restanti

    def imposta_batch(self, acceso):
        self.elabora_restanti = bool(acceso)
        return self.elabora_restanti

    # -- keyframe e piano -------------------------------------------------

    def _cfg_a_sinistra(self, idx):
        for j in range(idx, -1, -1):
            if self.frames[j].cfg is not None:
                return self.frames[j].cfg
        return self.merger_config

    def imposta_keyframe(self, idx):
        frame = self.frames[idx]
        if frame.cfg is None:
            frame.cfg = self._cfg_a_sinistra(idx).copy()
            self._segna_da_rifare(frame)
        nuovo = idx not in self.keyframes
        self.keyframes[idx] = frame.cfg.copy()
        return nuovo

    def togli_keyframe(self, idx):
        return self.keyframes.pop(idx, None) is not None

    def _cfg_del_piano(self, idx, chiavi, interpola):
        """La cfg che il piano assegna a `idx`, dati gli indici ordinati dei
        keyframe: prima del primo vale il primo, dopo l'ultimo l'ultimo; in
        mezzo i dieci campi interpolati linearmente e arrotondati, i cinque a
        scalino dal keyframe precedente; senza interpolazione, il precedente."""
        pos = bisect.bisect_right(chiavi, idx)
        if pos == 0:
            return self.keyframes[chiavi[0]].copy()
        prec = chiavi[pos - 1]
        if pos == len(chiavi) or not interpola or prec == idx:
            return self.keyframes[prec].copy()
        succ = chiavi[pos]
        a, b = self.keyframes[prec], self.keyframes[succ]
        cfg = a.copy()
        t = (idx - prec) / (succ - prec)
        for campo in CAMPI_INTERPOLATI:
            va, vb = getattr(a, campo), getattr(b, campo)
            setattr(cfg, campo, int(round(va + (vb - va) * t)))
        return cfg

    def applica_piano(self, interpola=True, anteprima=False):
        """Ricalcola la cfg di ogni frame con volto dai keyframe; segna da
        rifare quelli la cui cfg cambia. Con `anteprima` conta soltanto.
        Non tocca il batch: un piano applicato e' «ho finito di regolare»,
        e il pool puo' continuare sui frame nuovi. I frame cambiati vanno in
        testa alla coda (come per `sonda`): la finestra del prefetch e'
        legata al cursore, e un piano puo' toccare frame lontani da esso."""
        if not self.keyframes:
            return 0
        chiavi = sorted(self.keyframes)
        cambiati = 0
        toccati = []
        for frame in self.frames:
            if len(frame.frame_info.landmarks_list) == 0:
                continue
            nuova = self._cfg_del_piano(frame.idx, chiavi, interpola)
            if frame.cfg is None or frame.cfg != nuova:
                cambiati += 1
                if not anteprima:
                    frame.cfg = nuova
                    self._segna_da_rifare(frame)
                    toccati.append(frame.idx)
        if toccati:
            self._in_testa = toccati + [i for i in self._in_testa if i not in toccati]
        return cambiati

    # -- sonda ------------------------------------------------------------

    def sonda(self, n):
        """`n` frame con volto equispaziati, con la cfg corrente, in testa
        alla coda del pool: un frame sondato e' un frame fatto, e il batch
        non lo rifa'. Torna gli indici scelti, ordinati e senza duplicati."""
        validi = [f.idx for f in self.frames if len(f.frame_info.landmarks_list) > 0]
        if not validi:
            return []
        n = max(1, min(int(n), len(validi)))
        posizioni = np.linspace(0, len(validi) - 1, n)
        scelti = sorted({validi[int(round(p))] for p in posizioni})
        cfg = self._cfg_a_sinistra(self.cursore)
        for idx in scelti:
            frame = self.frames[idx]
            if frame.cfg is None or frame.cfg != cfg:
                frame.cfg = cfg.copy()
                self._segna_da_rifare(frame)
        self._in_testa = [i for i in scelti if not self.frames[i].is_done] + \
                         [i for i in self._in_testa if i not in scelti]
        return scelti

    # -- il pool ----------------------------------------------------------

    def _processing_frame(self, frame):
        frame.is_processing = True
        return ProcessingFrame(idx=frame.idx, cfg=frame.cfg.copy(),
                               prev_temporal_frame_infos=frame.prev_temporal_frame_infos,
                               frame_info=frame.frame_info,
                               next_temporal_frame_infos=frame.next_temporal_frame_infos,
                               output_filepath=frame.output_filepath,
                               output_mask_filepath=frame.output_mask_filepath,
                               need_return_image=True)

    def da_elaborare(self):
        while self._in_testa:
            frame = self.frames[self._in_testa[0]]
            if frame.is_done or frame.is_processing or frame.cfg is None:
                self._in_testa.pop(0)
                continue
            return self._processing_frame(frame)
        self._eredita_cfg()
        for i in range(self.cursore, min(len(self.frames), self.cursore + self.prefetch)):
            frame = self.frames[i]
            if not frame.is_done and not frame.is_processing and frame.cfg is not None:
                return self._processing_frame(frame)
        return None

    def su_ritorno(self, idx):
        frame = self.frames[idx]
        frame.is_done = False
        frame.is_processing = False

    def su_risultato(self, idx, cfg, image):
        frame = self.frames[idx]
        frame.is_processing = False
        if frame.cfg == cfg:
            frame.is_done = True
            frame.image = image

    def avanza_batch(self):
        """Il batch: se acceso e il corrente e' fatto, avanza; in fondo si
        spegne da solo. Torna True se il cursore si e' mosso."""
        if not self.elabora_restanti:
            return False
        mosso = self.successivo()
        if self.cursore >= len(self.frames) - 1 and self.frame_corrente().is_done:
            self.elabora_restanti = False
        return mosso

    # -- il .dat ----------------------------------------------------------

    def salva_sessione(self, path, model_iter):
        """Scrive il .dat senza mutare i frame in memoria: il servizio GUI
        puo' chiamarlo a sessione viva e continuare subito dopo. Il .dat
        non porta ne' percorsi di output ne' immagini, come sempre: qui li
        si azzera solo sulla COPIA che finisce nel pickle."""
        istantanea = []
        for frame in self.frames:
            copia = copy.copy(frame)
            copia.output_filepath = None
            copia.output_mask_filepath = None
            copia.image = None
            istantanea.append(copia)
        dati = {'frames': istantanea,
                'frames_idxs': list(range(self.cursore, len(self.frames))),
                'frames_done_idxs': list(range(self.cursore)),
                'model_iter': model_iter,
                'keyframes': [[i, self.keyframes[i].get_config()] for i in sorted(self.keyframes)]}
        Path(path).write_bytes(pickleex.dumps(dati))

    def carica_sessione(self, path, model_iter):
        try:
            dati = pickle.loads(Path(path).read_bytes())
        except Exception:
            return RIPRESA_NESSUNA
        s_frames = dati.get('frames')
        s_idxs = dati.get('frames_idxs')
        s_done = dati.get('frames_done_idxs')
        s_iter = dati.get('model_iter')
        if s_frames is None or s_idxs is None or s_done is None or s_iter is None \
                or len(s_frames) != len(self.frames):
            return RIPRESA_NON_CORRISPONDE
        for mio, suo in zip(self.frames, s_frames):
            if mio.frame_info.filepath.name != suo.frame_info.filepath.name:
                return RIPRESA_NON_CORRISPONDE
        output = (self.frames[0].output_filepath.parent, self.frames[0].output_mask_filepath.parent)
        for frame in s_frames:
            if frame.cfg is not None:
                frame.cfg = frame.cfg.__class__(**frame.cfg.get_config())
        self.frames = s_frames
        for i, frame in enumerate(self.frames):
            frame.idx = i
            frame.is_processing = False
        self.cursore = s_idxs[0] if s_idxs else len(self.frames) - 1
        self.keyframes = {}
        for idx, config in dati.get('keyframes', []):
            self.keyframes[int(idx)] = self.merger_config.__class__(**config)
        self.assegna_output(*output)
        esito = RIPRESA_OK
        if s_iter != model_iter:
            # il modello si e' allenato ancora: cio' che era fuso non vale piu'
            for frame in self.frames:
                frame.is_done = False
            self.cursore = 0
            esito = RIPRESA_AZZERATA
        elif not s_idxs:
            # il .dat di una fusione FINITA (la CLI vecchia salvava
            # frames_idxs vuoto): torna in testa il cursore, non i frame --
            # i PNG sul disco sono buoni, e rifarli sarebbe l'intera
            # fusione da capo
            self.cursore = 0
        self._riavvolgi_sui_mancanti()
        self.frames[self.cursore].is_shown = False
        return esito
