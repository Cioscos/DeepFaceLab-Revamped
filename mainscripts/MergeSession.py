"""Il servizio di fusione per la GUI: lo stesso nucleo della finestra cv2,
comandato da righe JSON su stdin, che manda eventi su stdout.

Stessa forma di ExtractManual.py (stdout solo per il protocollo, il resto
sotto redirect_stdout(sys.stderr); sorvegliante d'inattivita' con
os._exit) con due differenze: (1) il servizio VIVE dentro
Subprocessor.run() del pool di compositing -- i comandi arrivano da un
thread che legge stdin e li accoda, on_tick li svuota; (2) il sorvegliante
si sospende durante il batch, che per ore non manda comandi.

I frame non viaggiano: il pool scrive merged/<stem>.png e
merged_mask/<stem>.png, `frame_pronto` parte dopo on_result, quindi il
file esiste quando la GUI lo legge.

`chiudi` e' GARBATO: chiede la chiusura, non annulla il batch in corso.
Il ciclo del pool finisce quando non resta lavoro, e solo allora il
servizio salva la sessione e risponde -- cosi' la risposta a `chiudi` e'
sempre l'ultima riga del canale, e non c'e' un evento che la scavalca.
Chi vuole fermare il batch manda prima `batch` con `acceso` falso; chi
vuole troncare tutto uccide il processo (il `ferma()` del client).
"""
import contextlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from core.interact import interact as io
from mainscripts import CanaleComandi, FusioneGuasti
from merger import SessioneMerge
from merger.InteractiveMergerSubprocessor import InteractiveMergerSubprocessor

TIMEOUT_INATTIVITA_S = 300.0
INTERVALLO_SORVEGLIANZA_S = 5.0
INTERVALLO_AVANZAMENTO_S = 1.0


class Attivita:
    """Quando e' arrivato l'ultimo comando, e se il servizio sta lavorando.

    `occupato` copre i due tratti in cui non arriva nessun comando ma il
    servizio non e' affatto inattivo: il batch, che per ore non ne manda
    uno, e il caricamento iniziale (modello + raccolta degli allineati),
    che su un data_dst grande dura piu' del timeout."""

    def __init__(self):
        self.ultimo = time.time()
        # Nasce OCCUPATO: fra qui e il primo giro del ciclo c'e' tutto il
        # caricamento, e il sorvegliante parte prima di lui -- spegnerla
        # dentro `servi` lascerebbe una finestra in cui il conto corre.
        self.occupato = True

    def tocca(self, adesso):
        self.ultimo = adesso


def sorveglia(stato, timeout_s=TIMEOUT_INATTIVITA_S, orologio=None, dormi=None, esci=None):
    """Come ExtractManual.sorveglia, ma `occupato` sospende il conto: una
    fusione di un'ora intera -- e un caricamento di minuti -- non manda un
    solo comando, e senza questa sospensione il servizio si ucciderebbe da
    solo proprio mentre lavora (misurati 23 s di solo caricamento su 600
    fotogrammi: un data_dst grande su un disco lento supera i 300).

    Orologio, attesa e uscita iniettati per poterla provare senza thread
    ne' secondi veri. In produzione `esci` e' os._exit e non sys.exit, che
    su un thread demone verrebbe inghiottito senza fermare niente.
    """
    orologio = orologio or time.time
    dormi = dormi or time.sleep
    esci = esci or (lambda: os._exit(0))
    while True:
        dormi(INTERVALLO_SORVEGLIANZA_S)
        if stato.occupato:
            stato.tocca(orologio())
            continue
        if orologio() - stato.ultimo >= timeout_s:
            esci()
            return


def _rect_dei_landmark(frame):
    """[x0, y0, x1, y1] del primo volto del frame, o None: la lente della
    pagina ritaglia da qui. Interi Python, mai numpy (json)."""
    lista = frame.frame_info.landmarks_list
    if not lista:
        return None
    pts = np.asarray(lista[0], dtype=np.float32)
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [int(x0), int(y0), int(x1), int(y1)]


def _cfg_json(cfg):
    campi = set(SessioneMerge.CAMPI_INTERPOLATI) | set(SessioneMerge.CAMPI_A_SCALINO)
    out = {}
    for k, v in cfg.get_config().items():
        if k in campi:
            out[k] = v.item() if hasattr(v, "item") else v
    return out


def _avvisi_json(avvisi):
    """Gli avvisi della raccolta, come li vede la GUI: dei gruppi con piu'
    allineati viaggiano solo il nome del frame e i nomi degli allineati --
    le sorgenti, terzo elemento della terna, servono alla riga che
    Merger.main stampa in console, non alla pagina."""
    return {"senza_volto": list(avvisi.get("senza_volto") or []),
            "multipli": [[v[0], list(v[1])] for v in (avvisi.get("multipli") or [])]}


class ServizioMerge(InteractiveMergerSubprocessor):
    """Il subprocessor senza schermo: on_tick serve i comandi."""

    def __init__(self, uscita, comandi, stato, avvisi, *args, **kwargs):
        self._uscita = uscita
        self._lucchetto = threading.Lock()
        self._comandi = comandi
        self._stato = stato
        self._avvisi = avvisi
        self._t_batch = None
        self._frames_batch = 0
        self._ultimo_avanzamento = 0.0
        self._barra_aperta = False
        self._ripresa = SessioneMerge.RIPRESA_NESSUNA
        self._id_chiusura = None
        # la pulizia di `merged` la fa `servi` dopo aver caricato il .dat:
        # qui l'esito della ripresa non si conosce ancora
        super().__init__(*args, pulisci_subito=False, **kwargs)

    # -- il canale ------------------------------------------------------

    def manda(self, evento):
        with self._lucchetto:
            self._uscita.write(json.dumps(evento) + "\n")
            self._uscita.flush()

    def _errore(self, ident, codice, motivo):
        self.manda({"op": "error", "id": ident, "codice": codice, "motivo": str(motivo)})

    def _cfg_corrente(self):
        """La cfg del frame mostrato: quella della sessione se il frame non
        ne ha ancora una (un .dat ripreso puo' portarne di vuote)."""
        cur = self.sessione.frame_corrente()
        return _cfg_json(cur.cfg if cur.cfg is not None else self.sessione.merger_config)

    # -- il ciclo -------------------------------------------------------

    #override
    def on_clients_initialized(self):
        s = self.sessione
        s.elabora_restanti = False
        s.in_chiusura = False
        self.manda({"op": "pronto", "frame_totali": len(s.frames), "cursore": s.corrente(),
                    "ripresa": self._ripresa, "avvisi": _avvisi_json(self._avvisi),
                    "fatti_idx": s.stato()["fatti_idx"],
                    "rect": [_rect_dei_landmark(f) for f in s.frames],
                    "cfg": self._cfg_corrente()})

    #override
    def on_clients_finalized(self):
        self._chiudi_barra()
        self.sessione.salva_sessione(self.merger_session_filepath, self.model_iter)
        self.manda({"op": "chiudi", "id": self._id_chiusura, "esito": "ok"})

    #override
    def on_tick(self):
        io.process_messages()
        s = self.sessione
        try:
            while True:
                comando = self._comandi.get_nowait()
                self._stato.tocca(time.time())
                self._servi(comando)
        except queue.Empty:
            pass

        # La barra `Merging` esiste solo mentre il batch gira. Nella
        # finestra `cv2` era aperta per tutta la sessione -- un consuntivo
        # in console -- ma qui lo stesso canale finisce nella pila della
        # GUI, dove una barra aperta appena la sessione e' pronta dice
        # all'utente che la fusione e' partita da sola. Il ciclo dei
        # comandi non e' il posto giusto per aprirla e chiuderla: oltre a
        # `batch`, spengono `elabora_restanti` anche `vai`, `propaga`,
        # `cfg` e `chiudi`, e la barra resterebbe aperta dietro a uno di
        # loro. Qui la bandiera del nucleo e' una sola.
        self._sincronizza_barra(s.elabora_restanti)

        if s.elabora_restanti:
            if self._t_batch is None:
                self._t_batch = time.time()
                self._frames_batch = 0
            s.avanza_batch()
            adesso = time.time()
            if adesso - self._ultimo_avanzamento >= INTERVALLO_AVANZAMENTO_S:
                self._ultimo_avanzamento = adesso
                self.manda({"op": "avanzamento", **self._evento_avanzamento()})
            if not s.elabora_restanti:           # finito da solo
                self._fine_batch()
        self._stato.occupato = s.elabora_restanti

        avanzamento = self._avanzamento()
        delta = avanzamento - self._avanzamento_visto
        if delta:
            if self._barra_aperta:
                io.progress_bar_inc(delta)
            self._avanzamento_visto = avanzamento

        if not self.clis:
            # nessun processo di compositing e' sopravvissuto: restare nel
            # ciclo vorrebbe dire girare a vuoto per sempre.
            # Lo si DICE: da fuori questa chiusura e' identica a quella per
            # stdin chiuso -- stesso `chiudi` senza id, stessa frase sulla
            # pagina -- e senza questa riga sullo stderr non c'e' modo di
            # sapere quale delle due e' stata.
            if not s.in_chiusura:
                io.log_err("chiusura: nessun processo di compositing e' sopravvissuto")
            s.in_chiusura = True
            s.elabora_restanti = False
        return s.in_chiusura and not s.elabora_restanti

    def _sincronizza_barra(self, acceso):
        if acceso:
            if not self._barra_aperta:
                # `initial` col progresso di adesso: il batch di una
                # sessione ripresa non riparte da zero
                io.progress_bar("Merging", len(self.sessione.frames),
                                initial=self._avanzamento())
                self._avanzamento_visto = self._avanzamento()
                self._barra_aperta = True
        else:
            self._chiudi_barra()

    def _chiudi_barra(self):
        if not self._barra_aperta:
            return
        io.progress_bar_close()
        self._barra_aperta = False

    def _evento_avanzamento(self):
        st = self.sessione.stato()
        ms = self._ms_per_frame()
        eta = None if ms is None else int(ms * st["da_fare"] / 1000.0)
        return {"fatti": st["fatti"], "totali": st["frame_totali"],
                "ms_per_frame": ms, "eta_s": eta}

    def _ms_per_frame(self):
        if self._t_batch is None or self._frames_batch == 0:
            return None
        return (time.time() - self._t_batch) * 1000.0 / self._frames_batch

    def _fine_batch(self):
        s = self.sessione
        avvisi = _avvisi_json(self._avvisi)      # copie, mai le liste della raccolta
        rapporto = {"senza_volto": avvisi["senza_volto"],
                    "multipli": avvisi["multipli"],
                    "ms_per_frame": self._ms_per_frame() or 0.0,
                    "keyframes": [[i, _cfg_json(c)] for i, c in sorted(s.keyframes.items())]}
        try:
            Path(self.output_path, "merge_report.json").write_text(json.dumps(rapporto, indent=1))
        except OSError as e:
            # non e' un guasto con un codice: il rapporto viaggia comunque
            # nell'evento, e la pagina lo mostra senza leggere il file
            io.log_err("rapporto non scritto: %s" % e)
        self.manda({"op": "rapporto", **rapporto})
        self._chiudi_barra()
        self._t_batch = None

    #override
    def on_result(self, host_dict, pf_sent, pf_result):
        s = self.sessione
        s.su_risultato(pf_result.idx, pf_result.cfg, None)   # niente immagine in RAM: la GUI legge il PNG
        if s.frames[pf_result.idx].is_done:
            self._frames_batch += 1
            self.manda({"op": "frame_pronto", "idx": pf_result.idx,
                        "cfg": _cfg_json(pf_result.cfg),
                        "senza_volto": len(pf_result.frame_info.landmarks_list) == 0})

    #override
    def on_data_return(self, host_dict, pf):
        """Il client che aveva questo frame e' morto. Che sia caduto lo si
        SA -- ed e' CLIENT_CADUTO --, di che morte no: Subprocessor passa il
        dato, non il motivo, quindi un MemoryError (il SilenceException di
        Cli.process_data) e un errore qualunque di compositing arrivano
        identici, e il motivo grezzo resta accanto al codice. Il dettaglio
        della caduta e' sullo stderr, dove SubprocessorBase lo scrive.

        Non e' la fine della sessione: il frame torna in coda (`su_ritorno`
        del nucleo) e gli altri processi del pool continuano, come nella
        finestra cv2. L'evento porta l'indice perche' la pagina possa
        rimetterlo «da fare» sulla timeline."""
        super().on_data_return(host_dict, pf)
        self.manda({"op": "error", "id": None, "codice": FusioneGuasti.CLIENT_CADUTO,
                    "idx": pf.idx,
                    "motivo": "il processo che fondeva %s e' caduto" % pf.frame_info.filepath.name})

    #override
    def get_data(self, host_dict):
        if self.sessione.in_chiusura and not self.sessione.elabora_restanti:
            return None
        return self.sessione.da_elaborare()

    # -- i comandi ------------------------------------------------------

    def _servi(self, comando):
        s = self.sessione
        ident = comando.get("id")
        op = comando.get("op")
        try:
            if op == "stato":
                self.manda({"op": "stato", "id": ident, **s.stato(), "cfg": self._cfg_corrente()})
            elif op == "vai":
                s.elabora_restanti = False
                s.vai(int(comando["idx"]))
                self.manda({"op": "vai", "id": ident, **s.stato(), "cfg": self._cfg_corrente()})
            elif op == "cfg":
                if "idx" in comando:
                    s.vai(int(comando["idx"]))
                s.imposta_cfg(dict(comando.get("campi") or {}))
                self.manda({"op": "cfg", "id": ident, **s.stato(), "cfg": self._cfg_corrente()})
            elif op == "propaga":
                tutti = comando.get("fino_a") == "tutti"
                s.elabora_restanti = False
                if comando.get("verso") == "indietro":
                    s.precedente(propaga=True, fino_al_primo=tutti)
                else:
                    s.successivo(propaga=True, fino_all_ultimo=tutti)
                self.manda({"op": "propaga", "id": ident, **s.stato(), "cfg": self._cfg_corrente()})
            elif op == "batch":
                s.imposta_batch(bool(comando.get("acceso")))
                if not s.elabora_restanti:
                    self._t_batch = None
                self.manda({"op": "batch", "id": ident, **s.stato()})
            elif op == "keyframe":
                idx = int(comando["idx"])
                if comando.get("acceso"):
                    s.imposta_keyframe(idx)
                else:
                    s.togli_keyframe(idx)
                self.manda({"op": "keyframe", "id": ident, **s.stato(), "cfg": self._cfg_corrente(),
                            "acceso": idx in s.keyframes})
            elif op == "piano":
                anteprima = bool(comando.get("anteprima", False))
                t0 = time.perf_counter()
                da_rifare = s.applica_piano(interpola=bool(comando.get("interpola", True)),
                                            anteprima=anteprima)
                # sullo stderr, come tutto cio' che non e' protocollo: il
                # piano cammina tutti i frame, e quanto costi si legge solo
                # qui. L'anteprima si dichiara: la pagina la chiede a ogni
                # debounce, e chi legge il log deve distinguere «ho contato»
                # da «ho riscritto la cfg di N frame».
                io.log_info("piano%s: %d frame in %.3f s"
                            % (" (anteprima)" if anteprima else "", len(s.frames),
                               time.perf_counter() - t0))
                self.manda({"op": "piano", "id": ident, **s.stato(), "cfg": self._cfg_corrente(),
                            "da_rifare": da_rifare})
            elif op == "sonda":
                indici = s.sonda(int(comando.get("n", 8)))
                gia_fatti = [i for i in indici if s.frames[i].is_done]
                self.manda({"op": "sonda", "id": ident, **s.stato(), "cfg": self._cfg_corrente(),
                            "indici": indici, "gia_fatti": gia_fatti})
            elif op == "salva_sessione":
                s.salva_sessione(self.merger_session_filepath, self.model_iter)
                self.manda({"op": "salva_sessione", "id": ident, "esito": "ok"})
            elif op == "chiudi":
                # il batch si spegne: chi chiude un video da un'ora non
                # aspetta l'ora. Quel che e' gia' in volo si finisce, e la
                # risposta parte da on_clients_finalized a sessione salvata.
                # La prima chiusura vince: dopo un `chiudi` dell'utente il
                # thread di stdin ne accoda un altro senza id quando la GUI
                # muore, e la risposta deve portare l'id di chi ha chiesto
                if not s.in_chiusura:
                    self._id_chiusura = ident
                s.elabora_restanti = False
                self._t_batch = None
                s.in_chiusura = True
                # la risposta parte da on_clients_finalized, a sessione salvata
            else:
                self._errore(ident, FusioneGuasti.COMANDO_SCONOSCIUTO, "operazione sconosciuta: %s" % op)
        except Exception as e:
            # nessun codice inventato: un guasto che non e' fra quelli
            # previsti viaggia col solo motivo, e la GUI ha il suo ripiego
            self._errore(ident, None, e)


def _leggi_comandi(entrata, coda, avvio=None):
    """Legge le righe di comando fino a EOF, poi accoda la chiusura.

    `avvio` e' l'istante da cui contare nei messaggi: senza un tempo,
    «stdin chiuso» non dice se il tubo e' nato morto o si e' chiuso a
    meta' caricamento -- due guasti diversi con lo stesso sintomo."""
    quando = (lambda: "") if avvio is None else \
             (lambda: " (a %.1f s dall'avvio)" % (time.monotonic() - avvio))
    prima = True
    for riga in entrata:
        riga = riga.strip()
        if not riga:
            continue
        if prima:
            # Che il PRIMO comando sia arrivato distingue «il tubo non ha
            # mai funzionato» da «si e' chiuso dopo»: senza questa riga i
            # due casi hanno lo stesso log.
            io.log_info("primo comando ricevuto%s" % quando())
            prima = False
        try:
            coda.put(json.loads(riga))
        except ValueError:
            coda.put({"op": None, "id": None})
    # Il gemello della riga in on_tick: le due chiusure spontanee si
    # distinguono solo qui.
    io.log_err("chiusura: stdin chiuso, la GUI non parla piu'%s%s"
               % (quando(), "" if not prima else " -- e non era mai arrivato un comando"))
    coda.put({"op": "chiudi", "id": None})


def servi(entrata, uscita, costruisci, argomenti, stato=None, coda=None):
    """Tutto il servizio, con il costruttore del modello iniettato.

    `coda` gia' piena di comandi (e con il suo lettore gia' vivo) e' come
    lavora `main`: il lettore nasce PRIMA del caricamento del modello,
    perche' e' li' dentro che il tubo si e' visto morire. Senza, se la
    apre lui -- e' cosi' che lavorano i test."""
    stato = stato or Attivita()

    def _manda(evento):
        uscita.write(json.dumps(evento) + "\n")
        uscita.flush()

    # Il sorvegliante gira gia' (main lo avvia prima di qui) e conta i
    # secondi senza comandi: costruire il modello e raccogliere gli
    # allineati non ne manda nessuno, e il servizio si ucciderebbe a meta'
    # caricamento senza dire niente a nessuno. La bandiera nasce accesa,
    # questa riga la riaccende per chi passa uno `stato` gia' usato.
    stato.occupato = True
    with contextlib.redirect_stdout(sys.stderr):
        input_path = Path(argomenti.input_dir)
        aligned_path = Path(argomenti.aligned_dir)
        if not aligned_path.exists():
            _manda({"op": "error", "id": None, "codice": FusioneGuasti.ALLINEATI_ASSENTI,
                    "motivo": "cartella degli allineati assente: %s" % aligned_path})
            return
        if not Path(argomenti.model_dir).exists():
            _manda({"op": "error", "id": None, "codice": FusioneGuasti.MODELLO_ASSENTE,
                    "motivo": "cartella del modello assente: %s" % argomenti.model_dir})
            return
        for cartella in (argomenti.output_dir, argomenti.output_mask_dir):
            Path(cartella).mkdir(parents=True, exist_ok=True)

        try:
            predictor_func, shape, cfg, model_iter, session_path, enhancer, xseg = costruisci(argomenti)
        except Exception as e:
            _manda({"op": "error", "id": None, "codice": FusioneGuasti.MODELLO_ASSENTE, "motivo": str(e)})
            return

        from merger import preparazione
        frames, avvisi = preparazione.raccogli_frames(input_path, aligned_path)
        if len(frames) == 0:
            _manda({"op": "error", "id": None, "codice": FusioneGuasti.NESSUN_FRAME,
                    "motivo": "nessuna immagine in %s" % input_path})
            return

        if coda is None:
            coda = queue.Queue()
            threading.Thread(target=_leggi_comandi, args=(entrata, coda), daemon=True).start()

        servizio = ServizioMerge(uscita, coda, stato, avvisi,
                                 is_interactive=False,
                                 merger_session_filepath=session_path,
                                 predictor_func=predictor_func,
                                 predictor_input_shape=shape,
                                 face_enhancer_func=enhancer,
                                 xseg_256_extract_func=xseg,
                                 merger_config=cfg,
                                 frames=frames,
                                 frames_root_path=input_path,
                                 output_path=Path(argomenti.output_dir),
                                 output_mask_path=Path(argomenti.output_mask_dir),
                                 model_iter=model_iter,
                                 subprocess_count=int(argomenti.workers))
        if Path(session_path).exists():
            # una sessione che non corrisponde lascia il nucleo intatto
            # (carica_sessione esce prima di toccarlo): resta la sessione
            # nuova, e il `pronto` lo dice con `ripresa`
            servizio._ripresa = servizio.sessione.carica_sessione(session_path, model_iter)
        # come la CLI, ma DOPO il caricamento del .dat, che qui non sta nel
        # costruttore: senza, i PNG della fusione precedente restavano in
        # `merged` e finivano nel video di «merged to mp4»
        servizio.pulisci_uscita(servizio._ripresa)
        servizio._avanzamento_visto = servizio._avanzamento()
        # da qui in poi lo stato dell'occupazione lo tiene on_tick, che sa
        # se un batch e' in corso
        stato.occupato = False
        stato.tocca(time.time())
        servizio.run()


def _costruisci_modello(argomenti, force_gpu_idxs=None):
    """Il modello vero, come Merger.main; qui e non in servi() perche' i
    test iniettano un predittore finto.

    Gli indici delle GPU arrivano gia' convertiti in lista di interi, come
    per ogni altro verbo: la conversione dalla stringa che argparse
    consegna sta in main.py, in un posto solo, ed e' una regola legata.
    """
    import models
    from core.joblib import MPClassFuncOnDemand, MPFunc
    from core.leras import nn
    from facelib import FaceEnhancer, XSegNet
    saved_models_path = Path(argomenti.model_dir)
    model = models.import_model(argomenti.model_name)(is_training=False,
                                                      saved_models_path=saved_models_path,
                                                      force_gpu_idxs=force_gpu_idxs,
                                                      force_model_name=argomenti.force_model_name,
                                                      cpu_only=False)
    predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()
    predictor_func = MPFunc(predictor_func)
    run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
    xseg = MPClassFuncOnDemand(XSegNet, 'extract', name='XSeg', resolution=256,
                               weights_file_root=saved_models_path, place_model_on_cpu=True, run_on_cpu=run_on_cpu)
    enhancer = MPClassFuncOnDemand(FaceEnhancer, 'enhance', place_model_on_cpu=True, run_on_cpu=run_on_cpu)
    return (predictor_func, predictor_input_shape, cfg, model.get_iter(),
            model.get_strpath_storage_for_file('merger_session.dat'), enhancer, xseg)


def _canale_del_protocollo():
    """Lo stdout vero per il protocollo, e il descrittore 1 dirottato sullo
    stderr.

    redirect_stdout ribatte `sys.stdout` del solo processo che lo esegue: i
    processi di compositing sono figli, e una stampa di libreria dentro uno
    di loro andrebbe sul descrittore 1 ereditato -- cioe' in mezzo alle
    righe JSON, che il client non saprebbe piu' leggere.
    """
    canale = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    return canale


def main(argomenti, force_gpu_idxs=None):
    # I due canali PRIMA di ogni altra cosa: dopo, il caricamento del
    # modello passa da codice che ri-avvolge sia sys.stdout sia sys.stdin.
    avvio = time.monotonic()
    entrata = CanaleComandi.apri()
    canale = _canale_del_protocollo()
    stato = Attivita()
    threading.Thread(target=sorveglia, args=(stato,), daemon=True).start()
    # Il lettore PRIMA del modello: cosi' nessun comando mandato durante il
    # caricamento va perso, e il momento in cui il tubo muore si legge dal
    # log invece di dedurlo.
    coda = queue.Queue()
    threading.Thread(target=_leggi_comandi, args=(entrata, coda, avvio), daemon=True).start()
    servi(entrata, canale,
          lambda a: _costruisci_modello(a, force_gpu_idxs), argomenti, stato, coda=coda)
