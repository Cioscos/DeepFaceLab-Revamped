"""
SAEHDX: SAEHD con il cablaggio riscritto per le prestazioni.

Le reti, le loss e il grafo sono quelli di SAEHD, importati -- non copiati: una
copia divergerebbe al primo bugfix, e questo modello diventerebbe un SAEHD
vecchio invece che un SAEHD veloce. Quello che cambia e' il contorno del passo:
formato di memoria dei pesi, precisione dei kernel, dove vivono le loss, come
arriva il batch.

Salvataggio, caricamento pesi, anteprime ed export sono ereditati da
SAEHDModel e non compaiono qui. Anche il prompt delle opzioni e' quasi tutto
ereditato: l'unica opzione propria di questo modello e' cudnn.benchmark,
spiegata sotto `on_initialize_options`.
"""
import importlib.util
import logging
import os

import numpy as np
import torch

from core.interact import interact as io
from core.leras import nn
from models.Model_SAEHD.Model import SAEHDModel, saehd_train_step
from models.Model_SAEHDX import prerequisiti_windows
from samplelib import SampleGeneratorFace


class SAEHDXModel(SAEHDModel):
    # bf16 e non fp16: su Ampere il throughput e' lo stesso, ma bf16 ha
    # l'esponente di fp32 e non ha bisogno di un GradScaler -- che questo
    # cablaggio non ha, e la cui assenza con fp16 non solleverebbe nulla,
    # produrrebbe NaN dopo qualche migliaio di iterazioni.
    DTYPE_AUTOCAST = torch.bfloat16

    # Ogni quante iterazioni le loss tornano sull'host. A 1 sarebbe la
    # barriera di prima: il .cpu() blocca la CPU finche' la GPU non ha finito,
    # e nessuna delle leve seguenti puo' sovrapporsi a niente. La console le
    # mostra con un giro di ritardo, cosa che nessuno nota.
    INTERVALLO_SCARICO_LOSS = 10

    # Il precaricamento del batch successivo su uno stream secondario. E' un
    # attributo e non una costante scritta nel ramo perche' una misura che
    # confronta con e senza deve poterlo spegnere sull'istanza senza toccare il
    # sorgente ne' lasciare stato dietro di se'.
    #
    # Si spegne fra la costruzione e la prima iterazione, non a meta' corsa: la
    # macchina a stati del precaricamento (_prossimo, _parita, gli eventi per
    # parita') non ha un ripristino, e spegnerlo dopo che _prossimo e' pieno
    # abbandonerebbe un batch gia' caricato -- il passo seguente rileggerebbe
    # dai generatori, e quel batch verrebbe semplicemente saltato. Un ripristino
    # non c'e' perche' nessuno ha bisogno di cambiare idea a meta' corsa.
    PREFETCH = True

    # Il passo registrato una volta in un CUDA graph e rieseguito con replay().
    # Attributo e non solo voce di self.options per la stessa ragione di
    # PREFETCH: una misura che confronta con e senza deve poterlo accendere
    # sull'istanza. on_initialize lo allinea a self.options['cuda_graph'].
    CUDA_GRAPH = False

    # Il passo compilato da torch.compile invece che interpretato a ogni giro.
    # Attributo per la stessa ragione dei due sopra: una misura che confronta
    # con e senza deve poterlo accendere sull'istanza dopo la costruzione.
    # on_initialize lo allinea a self.options['torch_compile'].
    TORCH_COMPILE = False

    # La funzione compilata, quando c'e'. Vive come attributo di classe a None
    # cosi' che il passo possa leggerlo anche su un'istanza che non e' passata
    # da onTrainOneIter: e' anche il campo da cui si legge, dall'esterno, se la
    # compilazione e' davvero avvenuta o se la guardia l'ha rifiutata.
    _passo_compilato = None

    # Se il passo compilato ha gia' girato almeno una volta. Non e' un
    # dettaglio contabile: `torch.compile` e' pigro, e la generazione vera dei
    # kernel avviene alla *prima esecuzione*, non alla compilazione. E' li' che
    # si spende la memoria e li' che si scopre se questa macchina ce la fa.
    _compilazione_collaudata = False

    # Quante iterazioni ordinarie girano prima della cattura. Servono a far
    # arrivare l'ottimizzatore a regime: gli accumulatori di AdaBelief nascono
    # gia' allocati in initialize_variables, ma il ramo _foreach_ costruisce le
    # proprie liste al primo passo, e il grafo registra indirizzi -- non
    # operazioni su indirizzi che cambiano.
    ITERAZIONI_PRIMA_DELLA_CATTURA = 3

    @staticmethod
    def abilita_backend_veloce(cudnn_benchmark):
        """
        allow_tf32 sui matmul e' spento per default da torch 1.12 in poi;
        sulle conv cuDNN lo usa gia' -- qui va sempre acceso, non e' opzionale.

        cudnn.benchmark invece e' un parametro, non piu' un valore fisso: le
        shape sono fisse per tutta la corsa quindi l'autotuner di cuDNN
        pagherebbe una volta sola, ma sceglie l'algoritmo piu' veloce anche a
        costo di un workspace molto piu' grande -- misurato da solo, su un
        modello altrimenti fp32: piu' che raddoppia la memoria di picco
        (6462 -> 14154 MiB). Con le altre tre leve gia' accese il guadagno di
        velocita' che aggiunge sopra channels_last+autocast e' rumoroso da
        misurare (dieci corse ripetute davano fra il 22% e il 29% di guadagno
        su T_tot rispetto a SAEHD, contro il 32% con anche questo flag acceso
        -- la differenza fra le due code e' dentro il rumore della macchina,
        non un numero pulito). Non essendo un guadagno chiaramente
        superiore al rumore, resta un'opzione e non un default: chi ha VRAM da
        spendere la accende dal prompt di on_initialize_options, chi no la
        lascia spenta.

        Sta in un metodo statico e non in nn.initialize perche' e' una scelta
        di QUESTO modello: accenderlo per tutti cambierebbe i bit di SAEHD e
        di AMP, che devono restare numericamente invariati.
        """
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _risali_a_fp32(_modulo, _ingresso, uscita):
        """
        Hook di forward registrato su ogni rete: riporta l'uscita di QUESTA
        rete a fp32, lasciando bf16 tutto cio' che sta dentro (le conv/matmul
        interne restano sotto autocast e prendono comunque il tensor core).

        Serve perche' saehd_train_step confronta l'uscita delle reti con i
        target grezzi, che entrano sempre in fp32 -- to_t li porta a
        nn.floatx (models/Model_SAEHD/Model.py) senza che autocast possa
        intervenire, perche' un .to() esplicito non e' fra le op che
        autocast intercetta. dssim() (core/leras/ops) pretende gli stessi
        dtype sui due argomenti, contratto testato e non toccato: senza
        questo hook la prima dssim della loss lo viola e solleva subito,
        prima ancora di poter misurare qualunque guadagno.
        """
        if isinstance(uscita, torch.Tensor):
            return uscita.float() if uscita.dtype != torch.float32 else None
        if isinstance(uscita, tuple):
            if not any(isinstance(t, torch.Tensor) and t.dtype != torch.float32 for t in uscita):
                return None
            return tuple(t.float() if isinstance(t, torch.Tensor) and t.dtype != torch.float32 else t
                         for t in uscita)
        return None

    #override
    def on_initialize_options(self):
        super().on_initialize_options()

        # Solo al primo avvio, come resolution/archi (Model_SAEHD/Model.py):
        # e' una scelta strutturale di questa corsa, non un iperparametro che
        # ha senso ridiscutere ogni volta che si riprende il training -- e non
        # tocca i pesi salvati, quindi non serve nemmeno passare da
        # ask_override() per cambiarla fra una sessione e l'altra.
        default_cudnn_benchmark = self.options['cudnn_benchmark'] = \
            self.load_or_def_option('cudnn_benchmark', False)

        if self.is_first_run():
            self.options['cudnn_benchmark'] = io.input_bool(
                "Enable cudnn.benchmark", default_cudnn_benchmark,
                help_message="Lets cuDNN autotune the fastest convolution "
                             "algorithm for the fixed shapes of this run. The "
                             "extra speed on top of the other SAEHDX "
                             "optimizations was too small to tell apart from "
                             "measurement noise, while the extra VRAM was "
                             "consistent and large. Off by default: on recent, VRAM-"
                             "constrained cards the trade is usually not worth it.")

        default_cuda_graph = self.options['cuda_graph'] = \
            self.load_or_def_option('cuda_graph', False)

        if self.is_first_run():
            self.options['cuda_graph'] = io.input_bool(
                "Enable CUDA graph capture", default_cuda_graph,
                help_message="Records the training step once as a CUDA graph "
                             "and replays it, instead of re-issuing every "
                             "kernel from Python each iteration. It needs "
                             "fixed input buffers, so it turns the batch "
                             "prefetch off, and it only applies to a plain "
                             "src_dst step: with GAN, true face power, "
                             "learning rate dropout or clipgrad the capture is "
                             "skipped and training goes on unchanged. Off by "
                             "default.")

        default_torch_compile = self.options['torch_compile'] = \
            self.load_or_def_option('torch_compile', False)

        if self.is_first_run():
            self.options['torch_compile'] = io.input_bool(
                "Enable torch.compile", default_torch_compile,
                help_message="Compiles the training step ahead of time instead "
                             "of interpreting it kernel by kernel every "
                             "iteration. Verified here over a 5000-iteration "
                             "run: convergence is indistinguishable from the "
                             "uncompiled model, and an iteration takes about a "
                             "quarter less. It buys speed with memory, and "
                             "mostly with host RAM: measured here the peak "
                             "resident memory of the run nearly doubled (+7.1 "
                             "GiB, held for the whole run) and the allocated "
                             "VRAM grew by 15%. It needs a working Triton "
                             "backend and, on Windows, the MSVC compiler on "
                             "PATH: where either is missing the option turns "
                             "itself off and training goes on unchanged. The "
                             "first iteration also costs the compilation, "
                             "around half a minute. Not combined with the CUDA "
                             "graph. Off by default.")

        self._offri_i_prerequisiti_se_servono()

    #override
    def on_initialize(self):
        # Il sito di costruzione dei generatori vive nel genitore e non si
        # tocca: l'opt-in dei filename passa dal default di classe, acceso
        # solo per la durata della costruzione.
        SampleGeneratorFace.default_return_filenames = True
        try:
            super().on_initialize()
        finally:
            SampleGeneratorFace.default_return_filenames = False

        # `is_training` e non solo il device: tutto quello che segue avvolge il
        # passo di addestramento, e nel ramo di merge quel passo non esiste --
        # SAEHDModel definisce src_dst_train solo dentro il proprio
        # `if self.is_training:` (Model_SAEHD/Model.py:972), e al suo posto ci
        # mette AE_merge. Il merger e' l'unico ingresso non-addestrante che
        # costruisce il modello sulla GPU: export_dfm sfugge di qui solo perche'
        # ExportDFM.main cabla cpu_only=True, e sulla CPU il blocco non si
        # attiva. In merge SAEHDX torna a essere SAEHD, che e' anche l'unico
        # comportamento misurato: la convergenza del cablaggio veloce e'
        # verificata sull'addestramento, l'inferenza non e' mai stata confrontata
        # bit a bit con quella del genitore, e allow_tf32 la cambierebbe.
        if self.is_training and nn.device is not None and nn.device.type == "cuda":
            self.abilita_backend_veloce(self.options['cudnn_benchmark'])

            for rete, _ in self.model_filename_list:
                if isinstance(rete, torch.nn.Module):
                    rete.to(memory_format=torch.channels_last)
                    rete.register_forward_hook(self._risali_a_fp32)

            # train_cfg di SAEHDModel.on_initialize e' una variabile locale
            # (Model_SAEHD/Model.py:990-999), non un attributo: qui va
            # ricostruito leggendo self.options e i pochi attributi che
            # SAEHDModel.on_initialize salva su self (archi_type, resolution,
            # gan_power, pretrain -- gia' scontato il caso pretrain per
            # gan_power, che il genitore azzera alla riga 821). Stessi dieci
            # campi, stessi valori dell'originale.
            self._train_cfg = {'archi_type'      : self.archi_type,
                               'resolution'      : self.resolution,
                               'masked_training' : self.options['masked_training'],
                               'eyes_mouth_prio' : self.options['eyes_mouth_prio'],
                               'blur_out_mask'   : self.options['blur_out_mask'],
                               'gan_power'       : self.gan_power,
                               'true_face_power' : self.options['true_face_power'],
                               'face_style_power': self.options['face_style_power'],
                               'bg_style_power'  : self.options['bg_style_power'],
                               'pretrain'        : self.pretrain}

            self.inizializza_prefetch()
            self.inizializza_grafo()
            self.CUDA_GRAPH = self.options['cuda_graph']
            self.TORCH_COMPILE = self.options['torch_compile']

            def src_dst_train_su_device(*batch):
                """Copie e passo sullo stesso stream, senza sovrapposizione.

                E' il percorso di prima del precaricamento, e resta raggiungibile
                con PREFETCH spento: serve a misurare la leva contro se stessa.
                """
                return self._passo_su_tensori(
                    [self._in_pinned(x, (i, 0)) for i, x in enumerate(batch)])

            self.src_dst_train_su_device = src_dst_train_su_device

            passo_originale = self.src_dst_train

            def src_dst_train_veloce(*batch):
                with torch.autocast("cuda", dtype=self.DTYPE_AUTOCAST):
                    return passo_originale(*batch)

            self.src_dst_train = src_dst_train_veloce

    def inizializza_prefetch(self):
        """
        Doppio buffer di staging piu' lo stream secondario su cui caricare.

        **Perche' due buffer e non uno.** Mentre la GPU esegue il passo del
        batch N, il DMA che porta N sul device puo' non aver ancora finito di
        leggere i buffer pinnati di N; riempire quelli di N+1 nelle stesse
        pagine li riscriverebbe sotto il naso del driver, e il passo
        troverebbe meta' di un batch e meta' dell'altro senza che niente
        sollevi. La parita' alterna a ogni caricamento, e ogni parita' ricorda
        l'evento del proprio ultimo caricamento: prima di riscriverla la si
        aspetta. A regime quell'evento e' completo da un pezzo e l'attesa non
        costa niente -- e' la rete di sicurezza per quando la CPU corre piu'
        avanti di due giri, non un costo del percorso normale.

        Le forme sono fisse per tutta la corsa, quindi i buffer si allocano
        una volta sola: due per posizione nel batch invece di uno, cioe' il
        doppio di memoria pinnata sull'host -- non sulla GPU.
        """
        self._staging = {}
        self._eventi_staging = {}
        self._parita = 0
        self._prossimo = None
        self._stream_prefetch = torch.cuda.Stream()

    def inizializza_grafo(self):
        """Lo stato della cattura, tutto spento: il grafo nasce alla prima
        iterazione utile e non qui, perche' catturare vuole un batch vero e
        l'ottimizzatore gia' passato per qualche aggiornamento."""
        self._grafo = None
        self._tensori_statici = None
        self._loss_statiche = None
        self._giri_fatti = 0
        # Il valore che PREFETCH aveva prima che il grafo lo spegnesse, cosi'
        # una rinuncia lo restituisce com'era invece di riaccenderlo per
        # decreto: chi misura il precaricamento contro se stesso lo spegne
        # sull'istanza, e riaccenderglielo sotto il naso falserebbe la corsa.
        self._prefetch_sospeso = None

    def _rinuncia_al_grafo(self, motivo):
        """
        Torna al percorso senza grafo, lasciando detto perche'.

        Rimette il precaricamento com'era: senza questo, chi accende
        `cuda_graph` su una configurazione non catturabile farebbe l'intera
        corsa sul ramo *senza* sovrapposizione -- misurato 3.12% piu' lento del
        percorso ordinario -- cioe' verrebbe punito per aver chiesto
        un'ottimizzazione che non si e' potuta fare. Al momento della rinuncia
        `_prossimo` e' sempre None (il precaricamento non ha mai girato da
        quando e' stato sospeso), quindi non resta nessun batch abbandonato per
        strada.

        Azzera anche i buffer statici e le loss statiche: dopo una cattura
        fallita a meta' possono esistere e non voler dire piu' niente.
        """
        self._grafo = None
        self._tensori_statici = None
        self._loss_statiche = None
        self.CUDA_GRAPH = False

        if self._prefetch_sospeso is not None:
            self.PREFETCH = self._prefetch_sospeso
            self._prefetch_sospeso = None

        io.log_info(f"cuda_graph: {motivo}. "
                    f"L'addestramento prosegue senza grafo.")

    def _staging_pinnato(self, x, chiave):
        """Il buffer pinnato riusato per (posizione nel batch, parita), gia'
        riempito con x. Non tocca il device: chi chiama decide se mandarlo con
        un .to() che alloca o con un copy_ dentro un buffer che esiste gia'."""
        buf = self._staging.get(chiave)
        if buf is None or buf.shape != x.shape:
            buf = torch.empty(x.shape, dtype=torch.float32, pin_memory=True)
            self._staging[chiave] = buf
        buf.copy_(torch.from_numpy(np.ascontiguousarray(x)))
        return buf

    def _in_pinned(self, x, chiave):
        """
        Copia x in un buffer pinnato riusato e lo manda sul device senza
        bloccare. La chiave e' (posizione nel batch, parita).

        Senza memoria pinnata `non_blocking=True` e' una bugia: torch lo
        accetta e la copia resta sincrona, perche' il driver non puo' fare DMA
        da pagine che il kernel puo' spostare. Da sola l'asincronia non
        guadagnava niente, ed era previsto: sullo stesso stream del calcolo non
        c'e' niente con cui sovrapporsi. La copia host->device pesa 2.0 ms, cioe'
        l'1.4% dei ~140 ms che dura oggi un'iterazione di questo cablaggio --
        e infatti misurata da sola dava fra 0.5% e 1.0%, dentro il rumore.
        La memoria pinnata non e' un'ottimizzazione a se': e' il prerequisito
        del caricamento su uno stream separato, che e' cio' che sovrappone
        davvero.
        """
        return self._staging_pinnato(x, chiave).to(nn.device, non_blocking=True)

    def _carica_batch(self, batch):
        """
        Porta il batch sul device sullo stream secondario e ritorna
        (tensori, evento). L'evento e' cio' che il passo aspettera': registrato
        dopo le copie, si compie quando l'ultima ha finito.
        """
        parita = self._parita
        self._parita = 1 - parita

        ultimo = self._eventi_staging.get(parita)
        if ultimo is not None:
            ultimo.synchronize()

        with torch.cuda.stream(self._stream_prefetch):
            tensori = [self._in_pinned(x, (i, parita))
                       for i, x in enumerate(batch)]
            # blocking=True riguarda solo l'attesa dall'host, cioe' il
            # synchronize() qui sopra al giro dopo: senza, la CPU aspetterebbe
            # girando a vuoto su un core, e quel core serve ad accodare kernel.
            # L'attesa dal device (wait_event, in onTrainOneIter) non guarda
            # questo flag, quindi un evento solo va bene per entrambe.
            evento = torch.cuda.Event(blocking=True)
            evento.record(self._stream_prefetch)

        self._eventi_staging[parita] = evento
        return tensori, evento

    def _carica_nei_statici(self, batch):
        """
        Il caricamento del batch quando il passo e' un grafo: `copy_` dentro
        buffer device che esistono gia', invece del `.to()` di `_in_pinned`
        che ne alloca di nuovi a ogni giro.

        E' il vincolo che il grafo impone e non una preferenza: un grafo
        registra *indirizzi*, non variabili. Se il batch atterrasse ogni volta
        in tensori nuovi, il replay continuerebbe a leggere quelli della
        cattura -- lo stesso batch per sempre, senza che niente sollevi.

        Le copie vanno sullo stream corrente, lo stesso su cui gira il replay,
        quindi l'ordine fra le due cose e' garantito senza sincronizzare
        niente. La guardia per parita' resta necessaria per l'altro verso: e'
        l'host che non deve riscrivere un buffer pinnato mentre la DMA
        precedente lo sta ancora leggendo.
        """
        parita = self._parita
        self._parita = 1 - parita

        ultimo = self._eventi_staging.get(parita)
        if ultimo is not None:
            ultimo.synchronize()

        for i, x in enumerate(batch):
            self._tensori_statici[i].copy_(
                self._staging_pinnato(x, (i, parita)), non_blocking=True)

        evento = torch.cuda.Event(blocking=True)
        evento.record(torch.cuda.current_stream())
        self._eventi_staging[parita] = evento

    def _motivo_per_non_catturare(self):
        """
        Le condizioni sotto cui la cattura sarebbe scorretta, non lenta.
        None se si puo' catturare.

        Ognuna e' un pezzo di passo che *cambia* fra un'iterazione e l'altra
        pur non essendo un tensore di ingresso, cioe' esattamente cio' che un
        grafo congela senza dirlo:

        - fuori dal ramo `_foreach_` gli accumulatori di AdaBelief stanno sulla
          CPU, e le operazioni sulla CPU dentro una cattura girano una volta
          sola: al replay non riaccadono affatto;
        - lr_dropout sorteggia una maschera nuova a ogni passo, e catturarla
          significherebbe riusare per sempre la prima;
        - lr_cos rilegge il contatore delle iterazioni dall'host per calcolare
          il learning rate, che quindi resterebbe quello del giorno della
          cattura;
        - clipgrad calcola una norma e la usa in un ramo Python;
        - con GAN o true face power il passo non e' piu' uno solo, e i due
          ottimizzatori dei discriminatori restano fuori dal grafo.
        """
        opt = self.src_dst_opt
        if not getattr(opt, '_fused_path', False):
            return "gli accumulatori dell'ottimizzatore non stanno sul device"
        if opt.lr_dropout != 1.0:
            return "lr_dropout sorteggia una maschera a ogni passo"
        if opt.lr_cos != 0:
            return "lr_cos rilegge il contatore delle iterazioni a ogni passo"
        if opt.clipnorm != 0.0:
            return "clipgrad calcola una norma e la usa in un ramo Python"
        if self._train_cfg['gan_power'] != 0:
            return "gan_power aggiunge un passo che il grafo non contiene"
        if self._train_cfg['true_face_power'] != 0:
            return "true_face_power aggiunge un passo che il grafo non contiene"
        return None

    def _cattura_il_passo(self, batch):
        """
        Scaldata su stream secondario e cattura, nell'ordine che torch
        richiede: la scaldata alloca i workspace di cuDNN e fa costruire al
        ramo `_foreach_` le proprie liste, cosi' dentro la cattura non nasce
        piu' niente di nuovo.

        La scaldata e' fatta di tre aggiornamenti veri sullo stesso batch --
        non si simula un passo, lo si esegue -- mentre la cattura *registra e
        non esegue*: dentro `torch.cuda.graph` i kernel finiscono nel grafo e
        nessuno di essi gira, quindi i pesi restano quelli lasciati dalla
        scaldata. Sono tre iterazioni ordinarie su un batch solo, una volta per
        corsa: trascurabile su una corsa da decine di migliaia, ma va detto
        invece che scoperto.

        Da li' viene anche il ripristino del contatore qui sotto. L'incremento
        di `iters` vive sulla CPU, e le operazioni host dentro una cattura
        girano davvero, subito: senza il ripristino il contatore conterebbe un
        aggiornamento che nessuno ha fatto, e resterebbe avanti di uno per
        tutta la corsa.

        L'autocast e' quello di `_passo_su_tensori` in entrambe le fasi: se la
        scaldata girasse con un contesto diverso da quello della cattura, i
        kernel registrati non sarebbero quelli scaldati.
        """
        # La guardia del confine, e va letta come tale invece che come una
        # sincronizzazione di cortesia. I giri ordinari appena passati hanno
        # mandato il batch sul device con `_in_pinned`, che scrive sempre nella
        # parita' 0 e -- a differenza di `_carica_batch` e di
        # `_carica_nei_statici` -- non registra nessun evento: da fuori non c'e'
        # modo di sapere se quelle DMA abbiano finito di leggere i buffer
        # pinnati. Qui sotto si riscrivono proprio quelli. Costa una volta per
        # corsa e chiude l'unico punto in cui le due discipline si toccano.
        torch.cuda.synchronize()

        self._tensori_statici = [
            torch.empty(x.shape, dtype=torch.float32, device=nn.device)
            for x in batch]
        self._carica_nei_statici(batch)

        flusso = torch.cuda.Stream()
        flusso.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(flusso):
            # Stesso numero dei giri ordinari che precedono la cattura, e non
            # per coincidenza: e' la stessa domanda -- quante iterazioni
            # servono perche' lo stato dell'ottimizzatore sia fermo agli stessi
            # indirizzi -- posta due volte, prima sul percorso normale e poi
            # sullo stream della cattura.
            for _ in range(self.ITERAZIONI_PRIMA_DELLA_CATTURA):
                self._passo_su_tensori(self._tensori_statici)
        torch.cuda.current_stream().wait_stream(flusso)
        torch.cuda.synchronize()

        iterazioni_prima = int(self.src_dst_opt.iters.item())

        try:
            self._grafo = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._grafo):
                self._loss_statiche = self._passo_su_tensori(self._tensori_statici)
        finally:
            # Nel `finally` e non dopo il `with`: se la cattura solleva a meta',
            # l'incremento host di `iters` puo' essere gia' avvenuto lo stesso, e
            # il contatore resterebbe avanti di uno anche nella corsa che il
            # grafo non ce l'ha.
            with torch.no_grad():
                self.src_dst_opt.iters.fill_(iterazioni_prima)

    @staticmethod
    def _triton_disponibile():
        """Il backend che torch.compile usa per generare i kernel su CUDA.
        Separato in un metodo perche' e' l'unica condizione della guardia che
        dipende da com'e' fatta la macchina, e va poterla fingere assente."""
        return importlib.util.find_spec("triton") is not None

    def _motivo_per_non_compilare(self):
        """
        Le condizioni sotto cui non si compila. None se si puo'.

        La guardia esiste perche' torch.compile non e' disponibile ovunque:
        su Windows il backend Triton per anni non c'e' stato affatto, e dove
        manca `torch.compile` non solleva alla chiamata -- solleva alla prima
        esecuzione, cioe' a corsa avviata, dentro il thread di addestramento.
        Un'opzione di prestazioni che impedisce l'avvio e' peggio dell'opzione
        che non c'e': qui si prova a compilare qualcosa di banale *prima* di
        toccare il passo vero, e se non funziona si prosegue senza.

        Il grafo e' escluso non perche' sia impossibile ma perche' la
        combinazione dei due non e' stata misurata: torch.compile puo' a sua
        volta catturare grafi, e sovrapporre le due cose senza numeri sarebbe
        una promessa non verificata.
        """
        if self.CUDA_GRAPH:
            return "cuda_graph e' acceso, e i due insieme non sono misurati"
        if not hasattr(torch, "compile"):
            return "questa versione di torch non ha torch.compile"
        if prerequisiti_windows.su_windows() and \
                not prerequisiti_windows.ripara_path_msvc():
            return ("su Windows serve il compilatore MSVC (cl.exe) e non "
                    "c'e' -- senza, inductor fallisce alla prima esecuzione. "
                    "Per installarlo: "
                    + prerequisiti_windows.ricetta_build_tools()
                    + " -- al riavvio successivo non servira' altro")
        if not self._triton_disponibile():
            return "il backend Triton non e' installato su questa macchina"
        try:
            prova = torch.compile(lambda t: t * 2.0 + 1.0, dynamic=False)
            campione = torch.ones(8, device=nn.device)
            if not torch.equal(prova(campione), campione * 2.0 + 1.0):
                return "una compilazione di prova ha dato un risultato diverso"
        except Exception as errore:
            return ("la compilazione di prova e' fallita -- "
                    f"{type(errore).__name__}: {errore}")
        return None

    @staticmethod
    def _zittisci_i_log_di_compilazione():
        """
        I graph break della prima iterazione compilata escono come centinaia di
        righe di avviso: rumore atteso, non errori. La soglia sale a ERROR, a
        meno che l'utente non abbia gia' chiesto un logging suo con TORCH_LOGS.
        """
        if "TORCH_LOGS" not in os.environ:
            torch._logging.set_logs(dynamo=logging.ERROR,
                                    inductor=logging.ERROR)

    def _offri_i_prerequisiti_se_servono(self):
        # L'offerta va fatta qui, nella fase interattiva, e a ogni avvio in
        # cui manca qualcosa: i worker spawn non devono mai porre domande, e
        # uno stdin chiuso fa ricadere ogni prompt sul suo default (No).
        #
        # Solo in addestramento, pero': on_initialize_options gira anche in
        # merge e in export (ModelBase.py:189, senza guardia), e li' il passo
        # compilato non esiste nemmeno -- proporre a un utente Windows di
        # installare Triton e i Build Tools per fondere un video sarebbe
        # chiedergli mezzo giga di compilatore per una leva che non entrera'
        # mai in funzione.
        if self.is_training and self.options['torch_compile'] \
                and prerequisiti_windows.su_windows():
            prerequisiti_windows.offri_installazione(io)

    def _prepara_la_compilazione(self):
        """
        Compila il passo, o rinuncia dicendo perche'.

        `dynamic=False` perche' le forme sono fisse per tutta la corsa: senza,
        torch.compile terrebbe la porta aperta a ricompilare per forme diverse,
        pagando una generalita' che qui non serve a nessuno. Il `mode` resta
        quello di default: le varianti piu' aggressive riscrivono di piu' e
        vanno misurate a parte, non date per buone.

        La compilazione vera avviene alla prima chiamata del passo, non qui, e
        costa minuti: chi misura deve scartare abbastanza iterazioni di
        scaldamento da lasciarla fuori dalla finestra.
        """
        self._zittisci_i_log_di_compilazione()
        motivo = self._motivo_per_non_compilare()
        if motivo is not None:
            self._rinuncia_alla_compilazione(motivo)
            return

        self._passo_compilato = torch.compile(saehd_train_step, dynamic=False)
        self._compilazione_collaudata = False
        io.log_info("torch_compile: passo compilato. La prima iterazione "
                    "paga la generazione dei kernel, circa mezzo minuto.")

    def _rinuncia_alla_compilazione(self, motivo):
        """Torna al passo normale, lasciando detto perche'.

        Gemello di `_rinuncia_al_grafo`, e come quello non tocca il
        precaricamento: chi ha chiesto un'ottimizzazione che non si e' potuta
        fare non va punito facendogli girare la corsa sul ramo lento.
        """
        self._passo_compilato = None
        self.TORCH_COMPILE = False
        io.log_info(f"torch_compile: {motivo}. "
                    f"L'addestramento prosegue senza compilazione.")

    def _esegui_il_passo(self, passo, tensori):
        """Il passo, quale che sia, nel contesto in cui va eseguito.

        L'autocast sta fuori dalla funzione compilata e non dentro: e' il
        contesto in cui il passo viene tracciato, e spostarlo dentro
        significherebbe compilare due volte lo stesso codice per il solo
        gusto di annidare un `with`.
        """
        with torch.autocast("cuda", dtype=self.DTYPE_AUTOCAST):
            return passo(self.nets, self.src_dst_opt,
                         self.src_dst_trainable_weights,
                         tensori, self._train_cfg, 1)

    def _passo_su_tensori(self, tensori):
        """Il solo passo: autocast e saehd_train_step, nessuna copia.

        **La prima esecuzione del passo compilato sta dentro un `try`**, e non
        per prudenza generica. `torch.compile` e' pigro: `_prepara_la_compilazione`
        non compila niente, restituisce un involucro, e la generazione vera dei
        kernel avviene qui, alla prima chiamata. E' quindi qui che si spende la
        memoria -- misurati +7.1 GiB di RSS, che su una macchina da 16 GB
        possono non esserci -- ed e' qui che un backend rotto o una versione
        incompatibile sollevano. Senza questo ripiego un `MemoryError` o un
        `BackendCompilerFailed` ucciderebbe il thread di addestramento a corsa
        avviata, buttando via le iterazioni non ancora salvate: esattamente cio'
        che l'`except` attorno alla cattura del grafo esiste per evitare.

        Il ripiego riesegue il passo dall'interprete. Se la chiamata compilata
        fosse fallita *dopo* aver aggiornato qualche peso, quel batch
        contribuirebbe due volte: un'iterazione anomala su decine di migliaia,
        contro una corsa persa. Il conto e' facile.

        Dal secondo giro in poi non resta nessun `try`: `_compilazione_collaudata`
        dice che quel passo ha gia' girato una volta su questa macchina, e il
        percorso caldo torna a essere una chiamata secca.
        """
        if self._passo_compilato is None:
            return self._esegui_il_passo(saehd_train_step, tensori)

        if self._compilazione_collaudata:
            return self._esegui_il_passo(self._passo_compilato, tensori)

        try:
            uscita = self._esegui_il_passo(self._passo_compilato, tensori)
        except Exception as errore:
            self._rinuncia_alla_compilazione(
                "la prima esecuzione del passo compilato e' fallita -- "
                f"{type(errore).__name__}: {errore}")
            return self._esegui_il_passo(saehd_train_step, tensori)

        self._compilazione_collaudata = True
        return uscita

    def _batch_appiattito(self):
        """Le due quadruple dei generatori nell'ordine posizionale in cui
        saehd_train_step le vuole -- lo stesso di sempre."""
        src, dst = self.generate_next_samples()
        return (*src, *dst)

    #override
    def generate_next_samples(self):
        """
        I generatori con l'opt-in acceso in on_initialize restituiscono, in
        coda al batch, la lista dei filename dei campioni
        (samplelib/SampleGeneratorFace.py). Li si stacca qui, prima che
        onTrainOneIter e il percorso ereditato di SAEHDModel li vedano: sia
        il ramo veloce sopra sia il fallback CPU via super() si aspettano
        esattamente le quadruple di prima, non una quintupla.
        """
        grezzi = super().generate_next_samples()
        puliti, nomi = [], []
        for gen, uscita in zip(self.generator_list, grezzi):
            if getattr(gen, "return_filenames", False) and len(uscita) > 0:
                nomi.append(uscita[-1])
                puliti.append(uscita[:-1])
            else:
                nomi.append(None)
                puliti.append(uscita)
        self._nomi_ultimo_batch = nomi if any(n is not None for n in nomi) else None
        self.last_sample = puliti
        return puliti

    def get_preview_filenames(self):
        """
        I nomi dei campioni mostrati nell'ultima anteprima generata, gia'
        ridotti al basename e troncati a quanti ne mostra onGetPreview.
        None se l'opt-in non era attivo su almeno uno dei due lati.
        """
        nomi = getattr(self, "_nomi_ultimo_batch", None)
        if nomi is None or any(n is None for n in nomi):
            return None
        # Stessa formula di onGetPreview in Model_SAEHD (il numero di tile
        # mostrate): se la' cambia il conteggio, va aggiornata anche qui.
        mostrati = min(4, self.get_batch_size(), 800 // self.resolution)
        return tuple([os.path.basename(x) for x in lato[:mostrati]]
                     for lato in nomi)

    #override
    def onTrainOneIter(self):
        """
        Le loss restano sul device fino a INTERVALLO_SCARICO_LOSS: il .cpu()
        di SAEHDModel.src_dst_train (Model_SAEHD/Model.py:1008) e' la barriera
        che impedisce alla CPU di correre avanti mentre la GPU calcola --
        saehd_train_step qui sotto ritorna tensori, non numpy, e il .item()
        che li scarica sull'host si paga una iterazione su
        INTERVALLO_SCARICO_LOSS, non tutte.

        Il ritorno ha la stessa forma di SAEHDModel.onTrainOneIter --
        ModelBase.train_one_iter fa float(loss[1]) su ogni elemento -- ma
        senza D_train/D_src_dst_train: saehdx-liae-256, la configurazione che
        questo cablaggio misura, li tiene spenti (true_face_power e gan_power
        a zero), e il percorso veloce serve solo src_dst_train_su_device.

        src_dst_train_su_device esiste solo se on_initialize ha preso il ramo
        CUDA (stesso guardiano delle altre leve di questa classe): senza GPU
        si ricade sul passo intero di SAEHDModel, .cpu() compreso -- su CPU
        non c'e' niente da sovrapporre, e ModelBase.train_one_iter lo dice
        gia' nel proprio commento.
        """
        if not hasattr(self, "src_dst_train_su_device"):
            return super().onTrainOneIter()

        # «Questo passo e' catturabile?» si chiede al primo giro, prima di
        # toccare qualunque cosa, e non al terzo insieme alla cattura:
        # `_motivo_per_non_catturare` legge solo attributi che on_initialize ha
        # gia' fissato, quindi la risposta e' la stessa a ogni giro -- e
        # scoprire il no *dopo* aver spento il precaricamento significherebbe
        # lasciarlo spento per tutta la corsa, cioe' far pagare la leva a chi
        # non ha potuto averla.
        #
        # Il precaricamento si spegne solo se la cattura si fara' davvero: i
        # due non convivono. Con profondita' 1 e buffer statici unici il
        # caricamento del batch N+1 riscriverebbe gli stessi indirizzi che il
        # replay di N sta leggendo. La copia host->device che si perde vale
        # 2.0 ms su ~140 (il conto e' in `_in_pinned`); i buffer fissi sono
        # invece la condizione di esistenza del grafo, non un dettaglio.
        if self.CUDA_GRAPH and self._grafo is None:
            motivo = self._motivo_per_non_catturare()
            if motivo is not None:
                self._rinuncia_al_grafo(f"passo non catturato -- {motivo}")
            elif self.PREFETCH:
                self._prefetch_sospeso = self.PREFETCH
                self.PREFETCH = False

        # La compilazione si prepara al primo giro e non in on_initialize, per
        # la stessa ragione per cui il grafo nasce qui: chi misura accende la
        # leva sull'istanza dopo la costruzione, e una preparazione fatta prima
        # non la vedrebbe mai. Costa un `if` su un attributo per iterazione.
        #
        # **Dopo** il blocco qui sopra e non prima, e l'ordine e' l'unica cosa
        # che separa «l'utente ha una delle due leve» da «non ne ha nessuna».
        # Chi accende grafo e compilazione insieme su una configurazione non
        # catturabile vedeva, con l'ordine inverso, rifiutare prima la
        # compilazione (perche' cuda_graph era acceso) e subito dopo il grafo
        # (perche' il passo non e' catturabile), restando a mani vuote. Cosi'
        # invece il grafo si ritira per primo, spegne CUDA_GRAPH, e la
        # compilazione trova il campo libero nello stesso giro.
        #
        # Resta scoperto un caso, ed e' bene saperlo: se la cattura fallisce al
        # terzo giro -- cosa che si scopre solo provandola -- la compilazione e'
        # gia' stata rifiutata e non riparte. Chi ci finisce ha comunque i due
        # messaggi che glielo dicono.
        if self.TORCH_COMPILE and self._passo_compilato is None:
            self._prepara_la_compilazione()

        if self.CUDA_GRAPH and self._grafo is None \
                and self._giri_fatti >= self.ITERAZIONI_PRIMA_DELLA_CATTURA:
            try:
                self._cattura_il_passo(self._batch_appiattito())
            except Exception as errore:
                # L'elenco di `_motivo_per_non_catturare` copre cio' che si sa
                # in anticipo; una cattura puo' fallire lo stesso per una
                # versione diversa di torch o di cuDNN, o per una combinazione
                # di opzioni che nessuno ha misurato. Il prompt promette che in
                # quel caso l'addestramento prosegue: senza questo `except` la
                # promessa sarebbe falsa e l'eccezione ucciderebbe il thread di
                # addestramento a meta' corsa.
                self._rinuncia_al_grafo(
                    f"cattura fallita -- {type(errore).__name__}: {errore}")

        self._giri_fatti += 1

        if self._grafo is not None:
            self._carica_nei_statici(self._batch_appiattito())
            self._grafo.replay()
            # Il contatore dell'ottimizzatore vive sulla CPU (AdaBelief lo
            # alloca prima che le reti vadano sul device e nessuno lo sposta),
            # quindi il suo incremento e' finito dentro la cattura come
            # operazione host: eseguito una volta sola, mai piu' al replay.
            # Va rifatto qui o il contatore salvato in src_dst_opt.npy
            # resterebbe fermo al giorno della cattura.
            with torch.no_grad():
                self.src_dst_opt.iters += 1
            src, dst = self._loss_statiche
        elif self.PREFETCH:
            if self._prossimo is None:
                self._prossimo = self._carica_batch(self._batch_appiattito())
            tensori, evento = self._prossimo

            corrente = torch.cuda.current_stream()
            corrente.wait_event(evento)
            # wait_event ordina i kernel; record_stream informa l'allocatore.
            # Questi tensori sono nati sullo stream secondario, e senza
            # dichiararli in uso su quello di calcolo l'allocatore potrebbe
            # riassegnarne i blocchi al caricamento successivo -- che gira
            # proprio sullo stream secondario -- mentre il passo li legge
            # ancora.
            for t in tensori:
                t.record_stream(corrente)

            src, dst = self._passo_su_tensori(tensori)

            # Il passo e' gia' in coda: il prelievo dai generatori e le copie
            # del batch successivo si sovrappongono alla sua esecuzione. E' qui
            # che nasce il guadagno, e qui che nasce l'unico effetto visibile
            # altrove: da questo punto last_sample e i nomi dei file sono
            # quelli del batch *precaricato*, un giro avanti rispetto al passo
            # appena eseguito. E' accettabile e voluto -- l'anteprima resta
            # coerente con se stessa, immagini e nomi vengono dallo stesso
            # batch, semplicemente non da quello di cui si stampa la loss.
            self._prossimo = self._carica_batch(self._batch_appiattito())
        else:
            src, dst = self.src_dst_train_su_device(*self._batch_appiattito())

        if self.get_iter() % self.INTERVALLO_SCARICO_LOSS == 0:
            self._ultime_loss = (float(src.mean().item()),
                                 float(dst.mean().item()))

        s, d = getattr(self, "_ultime_loss", (0.0, 0.0))
        return (('src_loss', s), ('dst_loss', d))


Model = SAEHDXModel
