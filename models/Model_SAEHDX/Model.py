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
import numpy as np
import torch

from core.interact import interact as io
from core.leras import nn
from models.Model_SAEHD.Model import SAEHDModel, saehd_train_step


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
        di Quick96, che devono restare numericamente invariati.
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

    #override
    def on_initialize(self):
        super().on_initialize()

        if nn.device is not None and nn.device.type == "cuda":
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

            self._staging = {}

            def _in_pinned(x, indice):
                """
                Copia x in un buffer pinnato riusato e lo manda sul device
                senza bloccare. Il buffer e' per posizione nel batch: le forme
                sono fisse per tutta la corsa, quindi si alloca una volta.

                Senza memoria pinnata `non_blocking=True` e' una bugia: torch
                lo accetta e la copia resta sincrona, perche' il driver non
                puo' fare DMA da pagine che il kernel puo' spostare.
                """
                buf = self._staging.get(indice)
                if buf is None or buf.shape != x.shape:
                    buf = torch.empty(x.shape, dtype=torch.float32,
                                      pin_memory=True)
                    self._staging[indice] = buf
                buf.copy_(torch.from_numpy(np.ascontiguousarray(x)))
                return buf.to(nn.device, non_blocking=True)

            def src_dst_train_su_device(*batch):
                with torch.autocast("cuda", dtype=self.DTYPE_AUTOCAST):
                    return saehd_train_step(
                        self.nets, self.src_dst_opt,
                        self.src_dst_trainable_weights,
                        [_in_pinned(x, i) for i, x in enumerate(batch)],
                        self._train_cfg, 1)

            self.src_dst_train_su_device = src_dst_train_su_device

            passo_originale = self.src_dst_train

            def src_dst_train_veloce(*batch):
                with torch.autocast("cuda", dtype=self.DTYPE_AUTOCAST):
                    return passo_originale(*batch)

            self.src_dst_train = src_dst_train_veloce

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

        ((warped_src, target_src, target_srcm, target_srcm_em),
         (warped_dst, target_dst, target_dstm, target_dstm_em)) = \
            self.generate_next_samples()

        src, dst = self.src_dst_train_su_device(
            warped_src, target_src, target_srcm, target_srcm_em,
            warped_dst, target_dst, target_dstm, target_dstm_em)

        if self.get_iter() % self.INTERVALLO_SCARICO_LOSS == 0:
            self._ultime_loss = (float(src.mean().item()),
                                 float(dst.mean().item()))

        s, d = getattr(self, "_ultime_loss", (0.0, 0.0))
        return (('src_loss', s), ('dst_loss', d))


Model = SAEHDXModel
