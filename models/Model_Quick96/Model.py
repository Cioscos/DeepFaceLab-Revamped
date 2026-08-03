import multiprocessing
from functools import partial

import numpy as np
import torch

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *


def quick96_flow(nets, warped_src, warped_dst, need_src_dst=True):
    """
    Il forward delle quattro reti (Model.py:111-115 nella versione TF).

    Sei predizioni, nell'ordine in cui il dumper le scrive. Non riusa
    saehd_flow: quella calcola anche pred_src_dst_no_code_grad, che serve ai
    due style power di SAEHD e che Quick96 non ha, e importarla accoppierebbe
    due modelli che ModelBase tiene indipendenti.

    La topologia e' quella `df`: pred_src_dst esce dal *decoder_src* applicato
    al codice *dst* (Model.py:115). Le due reti decoder sono strutturalmente
    identiche, quindi scambiarle non solleva niente e produce un'altra faccia.

    `need_src_dst=False` salta quella terza applicazione del decoder_src e
    ritorna None nelle ultime due posizioni. Serve al solo
    quick96_train_step: la loss non nomina pred_src_dst ne' pred_src_dstm
    (Model.py:140-146 nella versione TF), e nel grafo TF quel ramo non veniva
    valutato affatto durante src_dst_train: pred_src_dst e pred_src_dstm
    esistevano nel grafo (Model.py:159, :162 nella versione TF) ma li fetchava
    solo AE_view (:186), e nn.tf_sess.run di src_dst_train (:172) chiedeva solo
    src_loss, dst_loss e l'update, quindi la potatura del grafo escludeva quel
    decoder.
    In torch eager il calcolo e' incondizionato, quindi senza questo parametro
    il passo di training pagherebbe un decoder intero che il TF non pagava --
    e ne terrebbe vivo il grafo autograd fino alla fine del passo. Il default
    resta True perche' quick96_view e quick96_merge quelle due uscite le usano.
    """
    src_code = nets['inter'](nets['encoder'](warped_src))
    dst_code = nets['inter'](nets['encoder'](warped_dst))

    pred_src_src, pred_src_srcm = nets['decoder_src'](src_code)
    pred_dst_dst, pred_dst_dstm = nets['decoder_dst'](dst_code)
    pred_src_dst, pred_src_dstm = nets['decoder_src'](dst_code) if need_src_dst else (None, None)

    return (pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm,
            pred_src_dst, pred_src_dstm)


def quick96_loss(target_src, target_srcm, pred_src_src, pred_src_srcm,
                 target_dst, target_dstm, pred_dst_dst, pred_dst_dstm,
                 resolution, masked_training):
    """
    Le due loss per campione (Model.py:125-146 nella versione TF).

    Ritorna vettori di lunghezza batch, non scalari: e' cosi' che
    onTrainOneIter li media (Model.py:180-181).

    A 96 si prende sempre il ramo `resolution < 256` della dssim -- un solo
    termine, filter_size int(96/11.6) == 8. Il ramo a due termini di SAEHD qui
    non esiste: la risoluzione e' cablata.

    Il terzo addendo di ciascuna loss usa la maschera NON sfocata
    (`target_srcm`, non `target_srcm_blur`): e' la trascrizione di Model.py:142
    e :146, non una svista.

    Non trascritte, di proposito: le tre locali morte di Model.py:129, :137 e
    :138 (gpu_target_dst_anti_masked, gpu_psd_target_dst_masked,
    gpu_psd_target_dst_anti_masked). Nel grafo TF erano nodi mai valutati; in
    torch eager costerebbero tre moltiplicazioni a piena risoluzione per
    iterazione.

    Non trascritta ma **viva**, e per questo inlinata invece che saltata:
    gpu_target_dst_masked (Model.py:128), che :132 legge per costruire
    gpu_target_dst_masked_opt. Qui il suo valore compare direttamente dentro
    target_dst_masked_opt (`target_dst*target_dstm_blur`). E' l'unica riga viva
    del contratto "cosa e' e cosa non e' trascritto": le altre quattro locali
    che spariscono sono morte, questa no.
    """
    target_srcm_blur = nn.gaussian_blur(target_srcm, max(1, resolution // 32))
    target_dstm_blur = nn.gaussian_blur(target_dstm, max(1, resolution // 32))

    target_src_masked_opt = target_src*target_srcm_blur if masked_training else target_src
    target_dst_masked_opt = target_dst*target_dstm_blur if masked_training else target_dst

    pred_src_src_masked_opt = pred_src_src*target_srcm_blur if masked_training else pred_src_src
    pred_dst_dst_masked_opt = pred_dst_dst*target_dstm_blur if masked_training else pred_dst_dst

    filter_size = int(resolution/11.6)

    src_loss  = torch.mean( 10*nn.dssim(target_src_masked_opt, pred_src_src_masked_opt, max_val=1.0, filter_size=filter_size), dim=[1])
    src_loss += torch.mean( 10*torch.square(target_src_masked_opt - pred_src_src_masked_opt), dim=[1,2,3])
    src_loss += torch.mean( 10*torch.square(target_srcm - pred_src_srcm), dim=[1,2,3])

    dst_loss  = torch.mean( 10*nn.dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, max_val=1.0, filter_size=filter_size), dim=[1])
    dst_loss += torch.mean( 10*torch.square(target_dst_masked_opt - pred_dst_dst_masked_opt), dim=[1,2,3])
    dst_loss += torch.mean( 10*torch.square(target_dstm - pred_dst_dstm), dim=[1,2,3])

    return src_loss, dst_loss


def quick96_train_step(nets, opt, trainable_weights, batch, cfg, gpu_count):
    """
    Il passo di onTrainOneIter (Model.py:151-182 nella versione TF): forward,
    loss, gradienti, update.

    `G_loss.sum() / gpu_count` e' il gradiente del TF, non una riformulazione:
    tf.gradients di una loss vettoriale ne somma le componenti (Model.py:152) e
    nn.average_gv_list divideva poi per il numero di GPU (Model.py:166).
    torch.autograd.grad vuole uno scalare, e questo e' lo scalare che da' lo
    stesso gradiente. Stessa identita' che vale per SAEHD.

    `need_src_dst=False` perche' la loss non nomina pred_src_dst ne'
    pred_src_dstm (Model.py:140-146 nella versione TF), ed e' l'unico punto
    in cui la porta rischiava di fare piu' lavoro del TF sul percorso caldo:
    nel grafo TF quel ramo del decoder non veniva valutato durante
    src_dst_train (nn.tf_sess.run a Model.py:172 chiedeva solo src_loss,
    dst_loss e l'update), mentre in torch eager si paga a ogni iterazione -- una terza
    applicazione del decoder_src, piu' il grafo autograd che le sue due uscite
    tengono vivo fino alla fine del passo. Numericamente e' un no-op: nessuna
    delle due uscite entra in G_loss, e i valori prodotti non si muovono di un
    ULP.
    """
    warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm = batch

    (pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm,
     _, _) = quick96_flow(nets, warped_src, warped_dst, need_src_dst=False)

    src_loss, dst_loss = quick96_loss(target_src, target_srcm, pred_src_src, pred_src_srcm,
                                      target_dst, target_dstm, pred_dst_dst, pred_dst_dstm,
                                      cfg['resolution'], cfg['masked_training'])

    G_loss = src_loss + dst_loss

    opt.step( nn.gradients(G_loss.sum() / gpu_count, trainable_weights) )
    return src_loss, dst_loss


def quick96_view(nets, warped_src, warped_dst):
    """
    Le cinque predizioni della preview, nel loro ordine (Model.py:186 nella
    versione TF): pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst,
    pred_src_dstm.

    L'ordine e' un contratto con onGetPreview, che le spacchetta posizionali
    (Model.py:285 nella versione TF). `torch.no_grad()` e' quello che nel TF
    era gratis: nn.tf_sess.run non costruiva nessun grafo all'indietro.

    Passa da quick96_flow e non da una seconda trascrizione: e' lo stesso
    cablaggio che il training percorre. Costa in piu' il ramo maschera di
    pred_src_srcm, che AE_view non chiedeva -- ma la preview non e' per
    iterazione (una all'avvio, una a ogni salvataggio).
    """
    with torch.no_grad():
        (pred_src_src, _, pred_dst_dst, pred_dst_dstm,
         pred_src_dst, pred_src_dstm) = quick96_flow(nets, warped_src, warped_dst)

    return [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm]


def quick96_merge(nets, warped_dst):
    """
    Il grafo del merger: tre tensori (Model.py:194-200 nella versione TF), nel
    solo ordine che predictor_func accetta -- bgr, mask_dst_dstm, mask_src_dstm,
    cioe' pred_src_dst, pred_dst_dstm, pred_src_dstm.

    Prende il solo warped_dst: il merger ha in mano un frame dst e nient'altro.
    E' anche il motivo per cui questa non e' una selezione da quick96_flow --
    passare due volte lo stesso frame darebbe gli stessi tre tensori pagando
    encoder, inter e decoder anche sul ramo src, e qui si paga per *fotogramma*.
    Che i due cablaggi non divergano lo fissa test_merge_is_the_same_wiring_as_the_flow.
    """
    with torch.no_grad():
        dst_code = nets['inter'](nets['encoder'](warped_dst))
        pred_src_dst, pred_src_dstm = nets['decoder_src'](dst_code)
        _, pred_dst_dstm            = nets['decoder_dst'](dst_code)

    return [pred_src_dst, pred_dst_dstm, pred_src_dstm]


class QModel(ModelBase):
    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        # NCHW sempre: nn.set_data_format solleva su qualunque altro valore.
        # Il TF ripiegava su NHWC in debug e senza GPU; quel ramo si perde
        # consapevolmente, perche' core/leras e' NCHW.
        # Stessa decisione di models/Model_SAEHD/Model.py:726.
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)

        resolution = self.resolution = 96
        self.face_type = FaceType.FULL
        ae_dims = 128
        e_dims = 64
        d_dims = 64
        d_mask_dims = 16
        self.pretrain = False
        self.pretrain_just_disabled = False

        masked_training = True

        models_opt_on_gpu = len(devices) >= 1 and all([dev.total_mem_gb >= 4 for dev in devices])
        # Scelta di porting, stessa applicata in models/Model_SAEHD/Model.py:807-809:
        # in TF le variabili (models_opt_device) e le op del merge
        # (nn.tf_default_device_name) vivevano su due scope tf.device
        # indipendenti; in torch un modulo vive su un solo device, quindi la
        # scelta e' una sola e si segue il device delle op.
        # Fuori dal training `not self.is_training` corto-circuita e
        # models_opt_on_gpu non entra: si prende sempre nn.device. La
        # conseguenza da sapere e' la stessa di SAEHD -- il merge tiene i pesi
        # in VRAM invece che in RAM -- ma qui non c'e' nessuna leva da cercare,
        # perche' Quick96 non ha opzioni (Model.py:22-31 nella versione TF):
        # l'unica via e' `--cpu-only` (mainscripts/Merger.py:53), che era
        # l'unica anche col TF.
        models_opt_device = nn.device if models_opt_on_gpu or not self.is_training else torch.device('cpu')
        optimizer_vars_on_cpu = models_opt_device.type == 'cpu'

        input_ch = 3

        self.model_filename_list = []

        model_archi = nn.DeepFakeArchi(resolution, opts='ud')

        # Niente placeholder: i tensori si creano dai numpy dentro le funzioni
        # di training (src_dst_train qui sotto).

        # Initializing model classes
        self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
        encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

        self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
        inter_out_ch = self.inter.get_out_ch()

        self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
        self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

        # I nomi sono quelli che quick96_flow indicizza, e sono anche i
        # prefissi delle chiavi dei pesi su disco: due Decoder identici hanno lo
        # stesso disk_key e senza il nome della rete davanti collidono.
        # Stesso schema di models/Model_SAEHD/Model.py:833-834.
        self.nets = {'encoder': self.encoder, 'inter': self.inter,
                     'decoder_src': self.decoder_src, 'decoder_dst': self.decoder_dst}

        self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                      [self.inter,       'inter.npy'      ],
                                      [self.decoder_src, 'decoder_src.npy'],
                                      [self.decoder_dst, 'decoder_dst.npy']  ]

        # build() e .to() prima di leggerne i pesi, come
        # models/Model_SAEHD/Model.py:879-881: get_weights() li enumera e
        # initialize_variables alloca gli accumulatori sul device del
        # parametro quando vars_on_cpu e' False. Qui dentro ci sono solo le
        # reti: l'ottimizzatore entra nella lista piu' sotto, dopo questo ciclo.
        for model, _ in self.model_filename_list:
            model.build()
            model.to(models_opt_device)

        def to_t(x):
            return torch.as_tensor(np.ascontiguousarray(x)).to(models_opt_device, nn.floatx)

        if self.is_training:
            self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()

            # optimizer_weights(), non get_weights(): initialize_variables
            # vuole le quadruple (name, param, owner, param_path) che
            # accoppiano ogni peso al suo accumulatore su disco
            # (core/leras/layers/Saveable.py), non i Parameter nudi che
            # nn.gradients/opt.step usano. Stesso schema di
            # models/Model_SAEHD/Model.py:898-902 (li' dietro
            # saehd_src_dst_weights; qui inline perche' Quick96 ha una sola
            # topologia, niente rami df/liae da smistare).
            src_dst_saveable_weights = self.encoder.optimizer_weights() + self.inter.optimizer_weights() + self.decoder_src.optimizer_weights() + self.decoder_dst.optimizer_weights()

            # Initialize optimizers
            self.src_dst_opt = nn.RMSprop(lr=2e-4, lr_dropout=0.3, name='src_dst_opt')
            self.src_dst_opt.initialize_variables(src_dst_saveable_weights, vars_on_cpu=optimizer_vars_on_cpu )
            self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, 4 // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            train_cfg = {'resolution': resolution, 'masked_training': masked_training}

            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm, \
                              warped_dst, target_dst, target_dstm):
                src_loss, dst_loss = quick96_train_step(
                    self.nets, self.src_dst_opt, self.src_dst_trainable_weights,
                    [to_t(x) for x in (warped_src, target_src, target_srcm,
                                       warped_dst, target_dst, target_dstm)],
                    train_cfg, gpu_count)
                # Model.py:180-181 nella versione TF: s = np.mean(s); d = np.mean(d)
                # dentro src_dst_train stesso -- onTrainOneIter (sotto) non
                # rimedia, a differenza di Model_SAEHD dove il mean sta la'.
                return np.mean(src_loss.detach().cpu().numpy()), np.mean(dst_loss.detach().cpu().numpy())
            self.src_dst_train = src_dst_train

            def AE_view(warped_src, warped_dst):
                return [x.detach().cpu().numpy() for x in
                        quick96_view(self.nets, to_t(warped_src), to_t(warped_dst))]
            self.AE_view = AE_view
        else:
            # Initializing merge function
            def AE_merge(warped_dst):
                return [x.detach().cpu().numpy() for x in
                        quick96_merge(self.nets, to_t(warped_dst))]
            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if model == self.inter:
                    do_init = True
            else:
                do_init = self.is_first_run()

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init and self.pretrained_model_path is not None:
                pretrained_filepath = self.pretrained_model_path / filename
                if pretrained_filepath.exists():
                    do_init = not model.load_weights(pretrained_filepath)

            if do_init:
                model.init_weights()

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True,  'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution}
                                              ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True,  'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution}
                                               ],
                        generators_count=dst_generators_count )
                             ])

            self.last_samples = None

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def onTrainOneIter(self):

        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = self.last_samples
            warped_src = target_src
            warped_dst = target_dst
        else:
            samples = self.last_samples = self.generate_next_samples()
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = samples

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm,
                                                 warped_dst, target_dst, target_dstm)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, samples, for_history=False):
        ( (warped_src, target_src, target_srcm),
          (warped_dst, target_dst, target_dstm) ) = samples

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(4, self.get_batch_size() )
        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96', np.concatenate (st, axis=0 )), ]

        st_m = []
        for i in range(n_samples):
            ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
            st_m.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96 masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]
        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=self.face_type,
                                     default_mode = 'overlay',
                                    )

Model = QModel
