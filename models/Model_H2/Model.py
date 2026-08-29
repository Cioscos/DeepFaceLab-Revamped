"""H2, «decoder a identita' modulata»: un encoder e un inter come SAEHD
liae-udt, un solo decoder (DecoderH2) che legge un vettore d'identita' nei
due stadi bassi, due vettori (Identita) appresi o presi da AdaFace. Stessa
macchina veloce di SAEHDX, stessi supervisori di H1 (mixin), export .dfm con
morph_value; default: la ricetta h2-ib. Il disegno e i criteri stanno nella
documentazione del ciclo H2 sotto docs/."""
import multiprocessing
import pathlib

import numpy as np
import onnx
import torch

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models.Model_H1.supervisione import Supervisori
from models.Model_H2 import innesto
from models.Model_H2.passo import H2ExportModule, h2_merge, h2_src_dst_weights, h2_train_step, h2_view
from models.Model_SAEHDX.Model import SAEHDXModel
from samplelib import SampleGeneratorFace, SampleProcessor


class H2Model(Supervisori, SAEHDXModel):
    archi_type = 'h2'       # _accelera lo scrive in _train_cfg; il banco riconosce H2 dalla classe, non da qui
    E_DIM = innesto.E_DIM   # stesso valore di models.Model_H2.innesto: un solo posto lo definisce

    #override
    def _passo(self):
        return h2_train_step

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()
        lowest_vram = device_config.devices.get_worst_device().total_mem_gb if len(device_config.devices) else 2
        suggest_batch_size = 8 if lowest_vram >= 4 else 4
        min_res, max_res = 64, 640

        default_resolution        = self.options['resolution']        = self.load_or_def_option('resolution', 224)
        default_face_type         = self.options['face_type']         = self.load_or_def_option('face_type', 'wf')
        default_models_opt_on_gpu = self.options['models_opt_on_gpu'] = self.load_or_def_option('models_opt_on_gpu', True)
        default_ae_dims           = self.options['ae_dims']           = self.load_or_def_option('ae_dims', 512)
        default_e_dims            = self.options['e_dims']            = self.load_or_def_option('e_dims', 64)
        default_d_dims            = self.options['d_dims']            = self.load_or_def_option('d_dims', 64)
        default_d_mask_dims       = self.options['d_mask_dims']       = self.load_or_def_option('d_mask_dims', 22)
        default_identita          = self.options['identita']          = self.load_or_def_option('identita', 'learned')
        default_masked_training   = self.options['masked_training']   = self.load_or_def_option('masked_training', True)
        default_eyes_mouth_prio   = self.options['eyes_mouth_prio']   = self.load_or_def_option('eyes_mouth_prio', False)
        default_uniform_yaw       = self.options['uniform_yaw']       = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask     = self.options['blur_out_mask']     = self.load_or_def_option('blur_out_mask', False)
        default_adabelief         = self.options['adabelief']         = self.load_or_def_option('adabelief', True)
        default_lr_dropout        = self.options['lr_dropout']        = self.load_or_def_option('lr_dropout', 'n')
        default_random_warp       = self.options['random_warp']       = self.load_or_def_option('random_warp', True)
        default_random_hsv_power  = self.options['random_hsv_power']  = self.load_or_def_option('random_hsv_power', 0.0)
        default_ct_mode           = self.options['ct_mode']           = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad          = self.options['clipgrad']          = self.load_or_def_option('clipgrad', False)
        default_id_power    = self.options['id_power']    = self.load_or_def_option('id_power', 2.0)
        default_ifsr_power  = self.options['ifsr_power']  = self.load_or_def_option('ifsr_power', 0.08)
        default_dino_power  = self.options['dino_power']  = self.load_or_def_option('dino_power', 0.0)
        default_dino_ogni   = self.options['dino_ogni']   = self.load_or_def_option('dino_ogni', 4)
        default_ffl_power   = self.options['ffl_power']   = self.load_or_def_option('ffl_power', 0.0)
        default_bleed_power = self.options['bleed_power'] = self.load_or_def_option('bleed_power', 1.0)
        default_bleed_campione = self.options['bleed_campione'] = self.load_or_def_option('bleed_campione', False)
        default_maschera_tronco   = self.options['maschera_tronco']   = self.load_or_def_option('maschera_tronco', False)
        default_innesto           = self.options['innesto']           = self.load_or_def_option('innesto', '')
        default_innesto_inter     = self.options['innesto_inter']     = self.load_or_def_option('innesto_inter', False)

        # Letti dall'eredita' di SAEHDX (_train_cfg, onTrainOneIter di SAEHD) e mai offerti: H2 non li ha.
        self.options['true_face_power'] = 0.0
        self.options['face_style_power'] = 0.0
        self.options['bg_style_power'] = 0.0
        self.options['gan_power'] = 0.0
        self.options['pretrain'] = False

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(suggest_batch_size)

        if self.is_first_run():
            sorgenti = innesto.sorgenti(self.saved_models_path)
            proposto = default_innesto if default_innesto in sorgenti else ""
            # Stringa libera e verifica propria, NON valid_list: con il file delle
            # risposte della GUI io.input_str ricadrebbe in silenzio sul default,
            # e un refuso nel nome farebbe partire da zero una corsa di ore.
            scelto = io.input_str("Graft from model", proposto, help_message="An H2 model is meant to be grafted from a trained liae-udt SAEHD/SAEHDX model in this model dir (a pretrained RTM works): the encoder and decoder weights are copied, the identity vectors and the optimizer start from zero. Empty = train from scratch (never measured). Available: " + (", ".join(sorted(sorgenti)) or "none")).strip()
            if scelto and scelto not in sorgenti:
                raise Exception(f"H2: «{scelto}» is not a liae-udt SAEHD/SAEHDX model in {self.saved_models_path}; "
                                f"available: {', '.join(sorted(sorgenti)) or 'none'}")
            self.options['innesto'] = scelto
            if self.options['innesto']:
                classe, o = sorgenti[self.options['innesto']]
                for k in innesto.OPZIONI_FISSATE_DAI_PESI:
                    self.options[k] = o[k]
                io.log_info(f"H2: innesto da «{self.options['innesto']}» ({classe}); fissati dai pesi: "
                            + ", ".join(f"{k}={self.options[k]}" for k in innesto.OPZIONI_FISSATE_DAI_PESI))
                self.options['innesto_inter'] = io.input_bool("Copy the inter from the source", default_innesto_inter, help_message="Warm inter: faster reconstruction at the start. Without supervisors it stops the identity vectors from separating (measured); with the supervisors on it is safe.")
            else:
                self.options['innesto_inter'] = False
                if sorgenti:
                    io.log_info("H2: nessun innesto, si parte da zero (mai misurato). Sorgenti liae-udt in questa cartella: "
                                + ", ".join(sorted(sorgenti)))
                resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="H2 always uses the -t encoder and the -d output: the value is rounded to a multiple of 32.")
                self.options['resolution'] = int(np.clip((resolution // 32) * 32, min_res, max_res))
                self.options['face_type'] = io.input_str("Face type", default_face_type, ['h', 'mf', 'f', 'wf', 'head'], help_message="Half / mid face / full face / whole face / head.").lower()
                self.options['ae_dims'] = np.clip(io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="Size of the code. The decoder receives twice this many channels, as liae does."), 32, 1024)
                e_dims = np.clip(io.input_int("Encoder dimensions", default_e_dims, add_info="16-256"), 16, 256)
                self.options['e_dims'] = e_dims + e_dims % 2
                d_dims = np.clip(io.input_int("Decoder dimensions", default_d_dims, add_info="16-256"), 16, 256)
                self.options['d_dims'] = d_dims + d_dims % 2
                d_mask_dims = np.clip(io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256"), 16, 256)
                self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2
            self.options['maschera_tronco'] = io.input_bool("Mask reads the identity-modulated trunk", default_maschera_tronco, help_message="Adds a zero-initialized 1x1 bridge from the identity-modulated trunk into the mask branch, so the mask can depend on the identity vector. Off: the mask depends on the code only, as in SAEHD. Fixed at the first start.")
            self.options['identita'] = io.input_str("Identity vectors", default_identita, ['learned', 'adaface'], help_message="learned: two free vectors trained with the model. adaface: the mean AdaFace embedding of each faceset, computed once at the first start and frozen.")

        if self.is_first_run() or ask_override:
            # Pesi della loss: si cambiano fra una sessione e l'altra (il 2026-08-29 DINO
            # acceso al riavvio restava a zero perche' stavano nel blocco del primo avvio).
            self.options['id_power']   = np.clip(io.input_number("Identity power", default_id_power, add_info="0.0 .. 10.0", help_message="Cosine loss between the AdaFace embedding of the swapped face and the mean embedding of the src faceset. 0 disables it."), 0.0, 10.0)
            self.options['ifsr_power'] = np.clip(io.input_number("IFSR power", default_ifsr_power, add_info="0.0 .. 10.0", help_message="L1 between AdaFace intermediate features of the swapped face and of the dst face. 0 disables it."), 0.0, 10.0)
            self.options['dino_power'] = np.clip(io.input_number("DINOv2 perceptual power", default_dino_power, add_info="0.0 .. 10.0", help_message="L1 between DINOv2-S tokens of the masked reconstructions and their targets. 0 disables it."), 0.0, 10.0)
            self.options['dino_ogni'] = int(np.clip(io.input_int("DINOv2 stride", default_dino_ogni, add_info="1 .. 100", help_message="Applies the DINOv2 term every N iterations instead of every one, scaled by N so the average gradient is unchanged (lazy regularization). 1 disables the stride."), 1, 100))
            self.options['ffl_power']  = np.clip(io.input_number("Focal frequency power", default_ffl_power, add_info="0.0 .. 10.0", help_message="Focal Frequency Loss on the masked reconstructions. 0 disables it."), 0.0, 10.0)
            self.options['bleed_power'] = np.clip(io.input_number("Bleed power", default_bleed_power, add_info="0.0 .. 10.0", help_message="Penalizes the swap when its AdaFace embedding drifts toward the mean embedding of the dst faceset beyond a fixed cosine margin. 0 disables it."), 0.0, 10.0)
            self.options['bleed_campione'] = io.input_bool("Bleed per sample", default_bleed_campione, help_message="Bleed repels the swap from its own sample's dst embedding instead of the dst faceset mean. Only effective when bleed power > 0.")
            if self.options['face_type'] in ('wf', 'head'):
                self.options['masked_training'] = io.input_bool("Masked training", default_masked_training, help_message="Clips the training area to the face mask.")
            self.options['eyes_mouth_prio'] = io.input_bool("Eyes and mouth priority", default_eyes_mouth_prio)
            self.options['uniform_yaw'] = io.input_bool("Uniform yaw distribution of samples", default_uniform_yaw)
            self.options['blur_out_mask'] = io.input_bool("Blur out mask", default_blur_out_mask)
            self.options['models_opt_on_gpu'] = io.input_bool("Place models and optimizer on GPU", default_models_opt_on_gpu)
            self.options['adabelief'] = io.input_bool("Use AdaBelief optimizer?", default_adabelief)
            self.options['lr_dropout'] = io.input_str("Use learning rate dropout", default_lr_dropout, ['n', 'y', 'cpu'])
            self.options['random_warp'] = io.input_bool("Enable random warp of samples", default_random_warp)
            self.options['random_hsv_power'] = np.clip(io.input_number("Random hue/saturation/light intensity", default_random_hsv_power, add_info="0.0 .. 0.3"), 0.0, 0.3)
            self.options['ct_mode'] = io.input_str("Color transfer for src faceset", default_ct_mode, ['none', 'rct', 'lct', 'mkl', 'idt', 'sot'])
            self.options['clipgrad'] = io.input_bool("Enable gradient clipping", default_clipgrad)

        self._opzioni_esecuzione()
        self._spegni_le_leve_incompatibili()
        self.gan_model_changed = False
        self.pretrain_just_disabled = False

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)

        self.resolution = resolution = self.options['resolution']
        self.face_type = {'h': FaceType.HALF, 'mf': FaceType.MID_FULL, 'f': FaceType.FULL,
                          'wf': FaceType.WHOLE_FACE, 'head': FaceType.HEAD}[self.options['face_type']]
        eyes_mouth_prio = self.options['eyes_mouth_prio']
        ae_dims, e_dims, d_dims, d_mask_dims = (self.options[k] for k in ('ae_dims', 'e_dims', 'd_dims', 'd_mask_dims'))
        self.pretrain = False
        self.gan_power = 0.0
        adabelief = self.options['adabelief']
        random_warp = self.options['random_warp']
        random_hsv_power = self.options['random_hsv_power']
        masked_training = self.options['masked_training']
        ct_mode = None if self.options['ct_mode'] == 'none' else self.options['ct_mode']
        allenabile = self.options['identita'] == 'learned'

        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = nn.device if models_opt_on_gpu or not self.is_training else torch.device('cpu')
        optimizer_vars_on_cpu = models_opt_device.type == 'cpu'

        model_archi = nn.DeepFakeArchi(resolution, use_fp16=False, opts='udt')
        self.encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name='encoder')
        encoder_out_ch = self.encoder.get_out_ch() * self.encoder.get_out_res(resolution) ** 2
        self.inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims * 2, name='inter')
        self.decoder = nn.DecoderH2(in_ch=self.inter.get_out_ch() * 2, d_ch=d_dims, d_mask_ch=d_mask_dims, e_dim=self.E_DIM,
                                    name='decoder', maschera_tronco=self.options['maschera_tronco'])
        self.identita = nn.Identita(self.E_DIM, allenabile=allenabile, name='identita')
        self.nets = {'encoder': self.encoder, 'inter': self.inter, 'decoder': self.decoder, 'identita': self.identita}
        self.model_filename_list = [[self.encoder, 'encoder.npy'], [self.inter, 'inter.npy'],
                                    [self.decoder, 'decoder.npy'], [self.identita, 'identita.npy']]
        for model, _ in self.model_filename_list:
            model.build()
            model.to(models_opt_device)

        def to_t(x):
            return torch.as_tensor(np.ascontiguousarray(x)).to(models_opt_device, nn.floatx)

        if self.is_training:
            lr = 5e-5
            lr_cos, lr_dropout = (500, 0.3) if self.options['lr_dropout'] in ['y', 'cpu'] else (0, 1.0)
            OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
            clipnorm = 1.0 if self.options['clipgrad'] else 0.0
            self.src_dst_saveable_weights, self.src_dst_trainable_weights = h2_src_dst_weights(self.nets)
            self.src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_opt.initialize_variables(self.src_dst_saveable_weights, vars_on_cpu=optimizer_vars_on_cpu,
                                                  lr_dropout_on_cpu=self.options['lr_dropout'] == 'cpu')
            self.model_filename_list += [(self.src_dst_opt, 'src_dst_opt.npy')]

            gpu_count = max(1, len(devices))
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size(gpu_count * bs_per_gpu)
            train_cfg = {'archi_type': self.archi_type, 'resolution': resolution, 'masked_training': masked_training,
                         'eyes_mouth_prio': eyes_mouth_prio, 'blur_out_mask': self.options['blur_out_mask'],
                         'gan_power': 0.0, 'true_face_power': 0.0, 'face_style_power': 0.0, 'bg_style_power': 0.0,
                         'pretrain': False}

            def src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,
                              warped_dst, target_dst, target_dstm, target_dstm_em):
                s, d = h2_train_step(self.nets, self.src_dst_opt, self.src_dst_trainable_weights,
                                     [to_t(x) for x in (warped_src, target_src, target_srcm, target_srcm_em,
                                                        warped_dst, target_dst, target_dstm, target_dstm_em)],
                                     train_cfg, gpu_count, loss_extra=self._loss_extra)
                return s.detach().cpu().numpy(), d.detach().cpu().numpy()
            self.src_dst_train = src_dst_train

            def AE_view(warped_src, warped_dst):
                return [x.cpu().numpy() for x in h2_view(self.nets, to_t(warped_src), to_t(warped_dst))]
            self.AE_view = AE_view
        else:
            def AE_merge(warped_dst, morph_value):
                return [x.cpu().numpy() for x in h2_merge(self.nets, to_t(warped_dst), morph_value)]
            self.AE_merge = AE_merge

        # I supervisori prima del caricamento: i vettori AdaFace servono a chi li congela.
        self._loss_extra = None
        self._termini_h1 = {}
        self._riferimento_src_calcolato = None
        self._riferimento_dst_calcolato = None
        potenze = self._potenze_accese() if self.is_training else {}
        serve_adaface = self._serve_adaface(potenze, allenabile)
        self._adaface = self._dinov2 = None
        if potenze or serve_adaface:
            self._richiedi_cuda()
            self._carica_supervisori(serve_adaface, 'dino_power' in potenze)

        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            caricato = (not self.is_first_run()) and model.load_weights(self.get_strpath_storage_for_file(filename))
            if not caricato:
                model.init_weights()
                if model is self.identita and not allenabile:
                    self._imposta_identita_da_adaface()

        if self.is_first_run() and self.options.get('innesto'):
            copiate = innesto.copia_pesi(self.nets, self.saved_models_path, self.options['innesto'],
                                         inter=bool(self.options.get('innesto_inter', False)))
            io.log_info(f"H2: innestati encoder e decoder ({copiate} chiavi) da «{self.options['innesto']}»"
                        + (", inter compreso" if self.options.get('innesto_inter') else "; inter dall'inizializzazione"))

        if potenze:
            # _imposta_identita_da_adaface ha gia' pagato i fino a 2000 JPEG di src: non si rilegge.
            self._theta, residuo_medio, residuo_max, self._e_src, coseno_medio, n_volti = \
                self._riferimento_src_calcolato or self._riferimento_src()
            io.log_info(f"H2 crop fisso: residuo medio per volto {residuo_medio:.2f} px, massimo {residuo_max:.2f} px")
            io.log_info(f"H2 e_src da {n_volti} volti, coseno medio dei volti al riferimento {coseno_medio:.3f}")
            if self._serve_e_dst_medio(potenze):
                # stesso risparmio del src: se i vettori congelati l'hanno gia' letto, non si rilegge
                *_, self._e_dst, coseno_dst, n_dst = self._riferimento_dst_calcolato or self._riferimento_dst()
                io.log_info(f"H2 e_dst da {n_dst} volti, coseno medio dei volti al riferimento {coseno_dst:.3f}")
            self._loss_extra = self._costruisci_loss_extra()

        if self.is_training:
            training_data_src_path = self.training_data_src_path
            training_data_dst_path = self.training_data_dst_path

            random_ct_samples_path = training_data_dst_path if ct_mode is not None else None

            cpu_count = multiprocessing.cpu_count()
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            # L'opt-in dei nomi dei file passa dal default di classe, acceso solo
            # per la durata della costruzione: qui l'on_initialize e' proprio.
            SampleGeneratorFace.default_return_filenames = True
            try:
                self.set_training_data_generators ([
                        SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                            sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_src_flip),
                            output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,   'random_hsv_shift_amount' : random_hsv_power,                                        'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                    {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                    {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                    {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                  ],
                            uniform_yaw_distribution=self.options['uniform_yaw'],
                            generators_count=src_generators_count ),

                        SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                            sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_dst_flip),
                            output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                    {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                    {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                    {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                  ],
                            uniform_yaw_distribution=self.options['uniform_yaw'],
                            generators_count=dst_generators_count )
                                 ])
            finally:
                SampleGeneratorFace.default_return_filenames = False

            self._accelera()

    def _serve_adaface(self, potenze, allenabile):
        """La rete AdaFace serve alle potenze che la usano (id, ifsr, bleed) e ai vettori
        congelati. La condizione dei vettori e' quella del ciclo di caricamento
        piu' sotto -- `not self.is_first_run()` -- e non la sola esistenza del
        file: al primo avvio i pesi si inizializzano comunque, quindi un
        identita.npy gia' su disco non evita il calcolo."""
        if potenze.keys() & {'id_power', 'ifsr_power', 'bleed_power'}:
            return True
        identita_su_disco = pathlib.Path(self.get_strpath_storage_for_file('identita.npy')).exists()
        return self.is_training and not allenabile and (self.is_first_run() or not identita_su_disco)

    def _imposta_identita_da_adaface(self):
        if not self.is_training:
            raise Exception("H2: identita.npy manca e i vettori AdaFace si calcolano solo addestrando")
        # Senza la rete _riferimento non guarda i pixel e torna un embedding di
        # zeri: imposta() li scriverebbe nei due vettori senza sollevare, e il
        # PIM resterebbe degenere per tutta la corsa.
        if self._adaface is None:
            raise Exception("H2: i vettori AdaFace richiedono la rete caricata, e _serve_adaface non l'ha prevista")
        self._riferimento_src_calcolato = self._riferimento(self.training_data_src_path)
        _, _, _, e_src, coseno_src, n_src = self._riferimento_src_calcolato
        self._riferimento_dst_calcolato = self._riferimento(self.training_data_dst_path)
        _, _, _, e_dst, coseno_dst, n_dst = self._riferimento_dst_calcolato
        self.identita.imposta(e_src, e_dst)
        io.log_info(f"H2 vettori AdaFace: src da {n_src} volti (coseno medio {coseno_src:.3f}), dst da {n_dst} (coseno medio {coseno_dst:.3f})")

    #override
    # La firma diverge dal genitore per morph_value, come in AMP: l'unico
    # chiamante di produzione e' la chiusura di get_MergerConfig.
    def predictor_func(self, face, morph_value):
        face = nn.to_data_format(face[None, ...], self.model_data_format, "NHWC")
        bgr, mask_dst_dstm, mask_src_dstm = [nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32)
                                             for x in self.AE_merge(face, morph_value)]
        return bgr[0], mask_src_dstm[0][..., 0], mask_dst_dstm[0][..., 0]

    #override
    def get_MergerConfig(self):
        morph_factor = np.clip(io.input_number("Morph factor", 1.0, add_info="0.0 .. 1.0"), 0.0, 1.0)

        def predictor_morph(face):
            return self.predictor_func(face, morph_factor)

        import merger
        return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), \
            merger.MergerConfigMasked(face_type=self.face_type, default_mode='overlay')

    #override
    def export_dfm(self):
        output_path = self.get_strpath_storage_for_file('model.dfm')
        io.log_info(f'Dumping .dfm to {output_path}')
        out_names = ['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0']
        # dynamo=False: l'esportatore nuovo e' il default in torch 2.13 e ignora
        # opset_version, scrivendo 18 dove il contratto vuole 12.
        torch.onnx.export(
            H2ExportModule(self.nets).eval(),
            (torch.zeros(1, self.resolution, self.resolution, 3, device=nn.device, dtype=nn.floatx),
             torch.zeros(1, device=nn.device, dtype=nn.floatx)),
            output_path,
            input_names=['in_face:0', 'morph_value:0'],
            output_names=out_names,
            opset_version=12,
            dynamic_axes={n: {0: 'batch'} for n in ['in_face:0'] + out_names},
            dynamo=False)
        # L'inferenza di forma perde altezza, larghezza e canali attraversando
        # le permute finali: si riscrivono sulle sole uscite, come in SAEHD.
        proto = onnx.load(output_path)
        for o, ch in zip(proto.graph.output, [1, 3, 1]):
            dims = o.type.tensor_type.shape.dim
            dims[1].dim_value = self.resolution
            dims[2].dim_value = self.resolution
            dims[3].dim_value = ch
        onnx.save(proto, output_path)


Model = H2Model
