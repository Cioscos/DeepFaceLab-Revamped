import multiprocessing

import numpy as np
import onnx
import torch

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType, XSegNet
from models import ModelBase
from samplelib import *


def xseg_loss(target, pred_logits, pred, resolution, pretrain):
    """
    La loss di XSeg, un valore per campione: forma (N,).

    A livello di modulo e non dentro on_initialize perche' e' l'unica parte
    del passo di training che un test possa esercitare -- costruire un
    XSegModel significa passare per ModelBase, che prompta l'utente. La
    chiama on_initialize, che non ne duplica il contenuto.
    """
    if pretrain:
        # Structural loss
        loss  = torch.mean (5*nn.dssim(target, pred, max_val=1.0, filter_size=int(resolution/11.6)), dim=[1])
        loss += torch.mean (5*nn.dssim(target, pred, max_val=1.0, filter_size=int(resolution/23.2)), dim=[1])
        # Pixel loss
        loss += torch.mean (10*torch.square(target-pred), dim=[1,2,3])
    else:
        loss = torch.mean( nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred_logits), dim=[1,2,3])
    return loss


class XSegExportModule(torch.nn.Module):
    """
    Il sottografo che DeepFaceLive consuma (Model.py:254-265 nella versione TF).

    Le due permute stanno DENTRO il modulo e non fuori: il contratto vuole un
    ingresso e un'uscita in NHWC con la conversione dentro il grafo esportato,
    perche' il consumatore passa il frame come lo ha e non lo riordina.

    E' un Module e non una funzione perche' torch.onnx.export traccia un
    Module: il wrapper esiste per l'export e per nient'altro, e infatti non
    compare in nessun percorso di training.

    `net` e' la facelib.XSegNet.XSegNet che export_dfm possiede (self.model),
    ma XSegNet e' un plain object, non un torch.nn.Module -- quindi
    assegnarla tale e quale a self.net non la registrerebbe come
    sotto-modulo, self.parameters() resterebbe vuoto, e torch.onnx.export
    fallirebbe con "Cannot insert a Tensor that requires grad as a constant"
    perche' i pesi non risulterebbero Parameter di NESSUN modulo raggiungibile
    dal tracciato -- misurato. Il modulo torch vero e' net.model (un nn.XSeg);
    e' quello che va registrato, e net.flow(x) non e' altro che
    net.model(x, pretrain=False).
    """
    def __init__(self, net):
        super().__init__()
        self.net = net.model if hasattr(net, "model") else net

    def forward(self, in_face):
        _, pred = self.net(in_face.permute(0,3,1,2))
        return pred.permute(0,2,3,1)


def xseg_train_step(forward, opt, weights, input_t, target_t, resolution, pretrain, gpu_count):
    """
    Un passo: forward, loss, gradienti, update. Ritorna la loss per campione.

    `loss.sum() / gpu_count` e' il gradiente del TF, non una riformulazione:
    tf.gradients di una loss vettoriale somma le sue componenti, e
    nn.average_gv_list divideva poi per il numero di GPU (Model.py:118, 126
    nella versione TF). torch.autograd.grad vuole uno scalare, e questo e' lo
    scalare che da' lo stesso gradiente.
    """
    pred_logits, pred = forward(input_t, pretrain=pretrain)
    loss = xseg_loss(target_t, pred_logits, pred, resolution, pretrain)
    opt.step( nn.gradients (loss.sum() / gpu_count, weights) )
    return loss


class XSegModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_model_class_name='XSeg', **kwargs)

    #override
    def on_initialize_options(self):
        ask_override = self.ask_override()

        if not self.is_first_run() and ask_override:
            if io.input_bool(f"Restart training?", False, help_message="Reset model weights and start training from scratch."):
                self.set_iter(0)

        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'wf')
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        if self.is_first_run():
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Choose the same as your deepfake model.").lower()

        if self.is_first_run() or ask_override:
            self.ask_batch_size(4, range=[2,16])
            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain)
        
        if not self.is_exporting and (self.options['pretrain'] and self.get_pretraining_data_path() is None):
            raise Exception("pretraining_data_path is not defined")
            
        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)
        
    #override
    def on_initialize(self):
        # NCHW sempre: nn.set_data_format solleva su qualunque altro valore.
        # Il TF ripiegava su NHWC in debug e su CPU; quel ramo si perde
        # consapevolmente, perche' core/leras e' NCHW.
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)

        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        self.resolution = resolution = 256


        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]


        place_model_on_cpu = len(devices) == 0

        # Initializing model classes
        self.model = XSegNet(name='XSeg',
                               resolution=resolution,
                               load_weights=not self.is_first_run(),
                               weights_file_root=self.get_model_root_path(),
                               training=True,
                               place_model_on_cpu=place_model_on_cpu,
                               optimizer=nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt'),
                               data_format=nn.data_format)
        
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)
            
        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            device  = self.model.device
            weights = self.model.get_weights()

            # DataParallel replica il modulo a ogni forward e distribuisce il
            # batch sui device: e' quello che il doppio tf.device del TF faceva
            # a mano, slice per GPU. Su un device solo non si passa dal wrapper
            # -- l'unico percorso verificabile su questa macchina resta quello
            # nudo. Il ramo multi-GPU non e' eseguibile qui (una sola GPU) e su
            # decisione dell'utente si da' per funzionante.
            self.forward = self.model.flow if gpu_count == 1 else \
                           torch.nn.DataParallel(self.model.model,
                                                 device_ids=[d.index for d in devices])

            def to_t(x):
                return torch.as_tensor(np.ascontiguousarray(x)).to(device, nn.floatx)
            self.to_t = to_t

            def train(input_np, target_np):
                loss = xseg_train_step(self.forward, self.model.opt, weights,
                                       to_t(input_np), to_t(target_np),
                                       resolution, self.pretrain, gpu_count)
                return loss.detach().cpu().numpy()
            self.train = train

            # initializing sample generators
            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_dst_generators_count = cpu_count // 2
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            
            if self.pretrain:
                pretrain_gen = SampleGeneratorFace(self.get_pretraining_data_path(), debug=self.is_debug(), batch_size=self.get_batch_size(),
                                    sample_process_options=SampleProcessor.Options(random_flip=True),
                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                            {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},                                                            
                                                          ],
                                    uniform_yaw_distribution=False,
                                    generators_count=cpu_count )
                self.set_training_data_generators ([pretrain_gen])
            else:   
                srcdst_generator = SampleGeneratorFaceXSeg([self.training_data_src_path, self.training_data_dst_path],
                                                            debug=self.is_debug(),
                                                            batch_size=self.get_batch_size(),
                                                            resolution=resolution,
                                                            face_type=self.face_type,
                                                            generators_count=src_dst_generators_count,
                                                            data_format=nn.data_format)

                src_generator = SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                                    sample_process_options=SampleProcessor.Options(random_flip=False),
                                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,  'warp':False, 'transform':False, 'channel_type' : SampleProcessor.ChannelType.BGR, 'border_replicate':False, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                                        ],
                                                    generators_count=src_generators_count,
                                                    raise_on_no_data=False )
                dst_generator = SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                                                    sample_process_options=SampleProcessor.Options(random_flip=False),
                                                    output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,  'warp':False, 'transform':False, 'channel_type' : SampleProcessor.ChannelType.BGR, 'border_replicate':False, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                                        ],
                                                    generators_count=dst_generators_count,
                                                    raise_on_no_data=False )

                self.set_training_data_generators ([srcdst_generator, src_generator, dst_generator])

    def view(self, input_np):
        """
        La preview: il `pred` della rete, senza gradiente.

        Un metodo e non una chiusura dentro on_initialize per la stessa ragione
        per cui xseg_loss sta a livello di modulo: una chiusura non e'
        raggiungibile da nessun test, perche' costruire un XSegModel significa
        passare per ModelBase, che prompta l'utente. Un metodo si chiama con un
        self finto.
        """
        with torch.no_grad():
            # pretrain=self.pretrain come nel TF, dove `view` eseguiva lo
            # stesso tensore `pred` costruito con quel flag: in pretrain le
            # skip connection sono azzerate, e una preview senza il flag
            # mostrerebbe un'altra rete.
            _, pred = self.forward(self.to_t(input_np), pretrain=self.pretrain)
        # Una lista di un array: onGetPreview la concatena con i propri
        # campioni (`[image_np, mask_np] + self.view(image_np)`), come
        # faceva nn.tf_sess.run([pred]).
        return [ pred.cpu().numpy() ]

    #override
    def get_model_filename_list(self):
        return self.model.model_filename_list

    #override
    def onSave(self):
        self.model.save_weights()

    #override
    def onTrainOneIter(self):
        image_np, target_np = self.generate_next_samples()[0]
        loss = self.train (image_np, target_np)
        
        return ( ('loss', np.mean(loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )
        
        if self.pretrain:
            srcdst_samples, = samples       
            image_np, mask_np = srcdst_samples     
        else:
            srcdst_samples, src_samples, dst_samples = samples
            image_np, mask_np = srcdst_samples

        I, M, IM, = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([image_np,mask_np] + self.view (image_np) ) ]
        M, IM, = [ np.repeat (x, (3,), -1) for x in [M, IM] ]

        green_bg = np.tile( np.array([0,1,0], dtype=np.float32)[None,None,...], (self.resolution,self.resolution,1) )

        result = []
        st = []
        for i in range(n_samples):
            if self.pretrain:
                ar = I[i], IM[i]
            else:
                ar = I[i]*M[i]+0.5*I[i]*(1-M[i])+0.5*green_bg*(1-M[i]), IM[i], I[i]*IM[i]+0.5*I[i]*(1-IM[i]) + 0.5*green_bg*(1-IM[i])
            st.append ( np.concatenate ( ar, axis=1) )
        result += [ ('XSeg training faces', np.concatenate (st, axis=0 )), ]

        if not self.pretrain and len(src_samples) != 0:
            src_np, = src_samples


            D, DM, = [ np.clip(nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([src_np] + self.view (src_np) ) ]
            DM, = [ np.repeat (x, (3,), -1) for x in [DM] ]

            st = []
            for i in range(n_samples):
                ar = D[i], DM[i], D[i]*DM[i] + 0.5*D[i]*(1-DM[i]) + 0.5*green_bg*(1-DM[i])
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('XSeg src faces', np.concatenate (st, axis=0 )), ]

        if not self.pretrain and len(dst_samples) != 0:
            dst_np, = dst_samples


            D, DM, = [ np.clip(nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([dst_np] + self.view (dst_np) ) ]
            DM, = [ np.repeat (x, (3,), -1) for x in [DM] ]

            st = []
            for i in range(n_samples):
                ar = D[i], DM[i], D[i]*DM[i]  + 0.5*D[i]*(1-DM[i]) + 0.5*green_bg*(1-DM[i])
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('XSeg dst faces', np.concatenate (st, axis=0 )), ]

        return result

    #override
    def get_preview_layout(self):
        #int() per la stessa ragione del gemello in Model_SAEHD: resolution
        #e' un np.int64 e il descrittore attraversa un canale JSON.
        n_samples = int(min(4, self.get_batch_size(), 800 // self.resolution ))

        def griglia(colonne, risultato):
            return { "righe": n_samples, "colonne": len(colonne),
                     "celle": [ list(colonne) for _ in range(n_samples) ],
                     "risultato": [0, colonne.index(risultato)],
                     "righe_sono_campioni": True }

        #Etichette in inglese come quelle di SAEHD e AMP: finiscono a schermo
        #-- didascalia di ogni cella, tooltip del riquadro grande, titolo
        #della finestra a dimensione naturale -- e sono le sole del roster
        #che erano rimaste in italiano.
        if self.pretrain:
            return { 'XSeg training faces': griglia(['face', 'predicted mask'],
                                                    'predicted mask') }

        applicata = ['face', 'predicted mask', 'applied mask']
        return { 'XSeg training faces': griglia(applicata, 'applied mask'),
                 'XSeg src faces':      griglia(applicata, 'applied mask'),
                 'XSeg dst faces':      griglia(applicata, 'applied mask') }

    #override
    def export_dfm (self):
        # model.onnx e non model.dfm: e' l'unico dei tre export a scrivere
        # questo nome, ed e' quello della build TF (Model.py:255).
        output_path = self.get_strpath_storage_for_file(f'model.onnx')
        io.log_info(f'Dumping .onnx to {output_path}')

        # dynamo=False: l'esportatore nuovo e' il default in torch 2.13 e
        # IGNORA opset_version, scrivendo 18 dove qui ne serve 13. Il contratto
        # con DeepFaceLive e' per modello e va rispettato alla lettera.
        torch.onnx.export(
            XSegExportModule(self.model).eval(),
            (torch.zeros(1, self.resolution, self.resolution, 3,
                         device=self.model.device, dtype=nn.floatx),),
            output_path,
            input_names  = ['in_face:0'],
            output_names = ['out_mask:0'],
            opset_version = 13,
            dynamic_axes = {'in_face:0': {0: 'batch'}, 'out_mask:0': {0: 'batch'}},
            dynamo = False)

        # L'inferenza di forma di torch.onnx.export perde altezza e larghezza
        # attraversando la permute finale di XSegExportModule.forward
        # (NCHW->NHWC): il proto dichiarava altezza e larghezza simboliche
        # (una etichettata col simbolo del batch) invece di [batch,256,256,1]
        # (misurato sul .dfm dello smoke di SAEHD, stesso difetto qui perche'
        # e' la stessa permute -- task-9-brief.md). tf2onnx (riferimento T6)
        # le dichiara concrete; qui si rilegge il proto appena scritto e si
        # riscrivono altezza, larghezza e canali (assi 1, 2 e 3) sulla sola
        # uscita, lasciando l'asse 0 (batch) simbolico -- e' un difetto nei
        # metadati dichiarati, non nel calcolo: a run time il grafo era gia'
        # corretto. L'asse dei canali e' gia' concreto qui (out_mask:0 non
        # passa da nessuno slice dipendente da un valore, a differenza
        # dell'out_celeb_face:0 di AMP, gemello di questo export): scriverlo
        # comunque tiene il codice identico a SAEHD/AMP invece di fare
        # affidamento sul fatto che torch l'abbia gia' inferito giusto.
        proto = onnx.load(output_path)
        for o in proto.graph.output:
            dims = o.type.tensor_type.shape.dim
            dims[1].dim_value = self.resolution
            dims[2].dim_value = self.resolution
            dims[3].dim_value = 1  # out_mask:0
        onnx.save(proto, output_path)

Model = XSegModel