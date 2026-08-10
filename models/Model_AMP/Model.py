import multiprocessing
import operator
from functools import partial

import numpy as np
import onnx
import torch
import torch.nn.functional as F

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *
from core.cv2ex import *


class AMPDownscale(nn.ModelBase):
    """
    Model.py:138-143 nella versione TF. leaky_relu a 0.1 dopo la conv strided.

    in_ch/out_ch/kernel_size erano gia' argomenti espliciti di on_build in
    TF; la chiusura che quel Model.py chiudeva qui era `conv_dtype`
    (Model.py:140, dtype=conv_dtype), derivata da `use_fp16`
    (Model.py:136 nella versione TF: `conv_dtype = tf.float16 if use_fp16 else tf.float32`).
    Qui e' un parametro esplicito, propagato nella stessa forma di
    `core/leras/archis/DeepFakeArchi.py:24,42` (che porta lo stesso
    `Downscale` di TF): `conv_dtype` si calcola una volta in `on_build` e va
    a `dtype=` di `nn.Conv2D`, cosi' la conv nasce gia' nel dtype giusto
    invece di doverlo scoprire a runtime da un cast a monte (che qui non
    c'e': la classe non vede mai un tensore, solo il suo dtype di
    costruzione).

    (use_fp16 va preso anche qui: ometterlo lascerebbe le conv in float32
    mentre i cast di ingresso e uscita fanno credere il contrario.)
    """
    def on_build(self, in_ch, out_ch, kernel_size=5, use_fp16=False):
        conv_dtype = torch.float16 if use_fp16 else torch.float32
        self.conv1 = nn.Conv2D(in_ch, out_ch, kernel_size=kernel_size, strides=2, padding='SAME', dtype=conv_dtype)

    def forward(self, x):
        return F.leaky_relu(self.conv1(x), 0.1)


class AMPUpscale(nn.ModelBase):
    """
    Model.py:145-151 nella versione TF. depth_to_space dopo una leaky_relu a
    0.1. `use_fp16`/`conv_dtype` propagati come in AMPDownscale (vedi il suo
    docstring) e in `DeepFakeArchi.py:24,71`.
    """
    def on_build(self, in_ch, out_ch, kernel_size=3, use_fp16=False):
        conv_dtype = torch.float16 if use_fp16 else torch.float32
        self.conv1 = nn.Conv2D(in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

    def forward(self, x):
        return nn.depth_to_space(F.leaky_relu(self.conv1(x), 0.1), 2)


class AMPResidualBlock(nn.ModelBase):
    """
    Model.py:153-163 nella versione TF.

    Le due leaky_relu sono a 0.2, non 0.1 come in Downscale/Upscale
    (Model.py:160, :162). La seconda somma l'ingresso NON trasformato
    (`inp`, non l'uscita di conv1) prima della leaky_relu (Model.py:162:
    `tf.nn.leaky_relu(inp+x, 0.2)`). `use_fp16`/`conv_dtype` propagati alle
    due conv come in AMPDownscale (vedi il suo docstring) e in
    `DeepFakeArchi.py:24,81-82`.
    """
    def on_build(self, ch, kernel_size=3, use_fp16=False):
        conv_dtype = torch.float16 if use_fp16 else torch.float32
        self.conv1 = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
        self.conv2 = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

    def forward(self, inp):
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class AMPEncoder(nn.ModelBase):
    """
    Model.py:165-190 nella versione TF.

    down1..down5/res1/res5/dense1 sono gli attributi da cui derivano le chiavi
    disco (`weight_encoder/down1/conv1/weight:0`, ecc.): i nomi non si
    toccano. `resolution` e `e_dims` sostituiscono le chiusure omonime;
    `input_ch` diventa `in_ch` (Model.py:111: `input_ch=3`).

    Model.py:188 fa `nn.pixel_norm(nn.flatten(x), axes=-1)` PRIMA di
    `dense1`: normalizzazione poi proiezione, non il contrario.

    use_fp16 fa doppio lavoro, come nel TF originale: il cast di
    ingresso/uscita qui sotto (Model.py:177-178, :186-187: `tf.cast(x,
    tf.float16)` ... `tf.cast(x, tf.float32)`, stesso schema di
    nn.DeepFakeArchi.Encoder, core/leras/archis/DeepFakeArchi.py:110-111,
    :127-128) E la propagazione a down1..down5/res1/res5 sotto, cosi' le
    loro conv nascono nello stesso dtype del tensore che le attraversa fra
    i due cast (Model.py:136,140,147,155,156 nella versione TF -- il
    `conv_dtype` chiuso su tutte le nn.Conv2D delle sei classi).

    (down1..down5/res1/res5 devono ricevere use_fp16: costruirle senza le
    lascia in float32, e il cast di ingresso/uscita fa sembrare l'opzione
    attiva mentre le conv sotto non la vedono affatto.)
    """
    def on_build(self, in_ch, e_dims, ae_dims, resolution, use_fp16):
        self.down1 = AMPDownscale(in_ch, e_dims, kernel_size=5, use_fp16=use_fp16)
        self.res1 = AMPResidualBlock(e_dims, use_fp16=use_fp16)
        self.down2 = AMPDownscale(e_dims, e_dims*2, kernel_size=5, use_fp16=use_fp16)
        self.down3 = AMPDownscale(e_dims*2, e_dims*4, kernel_size=5, use_fp16=use_fp16)
        self.down4 = AMPDownscale(e_dims*4, e_dims*8, kernel_size=5, use_fp16=use_fp16)
        self.down5 = AMPDownscale(e_dims*8, e_dims*8, kernel_size=5, use_fp16=use_fp16)
        self.res5 = AMPResidualBlock(e_dims*8, use_fp16=use_fp16)
        self.dense1 = nn.Dense((( resolution//(2**5) )**2) * e_dims*8, ae_dims)
        self.use_fp16 = use_fp16

    def forward(self, x):
        if self.use_fp16:
            x = x.to(torch.float16)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.res5(x)
        if self.use_fp16:
            x = x.to(torch.float32)
        x = nn.pixel_norm(nn.flatten(x), axes=-1)
        x = self.dense1(x)
        return x


class AMPInter(nn.ModelBase):
    """
    Model.py:193-201 nella versione TF.

    `dense2` e' la sola chiave su disco di questa rete (`inter_src/dense2/...`,
    `inter_dst/dense2/...`): AMPModel costruisce due istanze di
    questa classe, una per `name='inter_src'` e una per `name='inter_dst'`.
    """
    def on_build(self, ae_dims, inter_res, inter_dims):
        self.dense2 = nn.Dense(ae_dims, inter_res * inter_res * inter_dims)
        self.inter_res = inter_res
        self.inter_dims = inter_dims

    def forward(self, inp):
        x = inp
        x = self.dense2(x)
        x = nn.reshape_4D(x, self.inter_res, self.inter_res, self.inter_dims)
        return x


class AMPDecoder(nn.ModelBase):
    """
    Model.py:204-255 nella versione TF.

    Due rami indipendenti da `z`: il ramo immagine (upscale0..upscale3 +
    res0..res3, quattro upscale) e il ramo maschera (upscalem0..upscalem4,
    cinque upscale) -- un upscale in piu' perche' il ramo immagine chiude
    con un depth_to_space extra (Model.py:241-244) mentre quello maschera no.

    Model.py:241-244 concatena `out_conv, out_conv1, out_conv2, out_conv3`
    sull'asse canali IN QUEST'ORDINE prima del depth_to_space: scambiare due
    rami e' invisibile alla forma (tutti e quattro producono 3 canali) ma
    permuta i pixel dopo il depth_to_space.

    use_fp16 fa doppio lavoro come in AMPEncoder (vedi il suo docstring): il
    cast di ingresso/uscita qui sotto (Model.py:229-230, :252-254) E la
    propagazione a tutti i layer con pesi propri -- i dieci Upscale
    (upscale0..3, upscalem0..4) via `AMPUpscale(..., use_fp16=use_fp16)`, i
    quattro ResidualBlock via `AMPResidualBlock(..., use_fp16=use_fp16)`, e
    le cinque nn.Conv2D dirette di questa classe
    (out_convm/out_conv/out_conv1/out_conv2/out_conv3) via
    `dtype=conv_dtype`, calcolato qui una volta come in
    `DeepFakeArchi.py:24` (Model.py:221,223-226 nella versione TF).

    (Stessa ragione di AMPEncoder: senza `dtype=conv_dtype` qui, use_fp16=True
    lascerebbe tutti questi layer in float32.)
    """
    def on_build(self, inter_dims, d_dims, d_mask_dims, use_fp16):
        conv_dtype = torch.float16 if use_fp16 else torch.float32

        self.upscale0 = AMPUpscale(inter_dims, d_dims*8, kernel_size=3, use_fp16=use_fp16)
        self.upscale1 = AMPUpscale(d_dims*8, d_dims*8, kernel_size=3, use_fp16=use_fp16)
        self.upscale2 = AMPUpscale(d_dims*8, d_dims*4, kernel_size=3, use_fp16=use_fp16)
        self.upscale3 = AMPUpscale(d_dims*4, d_dims*2, kernel_size=3, use_fp16=use_fp16)

        self.res0 = AMPResidualBlock(d_dims*8, kernel_size=3, use_fp16=use_fp16)
        self.res1 = AMPResidualBlock(d_dims*8, kernel_size=3, use_fp16=use_fp16)
        self.res2 = AMPResidualBlock(d_dims*4, kernel_size=3, use_fp16=use_fp16)
        self.res3 = AMPResidualBlock(d_dims*2, kernel_size=3, use_fp16=use_fp16)

        self.upscalem0 = AMPUpscale(inter_dims, d_mask_dims*8, kernel_size=3, use_fp16=use_fp16)
        self.upscalem1 = AMPUpscale(d_mask_dims*8, d_mask_dims*8, kernel_size=3, use_fp16=use_fp16)
        self.upscalem2 = AMPUpscale(d_mask_dims*8, d_mask_dims*4, kernel_size=3, use_fp16=use_fp16)
        self.upscalem3 = AMPUpscale(d_mask_dims*4, d_mask_dims*2, kernel_size=3, use_fp16=use_fp16)
        self.upscalem4 = AMPUpscale(d_mask_dims*2, d_mask_dims*1, kernel_size=3, use_fp16=use_fp16)
        self.out_convm = nn.Conv2D(d_mask_dims*1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)

        self.out_conv  = nn.Conv2D(d_dims*2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
        self.out_conv1 = nn.Conv2D(d_dims*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
        self.out_conv2 = nn.Conv2D(d_dims*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
        self.out_conv3 = nn.Conv2D(d_dims*2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)

        self.use_fp16 = use_fp16

    def forward(self, z):
        if self.use_fp16:
            z = z.to(torch.float16)

        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        x = self.upscale3(x)
        x = self.res3(x)

        x = torch.sigmoid(nn.depth_to_space(torch.cat((self.out_conv(x),
                                                        self.out_conv1(x),
                                                        self.out_conv2(x),
                                                        self.out_conv3(x)), nn.conv2d_ch_axis), 2))
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = self.upscalem3(m)
        m = self.upscalem4(m)
        m = torch.sigmoid(self.out_convm(m))

        if self.use_fp16:
            x = x.to(torch.float32)
            m = m.to(torch.float32)
        return x, m


def amp_inter_rnd_binomial(batch_size, inter_dims, morph_factor, device=None, dtype=torch.float32):
    """
    La maschera che sceglie quali canali del codice vengono da inter_src
    (Model.py:360-365 nella versione TF).

    Nel TF e' `tf.stack([tf.random.shuffle(tf.concat([tile(1, k), tile(0,
    inter_dims-k)], 0)) for _ in range(bs_per_gpu)], 0)[...,None,None]` con
    k = int(inter_dims*morph_factor): una **permutazione** di k uni e
    inter_dims-k zeri, indipendente per ogni campione del batch. Non e' un
    campionamento di Bernoulli a probabilita' morph_factor -- che darebbe la
    stessa media e un conteggio fluttuante -- e non e' una maschera sola
    replicata sul batch.

    Un passo stocastico non e' confrontabile con un riferimento numerico: la
    correttezza di questa funzione -- permutazione e non Bernoulli, e
    indipendente per campione -- si verifica sulle proprieta' della maschera
    prodotta, non sui suoi valori.

    `torch.no_grad()` e' la trascrizione di tf.stop_gradient (Model.py:365).
    In torch e' gia' vero senza: ones/zeros/randperm sono foglie senza
    gradiente e il risultato ha requires_grad False comunque. Resta scritto
    perche' dichiara l'intento nel punto in cui il TF lo dichiarava, e perche'
    e' cio' che continuerebbe a valere se un domani la maschera nascesse da un
    tensore del grafo.

    `[...,None,None]` porta la forma a (bs, inter_dims, 1, 1): sotto NCHW
    broadcasta sui pixel dei due codici inter, che sono (bs, inter_dims,
    inter_res, inter_res).
    """
    inter_dims_bin = int(inter_dims*morph_factor)
    with torch.no_grad():
        base = torch.cat([torch.ones (inter_dims_bin,                dtype=dtype, device=device),
                          torch.zeros(inter_dims - inter_dims_bin,   dtype=dtype, device=device)], 0)
        rnd = torch.stack([base[torch.randperm(inter_dims, device=device)] for _ in range(batch_size)], 0)
    return rnd[...,None,None]


def amp_morph_slice(dst_inter_src_code, dst_inter_dst_code, inter_dims, morph_value):
    """
    Model.py:370-372 nella versione TF. `tf.cast(..., tf.int32)` TRONCA verso
    zero, non arrotonda: con inter_dims=1024 e morph=0.65 il taglio e' a 665
    (int(665.6)), non a 666. `int()` di Python fa lo stesso, ed e' il motivo
    per cui questa riga e' int() e non round().

    I due estremi sono percorsi a ogni anteprima (onGetPreview chiama AE_view
    con 0.0 e 1.00, Model.py:664-670): morph=0.0 lascia vuota la prima fetta,
    morph=1.0 la seconda. torch accetta entrambe le fette vuote e il concat le
    assorbe; e' pinnato da test_the_morph_slice_truncates_and_does_not_round.
    """
    k = int(inter_dims * morph_value)
    return torch.cat((dst_inter_src_code[:, :k], dst_inter_dst_code[:, k:]), 1)


def amp_flow(nets, warped_src, warped_dst, inter_rnd_binomial, morph_value=None):
    """
    Il forward delle quattro reti (Model.py:354-376 nella versione TF).

    `inter_rnd_binomial` entra da fuori e non e' generata qui: il passo di
    training di AMP e' stocastico (Model.py:360-367) e un passo stocastico
    non e' confrontabile con un riferimento numerico. La produzione la ottiene
    da amp_inter_rnd_binomial; chi vuole un forward deterministico la passa
    dall'esterno.

    `morph_value=None` salta il terzo giro del decoder e ritorna None nelle
    ultime due posizioni. Non e' un'ottimizzazione ma l'unica trascrizione
    possibile: nel TF quel ramo esisteva nel grafo (Model.py:376) ma
    `train` fetchava solo [src_loss, dst_loss, train_op] (Model.py:486) e
    il suo feed_dict non passava morph_value_t (Model.py:487-494). Calcolarlo
    in torch eager richiederebbe un valore di morph che il chiamante del
    passo di training non ha.
    """
    src_code = nets['encoder'](warped_src)
    dst_code = nets['encoder'](warped_dst)

    src_inter_src_code, src_inter_dst_code = nets['inter_src'](src_code), nets['inter_dst'](src_code)
    dst_inter_src_code, dst_inter_dst_code = nets['inter_src'](dst_code), nets['inter_dst'](dst_code)

    src_code = src_inter_src_code * inter_rnd_binomial + src_inter_dst_code * (1-inter_rnd_binomial)
    dst_code = dst_inter_dst_code

    pred_src_src, pred_src_srcm = nets['decoder'](src_code)
    pred_dst_dst, pred_dst_dstm = nets['decoder'](dst_code)

    if morph_value is None:
        return (pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, None, None)

    src_dst_code = amp_morph_slice(dst_inter_src_code, dst_inter_dst_code,
                                   src_inter_src_code.shape[1], morph_value)
    pred_src_dst, pred_src_dstm = nets['decoder'](src_dst_code)
    return (pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm,
            pred_src_dst, pred_src_dstm)


def amp_g_weights(nets):
    """
    I pesi che il passo allena: encoder + decoder, e basta (Model.py:301).

    I due Inter sono costruiti, salvati e ricaricati (Model.py:288-289) ma
    nessun update li tocca, perche' nn.gradients deriva rispetto a questa
    lista (Model.py:464). E' un difetto dell'originale -- la copia dormiente
    accanto porta commentata la riga che li avrebbe aggiunti sotto random_warp
    (defModel.py:299-300) -- e resta tale: il vincolo di questa migrazione e'
    che il comportamento non cambi, non che migliori. Il dump lo conferma
    (after_inter_* identico a weight_inter_*, nessun grad_inter_*), e lo fissa
    test_the_inter_networks_are_not_updated_by_the_step.

    Sta in una funzione e non inline in AMPModel perche' e' quello che i test
    devono poter chiamare: una lista ricostruita dentro il test verificherebbe
    se stessa.
    """
    return nets['encoder'].get_weights() + nets['decoder'].get_weights()


def amp_dloss_ones(logits):
    """DLossOnes, Model.py:333-334 nella versione TF: un valore per campione."""
    return torch.mean( nn.sigmoid_cross_entropy_with_logits(labels=torch.ones_like(logits), logits=logits), dim=[1,2,3])


def amp_dloss_zeros(logits):
    """DLossZeros, Model.py:336-337 nella versione TF."""
    return torch.mean( nn.sigmoid_cross_entropy_with_logits(labels=torch.zeros_like(logits), logits=logits), dim=[1,2,3])


def amp_mask_blur(targetm, resolution):
    """
    La maschera sfumata e riportata a [0,1] (Model.py:385-389 nella versione TF).

    Il raggio e' `resolution // 32` **senza** il `max(1, ...)` che SAEHD ha al
    suo posto (models/Model_SAEHD/Model.py:173): a resolution 64 -- il minimo
    che AMP accetta, Model.py:48 lo clippa a 64 -- il raggio e' 2, quindi la
    differenza non e' raggiungibile dai prompt; resta trascritto com'e' perche'
    e' com'e'.

    Sta in una funzione perche' i due chiamanti sono due passi distinti
    (amp_loss e amp_gan_train_step), come saehd_mask_blur per gli stessi due.
    """
    return torch.clamp( nn.gaussian_blur(targetm, resolution // 32), 0, 0.5) * 2


def amp_blur_out_mask(target, targetm, resolution):
    """
    Un lato di blur_out_mask (Model.py:393-404 nella versione TF), src o dst.

    `y = torch.where(y == 0, ones, y)` (Model.py:398 e :403) e' la guardia
    sulla divisione: dove la maschera sfocata vale esattamente 1 il residuo e'
    0, e senza la sostituzione x/y darebbe inf (o nan quando anche x e' zero),
    che l'anti-maschera propagherebbe a tutta la loss. Lo isola
    test_blur_out_mask_guards_the_division_by_zero -- con il blur vero il caso
    y == 0 non capita su un ingresso casuale, quindi il ramo va esercitato con
    uno spy, non sperando in un ingresso fortunato.

    Identica riga per riga a saehd_blur_out_mask (models/Model_SAEHD/Model.py:191):
    non e' importata da li' perche' ModelBase tiene i modelli indipendenti --
    e perche' quel sorgente e' la trascrizione di *un altro* Model.py, che
    puo' divergere.
    """
    sigma = resolution / 128

    targetm_anti = 1-targetm

    x = nn.gaussian_blur(target*targetm_anti, sigma)
    y = 1-nn.gaussian_blur(targetm, sigma)
    y = torch.where(y == 0, torch.ones_like(y), y)
    return target*targetm + (x/y)*targetm_anti


def amp_loss(target_src, target_srcm, target_srcm_em,
             target_dst, target_dstm, target_dstm_em,
             pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm,
             resolution, blur_out_mask, gan_power, GAN=None):
    """
    Le due loss per campione e la G_loss (Model.py:382-462 nella versione TF).

    Ritorna tre vettori di lunghezza batch, non tre scalari: e' cosi' che
    onTrainOneIter media le prime due (Model.py:657) ed e' la forma che
    tf.gradients sommava sulla terza.

    src_loss e dst_loss NON contengono i termini che vivono solo nel gradiente
    (Model.py:438-439 e :456-462): quelli stanno in G_loss. Nessun confronto
    sulle loss o sui gradienti li vede: portare a 1e-5 il coefficiente 1e-6 di
    Model.py:461 muove il peggiore dei 64 gradienti del caso con GAN da
    9.4730e+04 a 9.6395e+04 ULP, l'1.8%, contro un tetto di 2e+05 -- misurato,
    non supposto.

    Due termini dssim sempre, senza il ramo `if resolution < 256` di SAEHD
    (Model.py:417-420): a resolution 96 i filter_size sono int(96/11.6) == 8 e
    int(96/23.2) == 4.

    Il termine eyes+mouth (Model.py:426-428) e' sempre attivo: in AMP non
    esiste l'opzione eyes_mouth_prio di SAEHD, quindi target_srcm_em e
    target_dstm_em sono vivi in ogni caso.

    CONVENZIONE D'INGRESSO, e diverge da quella della gemella di SAEHD.
    `blur_out_mask` e' un parametro di questa funzione e viene applicato **qui
    dentro** (:475-477), quindi i target che arrivano devono essere **grezzi**.
    In models/Model_SAEHD/Model.py la stessa riga del TF sta invece **fuori**:
    saehd_src_dst_loss non ha nemmeno il parametro, e sono i due chiamanti ad
    applicare saehd_blur_out_mask prima (Model.py:336-338 e :440-441 di quel
    file), quindi li' i target arrivano **gia' sfocati**.

    Nel TF le due erano nello stesso posto -- entrambe inline nello stesso
    blocco (Model.py:393-404 di AMP e :391-402 di SAEHD nella versione TF) --
    quindi la divergenza e' del porting, non degli originali. Non e' un
    difetto: le due funzioni sono numericamente giuste ciascuna con la propria
    convenzione. E' una trappola per chi
    copia una chiamata dall'una all'altra: passare qui target gia' sfocati, o
    passare a saehd_src_dst_loss target grezzi con l'opzione accesa, non
    solleva niente e da' numeri sbagliati in silenzio. Prima di spostare una
    delle due si tenga presente che sarebbe un cambio di comportamento su
    modelli gia' validati e gia' passati allo smoke.

    `GAN` serve solo quando gan_power != 0 e allora e' obbligatorio; con
    gan_power == 0 il ramo non esiste, che e' la ragione per cui i due casi
    senza discriminatore possono passare None. Qui il discriminatore entra
    solo in **avanti**: i suoi pesi li aggiorna amp_gan_train_step, col
    proprio ottimizzatore. Delle quattro applicazioni di Model.py:443-446 qui
    ce ne sono due, quelle su pred_src_src e pred_dst_dst: le altre due
    (sui target) alimentano la sola gpu_GAN_loss, e nel TF il primo sess.run
    -- che fetchava [src_loss, dst_loss, train_op] (Model.py:486) -- non le
    valutava.
    """
    target_srcm_blur = amp_mask_blur(target_srcm, resolution)      # Model.py:385-386 e :388-389
    target_dstm_blur = amp_mask_blur(target_dstm, resolution)
    target_srcm_anti_blur = 1.0-target_srcm_blur                   # Model.py:390-391
    target_dstm_anti_blur = 1.0-target_dstm_blur

    if blur_out_mask:                                              # Model.py:393-404
        target_src = amp_blur_out_mask(target_src, target_srcm, resolution)
        target_dst = amp_blur_out_mask(target_dst, target_dstm, resolution)

    target_src_masked = target_src*target_srcm_blur                # Model.py:406-414
    target_dst_masked = target_dst*target_dstm_blur
    target_src_anti_masked = target_src*target_srcm_anti_blur
    target_dst_anti_masked = target_dst*target_dstm_anti_blur

    pred_src_src_masked = pred_src_src*target_srcm_blur
    pred_dst_dst_masked = pred_dst_dst*target_dstm_blur
    pred_src_src_anti_masked = pred_src_src*target_srcm_anti_blur
    pred_dst_dst_anti_masked = pred_dst_dst*target_dstm_anti_blur

    # Structural loss -- Model.py:417-420
    src_loss =  torch.mean (5*nn.dssim(target_src_masked, pred_src_src_masked, max_val=1.0, filter_size=int(resolution/11.6)), dim=[1])
    src_loss += torch.mean (5*nn.dssim(target_src_masked, pred_src_src_masked, max_val=1.0, filter_size=int(resolution/23.2)), dim=[1])
    dst_loss =  torch.mean (5*nn.dssim(target_dst_masked, pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/11.6) ), dim=[1])
    dst_loss += torch.mean (5*nn.dssim(target_dst_masked, pred_dst_dst_masked, max_val=1.0, filter_size=int(resolution/23.2) ), dim=[1])

    # Pixel loss -- Model.py:423-424
    src_loss += torch.mean (10*torch.square(target_src_masked-pred_src_src_masked), dim=[1,2,3])
    dst_loss += torch.mean (10*torch.square(target_dst_masked-pred_dst_dst_masked), dim=[1,2,3])

    # Eyes+mouth prio loss -- Model.py:427-428
    src_loss += torch.mean (300*torch.abs (target_src*target_srcm_em-pred_src_src*target_srcm_em), dim=[1,2,3])
    dst_loss += torch.mean (300*torch.abs (target_dst*target_dstm_em-pred_dst_dst*target_dstm_em), dim=[1,2,3])

    # Mask loss -- Model.py:431-432
    src_loss += torch.mean ( 10*torch.square( target_srcm - pred_src_srcm ),dim=[1,2,3] )
    dst_loss += torch.mean ( 10*torch.square( target_dstm - pred_dst_dstm ),dim=[1,2,3] )

    G_loss = src_loss + dst_loss                                   # Model.py:436-439
    # dst-dst background weak loss
    G_loss += torch.mean(0.1*torch.square(pred_dst_dst_anti_masked-target_dst_anti_masked),dim=[1,2,3] )
    G_loss += 0.000001*nn.total_variation_mse(pred_dst_dst_anti_masked)

    if gan_power != 0:                                             # Model.py:442-462
        pred_src_src_d = GAN(pred_src_src_masked)
        pred_dst_dst_d = GAN(pred_dst_dst_masked)

        G_loss += (amp_dloss_ones(pred_src_src_d.center) + amp_dloss_ones(pred_src_src_d.out) + \
                   amp_dloss_ones(pred_dst_dst_d.center) + amp_dloss_ones(pred_dst_dst_d.out)
                  ) * gan_power

        # Minimal src-src-bg rec with total_variation_mse to suppress random bright dots from gan
        G_loss += 0.000001*nn.total_variation_mse(pred_src_src)
        G_loss += 0.02*torch.mean(torch.square(pred_src_src_anti_masked-target_src_anti_masked),dim=[1,2,3] )

    return src_loss, dst_loss, G_loss


def amp_train_step(nets, opt, G_weights, batch, cfg, gpu_count, inter_rnd_binomial=None):
    """
    Il primo dei due passi di onTrainOneIter (Model.py:484-496 nella versione
    TF): forward, loss, gradienti, update. Ritorna (src_loss, dst_loss), le due
    loss per campione **senza** i termini che vivono solo dentro G_loss.

    `G_loss.sum() / gpu_count` e' il gradiente del TF, non una riformulazione:
    tf.gradients di una loss vettoriale ne somma le componenti (Model.py:464) e
    nn.average_gv_list divideva poi per il numero di GPU (Model.py:478).
    torch.autograd.grad vuole uno scalare, e questo e' lo scalare che da' lo
    stesso gradiente. Stessa identita' che vale per SAEHD.

    `inter_rnd_binomial=None` e' il percorso di produzione: la maschera si
    campiona qui, una per iterazione, come Model.py:360-365 la campionava a
    ogni sess.run. Passarla dall'esterno rende il passo deterministico e
    riproducibile.

    Il terzo giro del decoder resta fuori (morph_value non passato ad
    amp_flow): si veda il docstring di amp_flow per il perche' non e'
    un'ottimizzazione.
    """
    warped_src, target_src, target_srcm, target_srcm_em, \
    warped_dst, target_dst, target_dstm, target_dstm_em = batch

    if inter_rnd_binomial is None:
        inter_rnd_binomial = amp_inter_rnd_binomial(warped_src.shape[0], cfg['inter_dims'],
                                                    cfg['morph_factor'],
                                                    warped_src.device, warped_src.dtype)

    (pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, _, _) = \
        amp_flow(nets, warped_src, warped_dst, inter_rnd_binomial)

    src_loss, dst_loss, G_loss = amp_loss(
        target_src, target_srcm, target_srcm_em,
        target_dst, target_dstm, target_dstm_em,
        pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm,
        cfg['resolution'], cfg['blur_out_mask'], cfg['gan_power'], nets.get('GAN'))

    opt.step( nn.gradients (G_loss.sum() / gpu_count, G_weights) )
    return src_loss, dst_loss


def amp_gan_train_step(nets, GAN, GAN_opt, batch, cfg, gpu_count, inter_rnd_binomial=None):
    """
    Il secondo passo, quello del discriminatore (Model.py:442-454 e :500-509
    nella versione TF). Ritorna la GAN_loss, un valore per campione.

    Rifa' il forward del generatore: e' cio' che il secondo `nn.tf_sess.run`
    faceva, sui pesi src_dst **gia' aggiornati** dal passo 1 (Model.py:652-655
    li applica in sequenza) e con una **seconda** maschera binomiale.
    Ri-campionarla non e' un dettaglio: GAN_train_op dipende da
    gpu_pred_src_src_masked -> gpu_src_code -> la binomiale, e tf.random.shuffle
    e' un'operazione, non una costante, quindi ogni sess.run ne estrae una
    nuova. Il dump ne porta due, distinte (input_inter_rnd_binomial e
    input_inter_rnd_binomial_gan), e le fissa
    test_the_gan_step_resamples_the_binomial_mask.

    Le otto uscite della UNetPatchDiscriminator (due teste per ciascuno dei
    quattro ingressi) entrano tutte, ciascuna col suo 1/8: dimenticare una
    testa cambia la loss di un ottavo.

    Costo che il TF non pagava: amp_flow calcola anche pred_src_srcm e
    pred_dst_dstm, il ramo maschera del decoder, che questa loss non nomina --
    nel grafo pigro non venivano valutati. Numericamente e' un no-op; per
    toglierlo servirebbe un decoder con l'uscita maschera opzionale, cioe'
    cambiare AMPDecoder, che il contratto delle predizioni fissa com'e'.
    """
    warped_src, target_src, target_srcm, target_srcm_em, \
    warped_dst, target_dst, target_dstm, target_dstm_em = batch

    resolution = cfg['resolution']

    if inter_rnd_binomial is None:
        inter_rnd_binomial = amp_inter_rnd_binomial(warped_src.shape[0], cfg['inter_dims'],
                                                    cfg['morph_factor'],
                                                    warped_src.device, warped_src.dtype)

    (pred_src_src, _, pred_dst_dst, _, _, _) = \
        amp_flow(nets, warped_src, warped_dst, inter_rnd_binomial)

    target_srcm_blur = amp_mask_blur(target_srcm, resolution)
    target_dstm_blur = amp_mask_blur(target_dstm, resolution)

    if cfg['blur_out_mask']:
        target_src = amp_blur_out_mask(target_src, target_srcm, resolution)
        target_dst = amp_blur_out_mask(target_dst, target_dstm, resolution)

    target_src_masked = target_src*target_srcm_blur
    target_dst_masked = target_dst*target_dstm_blur
    pred_src_src_masked = pred_src_src*target_srcm_blur
    pred_dst_dst_masked = pred_dst_dst*target_dstm_blur

    pred_src_src_d = GAN(pred_src_src_masked)                      # Model.py:443-446
    pred_dst_dst_d = GAN(pred_dst_dst_masked)
    target_src_d = GAN(target_src_masked)
    target_dst_d = GAN(target_dst_masked)

    GAN_loss = (amp_dloss_ones (target_src_d.center)   + amp_dloss_ones (target_src_d.out) + \
                amp_dloss_zeros(pred_src_src_d.center) + amp_dloss_zeros(pred_src_src_d.out) + \
                amp_dloss_ones (target_dst_d.center)   + amp_dloss_ones (target_dst_d.out) + \
                amp_dloss_zeros(pred_dst_dst_d.center) + amp_dloss_zeros(pred_dst_dst_d.out)
               ) * (1.0 / 8)

    GAN_opt.step( nn.gradients (GAN_loss.sum() / gpu_count, GAN.get_weights()) )
    return GAN_loss

def amp_view(nets, warped_src, warped_dst, morph_value, cfg, inter_rnd_binomial=None):
    """
    Le cinque predizioni della preview, nel loro ordine (Model.py:513 nella
    versione TF): pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst,
    pred_src_dstm.

    L'ordine e' un contratto con onGetPreview, che le spacchetta posizionali
    (Model.py:664, :666-670). `torch.no_grad()` e' quello che nel TF era
    gratis: nn.tf_sess.run non costruiva nessun grafo all'indietro.

    Passa da amp_flow e non da una seconda trascrizione: e' lo stesso cablaggio
    che il training percorre. Costa in piu' il ramo maschera di pred_src_srcm, che
    AE_view non chiedeva -- ma la preview non e' per iterazione (una all'avvio,
    una a ogni salvataggio).

    `inter_rnd_binomial=None` campiona qui, come faceva ogni sess.run di
    AE_view: nel TF la maschera era un'operazione dentro lo stesso grafo
    (Model.py:360-365) e pred_src_src la attraversava, quindi ognuna delle sei
    chiamate di onGetPreview ne estraeva una nuova. I test la passano
    dall'esterno, che e' l'unico modo di confrontare due cablaggi.
    """
    if inter_rnd_binomial is None:
        inter_rnd_binomial = amp_inter_rnd_binomial(warped_src.shape[0], cfg['inter_dims'],
                                                    cfg['morph_factor'],
                                                    warped_src.device, warped_src.dtype)
    with torch.no_grad():
        (pred_src_src, _, pred_dst_dst, pred_dst_dstm,
         pred_src_dst, pred_src_dstm) = amp_flow(nets, warped_src, warped_dst,
                                                 inter_rnd_binomial, morph_value)

    return [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm]


class AMPExportModule(torch.nn.Module):
    """
    Il sottografo che DeepFaceLive consuma (Model.py:591-613 nella versione TF).

    Il taglio NON passa da amp_morph_slice, ed e' la sola differenza che conta
    fra questo modulo e amp_merge. amp_morph_slice fa `int(inter_dims *
    morph_value)`: in eager e' giusto, ma sotto il tracciamento di
    torch.onnx.export l'`int()` di un tensore lo trasforma in una costante
    Python, morph_value diventa un ingresso MAI letto dal grafo tracciato, e
    torch.onnx.export lo scarta dal grafo prodotto nonostante compaia in
    input_names -- misurato riapplicando questa sostituzione:
    `proto.graph.input` risulta `['in_face:0']`, non
    `['in_face:0', 'morph_value:0']`, e onnxruntime rifiuta la sessione con
    `Invalid input name: morph_value:0` appena la si alimenta.
    torch.narrow con k tensore traccia invece in uno Slice ONNX con indice
    calcolato a run time, e morph_value resta un ingresso vero del grafo.

    `.to(torch.int64)` tronca verso zero come il `tf.cast(..., tf.int32)` del
    TF, che e' la stessa ragione per cui amp_morph_slice usa int() e non
    round(): con inter_dims=1024 e morph=0.65 il taglio e' a 665, non a 666.

    Che morph_value resti un ingresso dichiarato del grafo lo prova
    test_the_exported_amp_declares_the_contract; che lo slice resti dinamico
    (l'output numerico dipenda davvero dal suo valore) lo prova
    test_the_exported_amp_morph_is_dynamic; che coincida con torch eager a
    ogni valore, compresi i due estremi con una fetta vuota, lo prova
    test_the_exported_amp_agrees_with_eager_torch_at_every_morph.

    out_face_mask esce dal ramo dst puro (dst_inter_dst_code) e non dal codice
    miscelato: e' l'asimmetria del grafo TF, e non dipende da morph_value.
    """
    def __init__(self, nets, inter_dims):
        super().__init__()
        self.nets       = torch.nn.ModuleDict(nets)
        self.inter_dims = inter_dims

    def forward(self, in_face, morph_value):
        x = in_face.permute(0,3,1,2)

        dst_code           = self.nets['encoder'](x)
        dst_inter_src_code = self.nets['inter_src'](dst_code)
        dst_inter_dst_code = self.nets['inter_dst'](dst_code)

        k = (self.inter_dims * morph_value[0]).to(torch.int64)
        src_dst_code = torch.cat(
            (torch.narrow(dst_inter_src_code, 1, 0, k),
             torch.narrow(dst_inter_dst_code, 1, k, self.inter_dims - k)), 1)

        pred_src_dst, pred_src_dstm = self.nets['decoder'](src_dst_code)
        _,            pred_dst_dstm = self.nets['decoder'](dst_inter_dst_code)

        return (pred_dst_dstm.permute(0,2,3,1),
                pred_src_dst.permute(0,2,3,1),
                pred_src_dstm.permute(0,2,3,1))


def amp_merge(nets, warped_dst, morph_value):
    """
    Il grafo del merger: tre tensori (Model.py:518-532 nella versione TF), nel
    solo ordine che predictor_func accetta -- bgr, mask_dst_dstm, mask_src_dstm,
    cioe' pred_src_dst, pred_dst_dstm, pred_src_dstm.

    Prende il solo warped_dst: il merger ha in mano un frame dst e nient'altro.
    E' anche il motivo per cui questa non e' una selezione da amp_flow -- che
    vorrebbe anche un warped_src e una maschera binomiale, e pagherebbe encoder
    e decoder anche sul ramo src, mentre qui si paga per *fotogramma*. Che i
    due cablaggi non divergano lo fissa
    test_the_merge_is_the_same_wiring_as_the_flow.

    `inter_dims` si legge dalla forma del codice (`shape[1]`), come in
    amp_flow: nel TF era la chiusura omonima (Model.py:524-526).
    """
    with torch.no_grad():
        dst_code = nets['encoder'](warped_dst)
        dst_inter_src_code = nets['inter_src'](dst_code)
        dst_inter_dst_code = nets['inter_dst'](dst_code)

        src_dst_code = amp_morph_slice(dst_inter_src_code, dst_inter_dst_code,
                                       dst_inter_src_code.shape[1], morph_value)

        pred_src_dst, pred_src_dstm = nets['decoder'](src_dst_code)
        _,            pred_dst_dstm = nets['decoder'](dst_inter_dst_code)

    return [pred_src_dst, pred_dst_dstm, pred_src_dstm]



class AMPModel(ModelBase):

    #override
    def on_initialize_options(self):
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 224)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'wf')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_inter_dims         = self.options['inter_dims']         = self.load_or_def_option('inter_dims', 1024)

        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_morph_factor       = self.options['morph_factor']       = self.options.get('morph_factor', 0.5)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)
        default_lr_dropout         = self.options['lr_dropout']         = self.load_or_def_option('lr_dropout', 'n')
        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(8)

        if self.is_first_run():
            resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 32 .")
            resolution = np.clip ( (resolution // 32) * 32, 64, 640)
            self.options['resolution'] = resolution
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['f','wf','head'], help_message="whole face / head").lower()


        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims']    = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )
            self.options['inter_dims'] = np.clip ( io.input_int("Inter dimensions", default_inter_dims, add_info="32-2048", help_message="Should be equal or more than AutoEncoder dimensions. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 2048 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

            morph_factor = np.clip ( io.input_number ("Morph factor.", default_morph_factor, add_info="0.1 .. 0.5", help_message="Typical fine value is 0.5"), 0.1, 0.5 )
            self.options['morph_factor'] = morph_factor

        if self.is_first_run() or ask_override:
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')
            self.options['blur_out_mask'] = io.input_bool ("Blur out mask", default_blur_out_mask, help_message='Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required.')
            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)

        if self.is_first_run() or ask_override:
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 5.0", help_message="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1"), 0.0, 5.0 )

            if self.options['gan_power'] != 0.0:
                gan_patch_size = np.clip ( io.input_int("GAN patch size", default_gan_patch_size, add_info="3-640", help_message="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8." ), 3, 640 )
                self.options['gan_patch_size'] = gan_patch_size

                gan_dims = np.clip ( io.input_int("GAN dimensions", default_gan_dims, add_info="4-512", help_message="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16." ), 4, 512 )
                self.options['gan_dims'] = gan_dims

            self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. If src faceset is deverse enough, then lct mode is fine in most cases.")
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

        self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        # "NCHW" era gia' cablato nel TF (Model.py:107 nella versione TF): AMP
        # non aveva il ripiego su NHWC che SAEHD sceglieva senza GPU o in
        # debug, quindi qui non c'e' nessuna decisione da prendere.
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)

        input_ch=3
        resolution  = self.resolution = self.options['resolution']
        e_dims      = self.options['e_dims']
        ae_dims     = self.options['ae_dims']
        inter_dims  = self.inter_dims = self.options['inter_dims']
        inter_res   = self.inter_res = resolution // 32
        d_dims      = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        face_type   = self.face_type = {'f'    : FaceType.FULL,
                                        'wf'   : FaceType.WHOLE_FACE,
                                        'head' : FaceType.HEAD}[ self.options['face_type'] ]
        morph_factor = self.options['morph_factor']
        gan_power    = self.gan_power = self.options['gan_power']
        random_warp  = self.options['random_warp']

        blur_out_mask = self.options['blur_out_mask']

        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        # use_fp16 si trascrive e non si cancella: le sei classi di rete lo
        # prendono e lo propagano alle proprie conv.
        # `export_dfm` esporta per davvero (AMPExportModule), quindi
        # un modello con use_fp16=True arriva a essere esportato -- non in
        # errore, in silenzio. self.nets viene costruito qui sotto con questo
        # use_fp16 (AMPEncoder/AMPDecoder castano a fp16 e tornano a fp32
        # internamente, Model.py:113-138 e :189-243), e AMPExportModule
        # avvolge self.nets cosi' come sono: il grafo ONNX prodotto porta i
        # nodi Cast fp16 delle conv. Il percorso quantizzato scrive
        # initializer fp16 reali per encoder e decoder, non una semplice
        # riscrittura del tipo dichiarato.
        # Il prompt "Export quantized?" resta raggiungibile come prima
        # (mainscripts/ExportDFM.py:18-22 costruisce il modello con
        # is_exporting=True e chiama export_dfm solo *dopo*; on_initialize gira
        # dentro ModelBase.__init__ -- `grep -n "is_exporting"
        # models/ModelBase.py mainscripts/ExportDFM.py` -> ModelBase.py:25,:40 e
        # ExportDFM.py:19) ma la sua conseguenza e' cambiata: chi risponde "y"
        # non vede piu' un errore leggibile, vede un .dfm fp16 non validato.
        # Chi tocca l'fp16 di AMP (fuori da questa fase) lo trovi qui, non lo
        # dia per bloccato da un NotImplementedError che non c'e' piu'.
        use_fp16 = False
        if self.is_exporting:
            use_fp16 = io.input_bool ("Export quantized?", False, help_message='Makes the exported model faster. If you have problems, disable this option.')

        # Scelta di porting, la stessa applicata in
        # models/Model_SAEHD/Model.py:807-809:
        # in TF `models_opt_device` era uno scope per le sole *variabili*
        # (Model.py:258, :281) mentre le op del merge stavano su
        # nn.tf_default_device_name (:519); in torch un modulo vive su un solo
        # device, quindi la scelta e' una sola e si segue il device delle op.
        # `optimizer_vars_on_cpu` continua a governare vars_on_cpu degli
        # accumulatori, che dal device del modulo e' indipendente davvero.
        # Fuori dal training `not self.is_training` corto-circuita: la
        # conseguenza da sapere e' che il merge tiene i pesi in VRAM invece che
        # in RAM, e chi non ce li fa stare ha `--cpu-only`
        # (mainscripts/Merger.py:53), che era l'unica via anche col TF.
        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = nn.device if models_opt_on_gpu or not self.is_training else torch.device('cpu')
        optimizer_vars_on_cpu = models_opt_device.type == 'cpu'

        self.model_filename_list = []

        # Niente placeholder (Model.py:265-278): i tensori si creano dai numpy
        # dentro le funzioni di training qui sotto. bgr_shape e mask_shape
        # servivano solo a dichiararli e spariscono con loro; morph_value_t
        # diventa il parametro `morph_value` di amp_view/amp_merge.

        # Initializing model classes
        self.encoder   = AMPEncoder(input_ch, e_dims, ae_dims, resolution, use_fp16, name='encoder')
        self.inter_src = AMPInter(ae_dims, inter_res, inter_dims, name='inter_src')
        self.inter_dst = AMPInter(ae_dims, inter_res, inter_dims, name='inter_dst')
        self.decoder   = AMPDecoder(inter_dims, d_dims, d_mask_dims, use_fp16, name='decoder')

        # I nomi sono quelli che amp_flow, amp_view e amp_merge indicizzano, e
        # sono anche i prefissi delle chiavi su disco: i due Inter sono
        # strutturalmente identici, hanno lo stesso disk_key e senza il nome
        # della rete davanti collidono.
        self.nets = {'encoder': self.encoder, 'inter_src': self.inter_src,
                     'inter_dst': self.inter_dst, 'decoder': self.decoder}

        self.model_filename_list += [   [self.encoder,  'encoder.npy'],
                                        [self.inter_src, 'inter_src.npy'],
                                        [self.inter_dst , 'inter_dst.npy'],
                                        [self.decoder , 'decoder.npy'] ]

        if self.is_training and gan_power != 0:
            self.GAN = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], name="GAN")
            self.nets['GAN'] = self.GAN

        # build() e .to() prima di leggerne i pesi, come
        # models/Model_SAEHD/Model.py:879-881: get_weights() li enumera e
        # initialize_variables alloca gli accumulatori sul device del parametro
        # quando vars_on_cpu e' False.
        #
        # Il ciclo gira su self.nets e non su model_filename_list -- come invece
        # fa SAEHD -- perche' quella lista segue l'ordine del TF,
        # che infila src_dst_opt fra decoder.npy e GAN.npy (Model.py:305 e
        # :311-312): quando la GAN vi entra, l'ottimizzatore che la precede e'
        # gia' costruito e non ha ne' build() ne' to().
        for net in self.nets.values():
            net.build()
            net.to(models_opt_device)

        def to_t(x):
            return torch.as_tensor(np.ascontiguousarray(x)).to(models_opt_device, nn.floatx)

        if self.is_training:
            # Initialize optimizers
            clipnorm = 1.0 if self.options['clipgrad'] else 0.0
            if self.options['lr_dropout'] in ['y','cpu']:
                lr_cos = 500
                lr_dropout = 0.3
            else:
                lr_cos = 0
                lr_dropout = 1.0

            # amp_g_weights, non una lista riscritta qui: i due Inter restano
            # fuori dall'insieme allenato (Model.py:301), che e' un difetto
            # dell'originale riprodotto per contratto -- si veda il docstring di
            # amp_g_weights.
            self.G_weights = amp_g_weights(self.nets)
            # initialize_variables vuole le quadruple (nome, parametro, owner,
            # path) di optimizer_weights(), non i Parameter nudi che
            # nn.gradients/opt.step usano (core/leras/optimizers/AdaBelief.py:85-104).
            # Le due liste devono coprire le stesse reti: lo pinna
            # test_the_optimizers_are_adabelief_with_the_hardcoded_lr.
            self.G_saveable_weights = self.encoder.optimizer_weights() + self.decoder.optimizer_weights()

            # lr_dropout_on_cpu non si passa: il TF di AMP non lo passava
            # (Model.py:304), a differenza di SAEHD (:341 del suo).
            self.src_dst_opt = nn.AdaBelief(lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_opt.initialize_variables (self.G_saveable_weights, vars_on_cpu=optimizer_vars_on_cpu)
            self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

            if gan_power != 0:
                self.GAN_opt = nn.AdaBelief(lr=5e-5, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                self.GAN_opt.initialize_variables ( self.GAN.optimizer_weights(), vars_on_cpu=optimizer_vars_on_cpu)
                self.model_filename_list += [ [self.GAN, 'GAN.npy'],
                                              [self.GAN_opt, 'GAN_opt.npy'] ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            #
            # gpu_count resta nel calcolo del batch e nella divisione del
            # gradiente, ma il percorso e' a device singolo: il loop `for gpu_id
            # in range(gpu_count)` del TF (Model.py:339), che tagliava il batch e
            # mediava i gradienti per GPU, non e' stato portato. Il multi-GPU ha
            # una fase dedicata dopo i modelli; fino ad allora con
            # piu' di una GPU si allena sulla prima sola, con l'intero batch.
            # Stessa decisione di models/Model_SAEHD/Model.py:915-926.
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            train_cfg = {'resolution'    : resolution,
                         'inter_dims'    : inter_dims,
                         'morph_factor'  : morph_factor,
                         'blur_out_mask' : blur_out_mask,
                         'gan_power'     : gan_power}

            # Initializing training and view functions
            def train(warped_src, target_src, target_srcm, target_srcm_em,  \
                              warped_dst, target_dst, target_dstm, target_dstm_em, ):
                s, d = amp_train_step(self.nets, self.src_dst_opt, self.G_weights,
                                      [to_t(x) for x in (warped_src, target_src, target_srcm, target_srcm_em,
                                                         warped_dst, target_dst, target_dstm, target_dstm_em)],
                                      train_cfg, gpu_count)
                return s.detach().cpu().numpy(), d.detach().cpu().numpy()
            self.train = train

            if gan_power != 0:
                def GAN_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                              warped_dst, target_dst, target_dstm, target_dstm_em, ):
                    amp_gan_train_step(self.nets, self.GAN, self.GAN_opt,
                                       [to_t(x) for x in (warped_src, target_src, target_srcm, target_srcm_em,
                                                          warped_dst, target_dst, target_dstm, target_dstm_em)],
                                       train_cfg, gpu_count)
                self.GAN_train = GAN_train

            def AE_view(warped_src, warped_dst, morph_value):
                return [x.detach().cpu().numpy() for x in
                        amp_view(self.nets, to_t(warped_src), to_t(warped_dst), morph_value, train_cfg)]

            self.AE_view = AE_view
        else:
            #Initializing merge function
            def AE_merge(warped_dst, morph_value):
                return [x.detach().cpu().numpy() for x in
                        amp_merge(self.nets, to_t(warped_dst), morph_value)]

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = self.is_first_run()
            if self.is_training and gan_power != 0 and model == self.GAN:
                if self.gan_model_changed:
                    do_init = True
            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )
            if do_init:
                model.init_weights()
        ###############


        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path #if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path #if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None else None #and not self.pretrain

            cpu_count = multiprocessing.cpu_count()
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)



            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                         'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                                         'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,  'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'],# or self.pretrain,
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                             'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                             'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE,  'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'],# or self.pretrain,
                        generators_count=dst_generators_count )
                             ])

    def export_dfm (self):
        # self.nets contiene GAN solo se self.is_training e gan_power != 0
        # (Model.py:874-877): export_dfm gira con is_exporting=True e
        # is_training=False (ModelBase.__init__), quindi qui dentro non c'e'
        # mai -- ModuleDict non la porta dentro come inizializzatore morto.
        output_path = self.get_strpath_storage_for_file('model.dfm')
        io.log_info(f'Dumping .dfm to {output_path}')

        out_names = ['out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0']

        # morph_value non ha asse dinamico: e' un valore di forma (1,), come il
        # placeholder TF. Il grafo si traccia a 0.5 e resta dinamico -- vedi
        # AMPExportModule.
        #
        # dynamo=False: l'esportatore nuovo e' il default in torch 2.13 e
        # ignora opset_version, scrivendo 18 dove il contratto vuole 12.
        torch.onnx.export(
            AMPExportModule(self.nets, self.inter_dims).eval(),
            (torch.zeros(1, self.resolution, self.resolution, 3,
                         device=nn.device, dtype=nn.floatx),
             torch.full((1,), 0.5, device=nn.device, dtype=nn.floatx)),
            output_path,
            input_names  = ['in_face:0', 'morph_value:0'],
            output_names = out_names,
            opset_version = 12,
            dynamic_axes = {n: {0: 'batch'} for n in ['in_face:0'] + out_names},
            dynamo = False)

        # L'inferenza di forma di torch.onnx.export perde altezza e larghezza
        # attraversando la permute finale di AMPExportModule.forward
        # (NCHW->NHWC): il proto dichiarava altezza e larghezza simboliche
        # (una etichettata col simbolo del batch) invece di [batch,96,96,1]
        # (stesso difetto misurato su SAEHD, task-9-brief.md, e stessa causa:
        # la permute finale). tf2onnx (riferimento T6) le dichiara concrete;
        # qui si rilegge il proto appena scritto e si riscrivono altezza,
        # larghezza e canali (assi 1, 2 e 3) sulle sole uscite, lasciando
        # l'asse 0 (batch) simbolico -- e' un difetto nei metadati
        # dichiarati, non nel calcolo: a run time il grafo era gia' corretto.
        # morph_value:0 e' un ingresso, non un'uscita: questo giro non lo
        # tocca, e la sua forma (1,) resta quella scritta da dynamic_axes qui
        # sopra.
        #
        # I canali (asse 3) si scrivono espliciti, non solo altezza/larghezza
        # come basterebbe in SAEHD: qui out_celeb_face:0 passa per il
        # torch.cat sullo slice dipendente da morph_value (AMPExportModule,
        # k = inter_dims*morph_value[0]) e l'inferenza di forma di torch perde
        # ANCHE l'asse dei canali su quella sola uscita, dichiarandolo
        # 'Transposeout_celeb_face:0_dim_3' invece di 3 -- misurato: le altre
        # due uscite (out_face_mask:0, out_celeb_face_mask:0), che non
        # passano dal cat dinamico, hanno gia' l'asse 3 concreto come in
        # SAEHD. Scrivere sempre tutti e tre gli assi evita di dover sapere
        # caso per caso quale asse l'inferenza di torch abbia deciso di
        # perdere.
        out_channels = [1, 3, 1]  # out_face_mask, out_celeb_face, out_celeb_face_mask
        proto = onnx.load(output_path)
        for o, ch in zip(proto.graph.output, out_channels):
            dims = o.type.tensor_type.shape.dim
            dims[1].dim_value = self.resolution
            dims[2].dim_value = self.resolution
            dims[3].dim_value = ch
        onnx.save(proto, output_path)

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def should_save_preview_history(self):
        return (not io.is_colab() and self.iter % ( 10*(max(1,self.resolution // 64)) ) == 0) or \
               (io.is_colab() and self.iter % 100 == 0)

    #override
    def onTrainOneIter(self):
        # Model.py:647 nella versione TF aveva qui `bs = self.get_batch_size()`:
        # una locale assegnata e mai letta in tutto il metodo (verificato con
        # `git show 84cc0f4:models/Model_AMP/Model.py | sed -n '646,657p' | grep
        # -n "\bbs\b"` -> solo la riga dell'assegnamento). Non trascritta.
        ( (warped_src, target_src, target_srcm, target_srcm_em), \
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = self.generate_next_samples()

        src_loss, dst_loss = self.train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        if self.gan_power != 0:
            self.GAN_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        ( (warped_src, target_src, target_srcm, target_srcm_em),
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = samples

        S, D, SS, DD, DDM_000, _, _ = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst, 0.0)  ) ]

        _, _, DDM_025, SD_025, SDM_025 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.25) ]
        _, _, DDM_050, SD_050, SDM_050 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.50) ]
        _, _, DDM_065, SD_065, SDM_065 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.65) ]
        _, _, DDM_075, SD_075, SDM_075 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 0.75) ]
        _, _, DDM_100, SD_100, SDM_100 = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in self.AE_view (target_src, target_dst, 1.00) ]

        (DDM_000,
         DDM_025, SDM_025,
         DDM_050, SDM_050,
         DDM_065, SDM_065,
         DDM_075, SDM_075,
         DDM_100, SDM_100) = [ np.repeat (x, (3,), -1) for x in (DDM_000,
                                                                 DDM_025, SDM_025,
                                                                 DDM_050, SDM_050,
                                                                 DDM_065, SDM_065,
                                                                 DDM_075, SDM_075,
                                                                 DDM_100, SDM_100) ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        result = []

        i = np.random.randint(n_samples) if not for_history else 0

        st =  [ np.concatenate ((S[i],  D[i],  DD[i]*DDM_000[i]), axis=1) ]
        st += [ np.concatenate ((SS[i], DD[i], SD_100[i] ), axis=1) ]

        result += [ ('AMP morph 1.0', np.concatenate (st, axis=0 )), ]

        st =  [ np.concatenate ((DD[i], SD_025[i],  SD_050[i]), axis=1) ]
        st += [ np.concatenate ((SD_065[i], SD_075[i], SD_100[i]), axis=1) ]
        result += [ ('AMP morph list', np.concatenate (st, axis=0 )), ]

        st =  [ np.concatenate ((DD[i], SD_025[i]*DDM_025[i]*SDM_025[i],  SD_050[i]*DDM_050[i]*SDM_050[i]), axis=1) ]
        st += [ np.concatenate ((SD_065[i]*DDM_065[i]*SDM_065[i], SD_075[i]*DDM_075[i]*SDM_075[i], SD_100[i]*DDM_100[i]*SDM_100[i]), axis=1) ]
        result += [ ('AMP morph list masked', np.concatenate (st, axis=0 )), ]

        return result

    #override
    def get_preview_layout(self):
        #Le righe qui non sono campioni: AMP mostra un solo campione e usa
        #le due righe per viste diverse dello stesso volto.
        def griglia(sopra, sotto, risultato):
            return { "righe": 2, "colonne": 3,
                     "celle": [ list(sopra), list(sotto) ],
                     "risultato": list(risultato),
                     "righe_sono_campioni": False }

        return { 'AMP morph 1.0':
                     griglia(['src', 'dst', 'dst->dst'],
                             ['src->src', 'dst->dst', 'morph 1.0'], (1, 2)),
                 'AMP morph list':
                     griglia(['dst->dst', 'morph 0.25', 'morph 0.50'],
                             ['morph 0.65', 'morph 0.75', 'morph 1.0'], (1, 2)),
                 'AMP morph list masked':
                     griglia(['dst->dst', 'morph 0.25', 'morph 0.50'],
                             ['morph 0.65', 'morph 0.75', 'morph 1.0'], (1, 2)) }

    def predictor_func (self, face, morph_value):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face, morph_value) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        morph_factor = np.clip ( io.input_number ("Morph factor", 1.0, add_info="0.0 .. 1.0"), 0.0, 1.0 )

        def predictor_morph(face):
            return self.predictor_func(face, morph_factor)


        import merger
        return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = AMPModel
