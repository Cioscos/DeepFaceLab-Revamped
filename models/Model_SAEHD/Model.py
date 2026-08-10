import multiprocessing
import operator
from functools import partial

import numpy as np
import onnx
import torch

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *


def saehd_src_code(nets, archi_type, warped_src):
    """
    Il codice src (Model.py:407 in df, :415-417 in liae nella versione TF).

    E' il tensore che porta quel nome *alla fine* del ramo, come nel TF: in
    `liae` il nome viene riassegnato alla concatenazione, ed e' quella -- non
    l'uscita di inter_AB -- che esce di qui.

    Sta a parte da saehd_flow perche' i due passi dei discriminatori leggono
    solo una fetta del forward, e nel grafo TF il resto non veniva mai
    valutato: vedi saehd_D_code_train_step e saehd_D_src_dst_train_step.

    Quanto vale (res 64, batch 2, CPU) -- conteggio ottenuto avvolgendo
    `net.forward` di ogni rete, tempo su 20 iterazioni dopo 3 di scaldata:

      passo                 forward prima              forward dopo
      true_face   enc 2 int 2 dec_src 3 dec_dst 1   enc 2 int 2
      GAN         enc 2 int 2 dec_src 3 dec_dst 1   enc 1 int 1 dec_src 1

      passo         prima        dopo
      true_face   80.8 ms/it   8.0 ms/it   (10.1x)
      GAN         87.2 ms/it  30.3 ms/it   ( 2.9x)

    Il risultato e' identico **bit per bit**: gli stessi due passi con il
    forward preso da saehd_flow danno `max|loss prima - dopo| = 0.0` e
    `max|grad prima - dopo| = 0.0` su entrambi i casi. Quei tensori non
    entravano in nessuna loss -- e' esattamente per questo che si potevano non
    calcolare, ed e' il motivo per cui i valori prodotti non si sono mossi.
    """
    if 'df' in archi_type:
        return nets['inter'](nets['encoder'](warped_src))
    elif 'liae' in archi_type:
        src_inter_AB_code = nets['inter_AB'](nets['encoder'](warped_src))
        return torch.cat([src_inter_AB_code,src_inter_AB_code], nn.conv2d_ch_axis)
    else:
        raise ValueError(f"archi_type sconosciuto: {archi_type}")


def saehd_dst_codes(nets, archi_type, warped_dst):
    """
    Il codice dst e il codice src->dst (Model.py:408 in df, :418-422 in liae).

    In `df` sono lo stesso tensore: e' decoder_src applicato al codice dst a
    produrre pred_src_dst (:411). In `liae` il codice src->dst e' inter_AB(dst)
    concatenato con se' stesso -- e' cosi' che liae fa uscire una src dal
    decoder unico, mettendo al posto della meta' "identita' dst" (inter_B) la
    stessa meta' condivisa.
    """
    if 'df' in archi_type:
        dst_code = nets['inter'](nets['encoder'](warped_dst))
        return dst_code, dst_code
    elif 'liae' in archi_type:
        dst_code          = nets['encoder'](warped_dst)
        dst_inter_B_code  = nets['inter_B'](dst_code)
        dst_inter_AB_code = nets['inter_AB'](dst_code)
        return ( torch.cat([dst_inter_B_code,dst_inter_AB_code], nn.conv2d_ch_axis),
                 torch.cat([dst_inter_AB_code,dst_inter_AB_code], nn.conv2d_ch_axis) )
    else:
        raise ValueError(f"archi_type sconosciuto: {archi_type}")


def saehd_decoders(nets, archi_type):
    """
    Il decoder del ramo src e quello del ramo dst: due reti in `df`
    (Model.py:409-412), la stessa rete due volte in `liae` (:424-427).
    """
    if 'df' in archi_type:
        return nets['decoder_src'], nets['decoder_dst']
    elif 'liae' in archi_type:
        return nets['decoder'], nets['decoder']
    else:
        raise ValueError(f"archi_type sconosciuto: {archi_type}")


def saehd_flow(nets, archi_type, warped_src, warped_dst):
    """
    Il forward delle reti: i due codici, le sei predizioni e la predizione su
    un codice staccato dal grafo (Model.py:406-427 nella versione TF).

    Le quattro chiamate ai decoder sono comuni alle due topologie perche' le
    due differenze -- quale codice riceve il ramo src->dst e quale rete lo
    decodifica -- stanno gia' dentro saehd_dst_codes e saehd_decoders. In df
    `src_dst_code` **e'** `dst_code`, quindi `decoder_src(src_dst_code)` e' la
    :411 alla lettera; in liae e' la concatenazione doppia della :422 e il
    decoder e' quello unico. Le op eseguite, e il loro ordine, sono le stesse
    di prima: il ritorno e' bit per bit quello della versione TF.

    Il solo consumatore dei due codici e' il code discriminator del
    true_face_power, che pero' esiste solo in `df`: l'opzione e' forzata a 0.0
    per ogni altra archi (Model.py:167-170 nella versione TF) e la rete e'
    costruita dentro il ramo `df` (:295). In `liae` i due valori li scarta gia'
    oggi saehd_train_step; restano nel ritorno perche' la firma e' una sola.

    A livello di modulo, come xseg_loss/xseg_train_step e per la stessa
    ragione: una chiusura dentro on_initialize non e' raggiungibile
    dall'esterno, perche' costruire un SAEHDModel significa passare da
    ModelBase, che prompta l'utente da console.
    """
    src_code               = saehd_src_code (nets, archi_type, warped_src)
    dst_code, src_dst_code = saehd_dst_codes(nets, archi_type, warped_dst)
    decoder_src, decoder_dst = saehd_decoders(nets, archi_type)

    pred_src_src, pred_src_srcm = decoder_src(src_code)
    pred_dst_dst, pred_dst_dstm = decoder_dst(dst_code)
    pred_src_dst, pred_src_dstm = decoder_src(src_dst_code)
    # .detach() e' il tf.stop_gradient della 412 (df) e della 427 (liae). Lo
    # consuma solo il termine face_style, quindi in `df` nessun confronto
    # numerico lo esercita: il contratto e' il .detach() stesso.
    pred_src_dst_no_code_grad, _ = decoder_src(src_dst_code.detach())

    return (src_code, dst_code, pred_src_src, pred_src_srcm,
            pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm,
            pred_src_dst_no_code_grad)


def saehd_src_dst_weights(nets, archi_type, random_warp):
    """
    Le due liste di pesi del passo src_dst: (saveable, trainable).

    Non sono piu' lo stesso oggetto come nel TF (Model.py:332 e :336):
    initialize_variables vuole le quadruple (name, param, owner, param_path) di
    optimizer_weights() -- una tf.Variable portava il proprio `.name` addosso,
    un Parameter torch no -- mentre nn.gradients e opt.step vogliono i
    Parameter nudi.

    Divergono nel *contenuto* solo per liae con random_warp=False
    (Model.py:335-338): l'ottimizzatore resta inizializzato su tutti i pesi --
    gli accumulatori di inter_AB continuano a esistere e ad andare su disco --
    ma i gradienti si calcolano senza inter_AB. E' l'unica configurazione in
    cui la lista dei pesi ottimizzati e quella dei pesi che ricevono gradiente
    non coincidono: chi tocca questa funzione lo tenga presente.

    A livello di modulo per la stessa ragione di saehd_flow: dentro
    on_initialize non e' raggiungibile da nessun test.
    """
    if 'df' in archi_type:
        saveable_order = trainable_order = ('encoder', 'inter', 'decoder_src', 'decoder_dst')
    elif 'liae' in archi_type:
        saveable_order  = ('encoder', 'inter_AB', 'inter_B', 'decoder')
        trainable_order = saveable_order if random_warp else ('encoder', 'inter_B', 'decoder')
    else:
        raise ValueError(f"archi_type sconosciuto: {archi_type}")

    return ( [w for name in saveable_order  for w in nets[name].optimizer_weights()],
             [w for name in trainable_order for w in nets[name].get_weights()] )


def saehd_dloss(labels, logits):
    """La DLoss del modello (Model.py:499-500 nella versione TF): un valore per campione."""
    return torch.mean( nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), dim=[1,2,3])


def saehd_mask_blur(targetm, resolution):
    """La maschera sfumata e riportata a [0,1] (Model.py:437-438 e :441-442)."""
    m = nn.gaussian_blur(targetm,  max(1, resolution // 32) )
    return torch.clamp(m, 0, 0.5) * 2


def saehd_masked_opt(target, pred, mask_blur, masked_training):
    """
    Gli operandi di un lato della loss quando masked_training e' acceso
    (Model.py:453-456).

    Sta in una funzione, e non ripetuto sui tre siti, perche' la coppia src la
    rilegge anche il discriminatore GAN (Model.py:517-518 e :526-527): e' l'unico punto
    dove gan_power e masked_training si toccano davvero: accendendo un flag
    per volta la loro interazione non si vede.
    """
    return (target*mask_blur, pred*mask_blur) if masked_training else (target, pred)


def saehd_blur_out_mask(target, targetm, resolution):
    """Un lato di blur_out_mask (Model.py:391-402), src o dst."""
    sigma = resolution / 128

    targetm_anti = 1-targetm

    x = nn.gaussian_blur(target*targetm_anti, sigma)
    y = 1-nn.gaussian_blur(targetm, sigma)
    y = torch.where(y == 0, torch.ones_like(y), y)
    return target*targetm + (x/y)*targetm_anti


def saehd_src_dst_loss(target_src, target_srcm, target_srcm_em, pred_src_src, pred_src_srcm,
                       target_dst, target_dstm, target_dstm_em, pred_dst_dst, pred_dst_dstm,
                       pred_src_dst, pred_src_dstm, pred_src_dst_no_code_grad,
                       resolution, masked_training, eyes_mouth_prio,
                       face_style_power, bg_style_power, pretrain):
    """
    La src-loss e la dst-loss, un valore per campione: forma (N,) ciascuna, piu'
    la maschera src sfumata, che il ramo GAN di saehd_train_step rilegge.

    Trascrizione di Model.py:437-492, termini di stile compresi. Stanno qui e
    non sommati fuori sul valore di ritorno -- che darebbe lo stesso float bit
    per bit, perche' src_loss e dst_loss sono accumulatori distinti -- perche'
    e' qui che stanno nel TF, fra l'ultimo termine della src-loss (:468) e il
    primo della dst-loss (:482), ed e' cosi' che questa funzione resta
    confrontabile riga per riga con l'originale.

    CONVENZIONE D'INGRESSO, e diverge da quella della gemella di AMP.
    `blur_out_mask` **non** e' un parametro di questa funzione: quando l'opzione
    e' accesa e' il chiamante ad applicarlo, e i target che arrivano qui sono
    gia' passati per saehd_blur_out_mask (saehd_train_step, Model.py:336-338, e
    saehd_gan_train_step, :440-441).
    In models/Model_AMP/Model.py la stessa riga del TF sta invece **dentro**
    amp_loss (Model.py:475-477 di quel file), che ha il parametro
    `blur_out_mask` e riceve i target **grezzi**.

    Nel TF le due erano nello stesso posto -- entrambe inline nello stesso
    blocco (Model.py:391-402 di SAEHD e :393-404 di AMP nella versione TF) --
    quindi la divergenza e' del porting, non degli originali. Non e' un
    difetto: le due funzioni sono numericamente giuste ciascuna con la propria
    convenzione. E' una trappola per
    chi copia una chiamata dall'una all'altra: passare qui target grezzi con
    l'opzione accesa, o passare ad amp_loss target gia' sfocati, non solleva
    niente e da' numeri sbagliati in silenzio. Prima di spostare una delle due
    si tenga presente che sarebbe un cambio di comportamento su modelli gia'
    validati e gia' passati allo smoke.
    """
    target_srcm_blur = saehd_mask_blur(target_srcm, resolution)
    target_dstm_blur = saehd_mask_blur(target_dstm, resolution)

    # Model.py:444-445 trascritte come stanno, non come erano intese. La 444
    # calcola la maschera da pred_src_dstm*pred_dst_dstm; la 445 la
    # *sovrascrive* con un clip di target_srcm_blur senza averla mai usata.
    # Due conseguenze, entrambe deliberate qui: la maschera di bg_style_power e'
    # quella di **src** e non il prodotto delle due predette, e il clip e' un
    # no-op, perche' saehd_mask_blur restituisce gia' valori in [0,1]. E' un
    # difetto dell'originale e resta tale: il vincolo di questa migrazione e'
    # che il comportamento non cambi, non che migliori. Lo fissa
    # test_bg_style_reads_the_src_mask.
    #
    # La riga 444 ha pero' un costo che nel TF non aveva: il suo nodo non aveva
    # consumatori e il grafo pigro non lo valutava **mai**, mentre torch in
    # eager la esegue a ogni iterazione, di ogni modello SAEHD, anche con i due
    # style power a zero. E' una convoluzione depthwise a un canale con
    # kernel_size = max(3, 4*(resolution//32)) dispari:
    #
    #   res=  64  radius= 2  kernel= 9x9    0.001 GMAC     ~0.2 ms
    #   res= 224  radius= 7  kernel=29x29   0.169 GMAC     ~4.5 ms
    #   res= 512  radius=16  kernel=65x65   4.430 GMAC   ~144   ms
    #
    # (batch 4, CPU; i GMAC sono esatti, i millisecondi indicativi di questa
    # macchina.)
    # A 512 sono dell'ordine di un intero passaggio del decoder buttato via a
    # ogni iterazione. Chi porta il merge o tocca le prestazioni lo sappia: la
    # riga resta perche' il comportamento deve restare identico, non perche'
    # costi poco.
    style_mask_blur = nn.gaussian_blur(pred_src_dstm*pred_dst_dstm,  max(1, resolution // 32) )
    style_mask_blur = torch.clamp(target_srcm_blur, 0, 1.0).detach()
    style_mask_anti_blur = 1.0 - style_mask_blur

    target_src_masked_opt, pred_src_src_masked_opt = saehd_masked_opt(target_src, pred_src_src, target_srcm_blur, masked_training)
    target_dst_masked_opt, pred_dst_dst_masked_opt = saehd_masked_opt(target_dst, pred_dst_dst, target_dstm_blur, masked_training)

    if resolution < 256:
        src_loss =  torch.mean ( 10*nn.dssim(target_src_masked_opt, pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), dim=[1])
    else:
        src_loss =  torch.mean ( 5*nn.dssim(target_src_masked_opt, pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), dim=[1])
        src_loss += torch.mean ( 5*nn.dssim(target_src_masked_opt, pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/23.2)), dim=[1])
    src_loss += torch.mean ( 10*torch.square ( target_src_masked_opt - pred_src_src_masked_opt ), dim=[1,2,3])

    if eyes_mouth_prio:
        src_loss += torch.mean ( 300*torch.abs ( target_src*target_srcm_em - pred_src_src*target_srcm_em ), dim=[1,2,3])

    src_loss += torch.mean ( 10*torch.square( target_srcm - pred_src_srcm ),dim=[1,2,3] )

    face_style_power = face_style_power / 100.0
    if face_style_power != 0 and not pretrain:
        src_loss += nn.style_loss(pred_src_dst_no_code_grad*pred_src_dstm.detach(), (pred_dst_dst*pred_dst_dstm).detach(), gaussian_blur_radius=resolution//8, loss_weight=10000*face_style_power)

    bg_style_power = bg_style_power / 100.0
    if bg_style_power != 0 and not pretrain:
        target_dst_style_anti_masked = target_dst*style_mask_anti_blur
        psd_style_anti_masked = pred_src_dst*style_mask_anti_blur

        src_loss += torch.mean( (10*bg_style_power)*nn.dssim( psd_style_anti_masked,  target_dst_style_anti_masked, max_val=1.0, filter_size=int(resolution/11.6)), dim=[1])
        src_loss += torch.mean( (10*bg_style_power)*torch.square(psd_style_anti_masked - target_dst_style_anti_masked), dim=[1,2,3] )

    if resolution < 256:
        dst_loss = torch.mean ( 10*nn.dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), dim=[1])
    else:
        dst_loss = torch.mean ( 5*nn.dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), dim=[1])
        dst_loss += torch.mean ( 5*nn.dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/23.2) ), dim=[1])
    dst_loss += torch.mean ( 10*torch.square(  target_dst_masked_opt- pred_dst_dst_masked_opt ), dim=[1,2,3])

    if eyes_mouth_prio:
        dst_loss += torch.mean ( 300*torch.abs ( target_dst*target_dstm_em - pred_dst_dst*target_dstm_em ), dim=[1,2,3])

    dst_loss += torch.mean ( 10*torch.square( target_dstm - pred_dst_dstm ),dim=[1,2,3] )

    return src_loss, dst_loss, target_srcm_blur


def saehd_train_step(nets, opt, trainable_weights, batch, cfg, gpu_count):
    """
    Il primo dei tre passi di onTrainOneIter: forward, loss, gradienti, update
    delle reti src_dst. Ritorna (src_loss, dst_loss), le due loss per campione
    *senza* i termini che vivono solo nel gradiente (Model.py:497 e :516-545).

    `G_loss.sum() / gpu_count` e' il gradiente del TF, non una riformulazione:
    tf.gradients di una loss vettoriale ne somma le componenti (Model.py:497 e
    :547) e nn.average_gv_list divideva poi per il numero di GPU (Model.py:564).
    torch.autograd.grad vuole uno scalare, e questo e' lo scalare che da' lo
    stesso gradiente.

    I due discriminatori entrano qui solo in *avanti*: i gradienti si prendono
    sui soli trainable_weights, e i pesi di `code_discriminator` e `D_src` li
    aggiornano i due passi seguenti, ciascuno col proprio ottimizzatore.
    """
    warped_src, target_src, target_srcm, target_srcm_em, \
    warped_dst, target_dst, target_dstm, target_dstm_em = batch

    resolution = cfg['resolution']

    if cfg['blur_out_mask']:
        target_src = saehd_blur_out_mask(target_src, target_srcm, resolution)
        target_dst = saehd_blur_out_mask(target_dst, target_dstm, resolution)

    src_code, dst_code, pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, \
    pred_src_dst, pred_src_dstm, pred_src_dst_no_code_grad = \
        saehd_flow(nets, cfg['archi_type'], warped_src, warped_dst)

    src_loss, dst_loss, target_srcm_blur = saehd_src_dst_loss(
        target_src, target_srcm, target_srcm_em, pred_src_src, pred_src_srcm,
        target_dst, target_dstm, target_dstm_em, pred_dst_dst, pred_dst_dstm,
        pred_src_dst, pred_src_dstm, pred_src_dst_no_code_grad,
        resolution, cfg['masked_training'], cfg['eyes_mouth_prio'],
        cfg['face_style_power'], cfg['bg_style_power'], cfg['pretrain'])

    G_loss = src_loss + dst_loss

    if cfg['true_face_power'] != 0:                                   # Model.py:502-508
        src_code_d = nets['code_discriminator']( src_code )

        G_loss += cfg['true_face_power']*saehd_dloss(torch.ones_like(src_code_d), src_code_d)

    if cfg['gan_power'] != 0:                                         # Model.py:516-545
        _, pred_src_src_masked_opt = saehd_masked_opt(target_src, pred_src_src, target_srcm_blur, cfg['masked_training'])

        pred_src_src_d            = nets['D_src'](pred_src_src_masked_opt)

        G_loss += cfg['gan_power']*(saehd_dloss(torch.ones_like(pred_src_src_d.center), pred_src_src_d.center)  + \
                                    saehd_dloss(torch.ones_like(pred_src_src_d.out), pred_src_src_d.out))

        if cfg['masked_training']:
            # Minimal src-src-bg rec with total_variation_mse to suppress random bright dots from gan
            target_srcm_anti_blur = 1.0-target_srcm_blur
            G_loss += 0.000001*nn.total_variation_mse(pred_src_src)
            G_loss += 0.02*torch.mean(torch.square(pred_src_src*target_srcm_anti_blur-target_src*target_srcm_anti_blur),dim=[1,2,3] )

    opt.step( nn.gradients (G_loss.sum() / gpu_count, trainable_weights) )
    return src_loss, dst_loss


def saehd_D_code_train_step(nets, opt, batch, cfg, gpu_count):
    """
    Il secondo passo: il code discriminator di true_face_power (Model.py:502-514
    e :589-592). Ritorna la D_code_loss, un valore per campione.

    Prende i soli due warped, come `D_train` nel modello: la loss del code
    discriminator non tocca ne' i target ne' le maschere, e restringere la firma
    e' quello che tiene onesta questa affermazione. Gira *dopo* src_dst_train,
    quindi su encoder/inter gia' aggiornati: i due codici che discrimina sono
    quelli dei pesi nuovi, non quelli del passo 1.

    Con archi liae `nets` non ha nessun 'code_discriminator' e questa funzione
    solleva. E' il comportamento dell'originale, non un ripiego: la rete e'
    costruita dentro il ramo df (Model.py:295) mentre l'opzione e' forzata a 0
    per ogni altra archi (Model.py:167-170), quindi il ramo e' irraggiungibile
    passando dai prompt -- e un true_face_power != 0 scritto a mano nel
    model.dat fa gia' morire on_initialize con un AttributeError su
    self.code_discriminator, prima di arrivare qui.
    """
    warped_src, warped_dst = batch
    code_discriminator = nets['code_discriminator']

    # I soli due codici, non tutto saehd_flow: gpu_D_code_loss dipende
    # esclusivamente da gpu_src_code/gpu_dst_code (Model.py:503-514), quindi il
    # grafo pigro del TF non valutava **nessun** decoder per D_loss_gv_op
    # (:567). Passando da saehd_flow torch ne eseguirebbe quattro in eager, con
    # il loro grafo all'indietro, per poi buttarli via.
    src_code    = saehd_src_code (nets, cfg['archi_type'], warped_src)
    dst_code, _ = saehd_dst_codes(nets, cfg['archi_type'], warped_dst)

    src_code_d = code_discriminator( src_code )
    dst_code_d = code_discriminator( dst_code )

    D_code_loss = (saehd_dloss(torch.ones_like(dst_code_d) , dst_code_d) + \
                   saehd_dloss(torch.zeros_like(src_code_d), src_code_d) ) * 0.5

    opt.step( nn.gradients (D_code_loss.sum() / gpu_count, code_discriminator.get_weights()) )
    return D_code_loss


def saehd_D_src_dst_train_step(nets, opt, batch, cfg, gpu_count):
    """
    Il terzo passo: il discriminatore GAN (Model.py:516-537 e :595-605).
    Ritorna la D_src_dst_loss, un valore per campione.

    Riceve tutti e otto gli ingressi come `D_src_dst_train` nel modello, ma ne
    legge **tre**: warped_src per il forward, target_src e target_srcm per
    l'esempio "vero". Anche questo gira sui pesi src_dst gia' aggiornati dal
    passo 1, ed e' da li' che viene il pred_src_src che discrimina.

    warped_dst resta nella firma perche' la chiusura D_src_dst_train inoltra
    tutti e otto gli ingressi come nel TF, dove pero' erano *placeholder*:
    gpu_D_src_dst_loss dipende dal solo ramo src (Model.py:517-537), quindi la
    sessione non valutava mai il ramo dst per src_D_src_dst_loss_gv_op (:570).
    """
    warped_src, target_src, target_srcm, target_srcm_em, \
    warped_dst, target_dst, target_dstm, target_dstm_em = batch

    resolution = cfg['resolution']
    D_src = nets['D_src']

    # Solo il lato src: nel TF il nodo gpu_target_dst_masked_opt non entra in
    # questa loss, e con blur_out_mask acceso la sessione non lo valuterebbe.
    if cfg['blur_out_mask']:
        target_src = saehd_blur_out_mask(target_src, target_srcm, resolution)

    # Il solo ramo src, non tutto saehd_flow: la 409 (`decoder(src_code)` in
    # liae, la 424) e nient'altro.
    decoder_src, _  = saehd_decoders(nets, cfg['archi_type'])
    pred_src_src, _ = decoder_src( saehd_src_code(nets, cfg['archi_type'], warped_src) )

    target_src_masked_opt, pred_src_src_masked_opt = saehd_masked_opt(
        target_src, pred_src_src, saehd_mask_blur(target_srcm, resolution), cfg['masked_training'])

    pred_src_src_d            = D_src(pred_src_src_masked_opt)

    target_src_d             = D_src(target_src_masked_opt)

    D_src_dst_loss = (saehd_dloss(torch.ones_like(target_src_d.center)     , target_src_d.center) + \
                      saehd_dloss(torch.zeros_like(pred_src_src_d.center)  , pred_src_src_d.center) ) * 0.5 + \
                     (saehd_dloss(torch.ones_like(target_src_d.out)     , target_src_d.out) + \
                      saehd_dloss(torch.zeros_like(pred_src_src_d.out)  , pred_src_src_d.out) ) * 0.5

    opt.step( nn.gradients (D_src_dst_loss.sum() / gpu_count, D_src.get_weights()) )
    return D_src_dst_loss


def saehd_view(nets, archi_type, warped_src, warped_dst):
    """
    Le cinque predizioni della preview, nel loro ordine (Model.py:608-612 nella
    versione TF): pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst,
    pred_src_dstm.

    L'ordine e' un contratto con onGetPreview, che le spacchetta posizionali
    (`S, D, SS, DD, DDM, SD, SDM = [...] + self.AE_view(...)`, Model.py:789
    nella versione TF): scambiare due valori dello stesso numero di canali non
    solleva niente e mostra un'altra faccia. Lo fissa
    test_view_returns_the_five_predictions_in_the_order_onGetPreview_unpacks.
    Lo stesso vale per i due *argomenti*, che la chiusura AE_view inoltra qui
    nell'ordine (src, dst): lo fissa
    test_onGetPreview_builds_its_two_mosaics_from_AE_view.

    `torch.no_grad()` e' quello che nel TF era gratis: `nn.tf_sess.run` non
    costruiva nessun grafo all'indietro. Senza, ogni preview lascerebbe in
    memoria il grafo di due forward completi.

    Passa da saehd_flow e non da una seconda trascrizione delle due topologie:
    e' lo stesso cablaggio del training. Costa in piu' un forward di
    decoder -- quello sul codice staccato (Model.py:412 e :427) -- e il ramo
    maschera di pred_src_srcm, che la sessione TF non valutava perche' AE_view
    non li chiedeva. La preview pero' non e' per iterazione: una all'avvio
    (Trainer.py:176-179), una a ogni salvataggio e, con write_preview_history
    acceso, una ogni 10*max(1, resolution//64) iterazioni (ModelBase.py:479 e
    should_save_preview_history qui sotto).
    """
    with torch.no_grad():
        (_, _, pred_src_src, _, pred_dst_dst, pred_dst_dstm,
         pred_src_dst, pred_src_dstm, _) = saehd_flow(nets, archi_type, warped_src, warped_dst)

    return [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm]


def saehd_merge(nets, archi_type, warped_dst):
    """
    Il grafo del merger: tre tensori (Model.py:615-635 nella versione TF), nel
    solo ordine che predictor_func accetta -- `bgr, mask_dst_dstm,
    mask_src_dstm`, cioe' pred_src_dst, pred_dst_dstm, pred_src_dstm.

    Prende il solo `warped_dst`: il merger ha in mano un frame dst e nient'altro,
    e non c'e' nessun ramo src da calcolare. E' anche il motivo per cui questa
    non e' una selezione da saehd_flow come saehd_view: passare due volte lo
    stesso frame a saehd_flow darebbe gli stessi tre tensori, ma pagando encoder,
    inter e decoder anche sul ramo src -- e qui si paga per **fotogramma**, non
    per preview. Che i due cablaggi non divergano lo fissa
    test_merge_is_the_same_wiring_as_the_flow, che li confronta valore per
    valore su entrambe le topologie.
    """
    with torch.no_grad():
        if 'df' in archi_type:
            dst_code = nets['inter'](nets['encoder'](warped_dst))
            pred_src_dst, pred_src_dstm = nets['decoder_src'](dst_code)
            _, pred_dst_dstm            = nets['decoder_dst'](dst_code)
        elif 'liae' in archi_type:
            dst_code          = nets['encoder'](warped_dst)
            dst_inter_B_code  = nets['inter_B'](dst_code)
            dst_inter_AB_code = nets['inter_AB'](dst_code)
            dst_code          = torch.cat([dst_inter_B_code,dst_inter_AB_code], nn.conv2d_ch_axis)
            src_dst_code      = torch.cat([dst_inter_AB_code,dst_inter_AB_code], nn.conv2d_ch_axis)

            pred_src_dst, pred_src_dstm = nets['decoder'](src_dst_code)
            _, pred_dst_dstm            = nets['decoder'](dst_code)
        else:
            raise ValueError(f"archi_type sconosciuto: {archi_type}")

    return [pred_src_dst, pred_dst_dstm, pred_src_dstm]


class SAEHDExportModule(torch.nn.Module):
    """
    Il sottografo che DeepFaceLive consuma (Model.py:708-734 nella versione TF).

    L'ordine delle uscite NON e' quello di saehd_merge: qui e' pred_dst_dstm,
    pred_src_dst, pred_src_dstm, cioe' out_face_mask, out_celeb_face,
    out_celeb_face_mask. Sono tensori della stessa forma e scambiarli non
    solleva nulla, quindi l'ordine e' pinnato da
    test_the_saehd_export_order_is_not_the_merge_order.

    Nessun torch.no_grad() qui, al contrario di saehd_merge: il tracciamento
    di torch.onnx.export gira gia' senza gradiente e un no_grad dentro il
    forward non cambia il grafo esportato -- il modulo pero' e' costruito
    .eval() dal chiamante, come il TF costruiva il grafo di export a parte.

    nn.set_data_format('NCHW') del TF (Model.py:706) non si traduce: nella
    porta il formato e' gia' e solo NCHW.

    self.nets = torch.nn.ModuleDict(nets): le reti di questo repo (encoder,
    inter, decoder...) derivano da Saveable, che e' gia' un torch.nn.Module
    (core/leras/layers/Saveable.py) -- a differenza di facelib.XSegNet.XSegNet
   , che era un wrapper plain-object attorno alla vera rete nn.XSeg.
    Qui ModuleDict registra le reti stesse, non un wrapper, e i loro
    Parameter risultano raggiungibili dal tracciato.
    """
    def __init__(self, nets, archi_type):
        super().__init__()
        self.nets       = torch.nn.ModuleDict(nets)
        self.archi_type = archi_type

    def forward(self, in_face):
        x = in_face.permute(0,3,1,2)

        if 'df' in self.archi_type:
            dst_code = self.nets['inter'](self.nets['encoder'](x))
            pred_src_dst, pred_src_dstm = self.nets['decoder_src'](dst_code)
            _,            pred_dst_dstm = self.nets['decoder_dst'](dst_code)
        elif 'liae' in self.archi_type:
            dst_code          = self.nets['encoder'](x)
            dst_inter_B_code  = self.nets['inter_B'](dst_code)
            dst_inter_AB_code = self.nets['inter_AB'](dst_code)
            dst_code          = torch.cat([dst_inter_B_code, dst_inter_AB_code], nn.conv2d_ch_axis)
            src_dst_code      = torch.cat([dst_inter_AB_code, dst_inter_AB_code], nn.conv2d_ch_axis)

            pred_src_dst, pred_src_dstm = self.nets['decoder'](src_dst_code)
            _,            pred_dst_dstm = self.nets['decoder'](dst_code)
        else:
            raise ValueError(f"archi_type sconosciuto: {self.archi_type}")

        return (pred_dst_dstm.permute(0,2,3,1),
                pred_src_dst.permute(0,2,3,1),
                pred_src_dstm.permute(0,2,3,1))


class SAEHDModel(ModelBase):

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()

        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb

        if lowest_vram >= 4:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        yn_str = {True:'y',False:'n'}
        min_res = 64
        max_res = 640

        #default_usefp16            = self.options['use_fp16']           = self.load_or_def_option('use_fp16', False)
        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 128)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'f')
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)

        default_archi              = self.options['archi']              = self.load_or_def_option('archi', 'liae-ud')

        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        default_masked_training    = self.options['masked_training']    = self.load_or_def_option('masked_training', True)
        default_eyes_mouth_prio    = self.options['eyes_mouth_prio']    = self.load_or_def_option('eyes_mouth_prio', False)
        default_uniform_yaw        = self.options['uniform_yaw']        = self.load_or_def_option('uniform_yaw', False)
        default_blur_out_mask      = self.options['blur_out_mask']      = self.load_or_def_option('blur_out_mask', False)

        default_adabelief          = self.options['adabelief']          = self.load_or_def_option('adabelief', True)

        lr_dropout = self.load_or_def_option('lr_dropout', 'n')
        lr_dropout = {True:'y', False:'n'}.get(lr_dropout, lr_dropout) #backward comp
        default_lr_dropout         = self.options['lr_dropout'] = lr_dropout

        default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        default_random_hsv_power   = self.options['random_hsv_power']   = self.load_or_def_option('random_hsv_power', 0.0)
        default_true_face_power    = self.options['true_face_power']    = self.load_or_def_option('true_face_power', 0.0)
        default_face_style_power   = self.options['face_style_power']   = self.load_or_def_option('face_style_power', 0.0)
        default_bg_style_power     = self.options['bg_style_power']     = self.load_or_def_option('bg_style_power', 0.0)
        default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)
        default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(suggest_batch_size)
            #self.options['use_fp16'] = io.input_bool ("Use fp16", default_usefp16, help_message='Increases training/inference speed, reduces model size. Model may crash. Enable it after 1-5k iters.')

        if self.is_first_run():
            resolution = io.input_int("Resolution", default_resolution, add_info="64-640", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi.")
            resolution = np.clip ( (resolution // 16) * 16, min_res, max_res)
            self.options['resolution'] = resolution



            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()

            while True:
                archi = io.input_str ("AE architecture", default_archi, help_message=\
"""
'df' keeps more identity-preserved face.
'liae' can fix overly different face shapes.
'-u' increased likeness of the face.
'-d' (experimental) doubling the resolution using the same computation cost.
Examples: df, liae, df-d, df-ud, liae-ud, ...
""").lower()

                archi_split = archi.split('-')

                if len(archi_split) == 2:
                    archi_type, archi_opts = archi_split
                elif len(archi_split) == 1:
                    archi_type, archi_opts = archi_split[0], None
                else:
                    continue

                if archi_type not in ['df', 'liae']:
                    continue

                if archi_opts is not None:
                    if len(archi_opts) == 0:
                        continue
                    if len([ 1 for opt in archi_opts if opt not in ['u','d','t','c'] ]) != 0:
                        continue

                    if 'd' in archi_opts:
                        self.options['resolution'] = np.clip ( (self.options['resolution'] // 32) * 32, min_res, max_res)

                break
            self.options['archi'] = archi

        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if self.options['face_type'] == 'wf' or self.options['face_type'] == 'head':
                self.options['masked_training']  = io.input_bool ("Masked training", default_masked_training, help_message="This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly.")

            self.options['eyes_mouth_prio'] = io.input_bool ("Eyes and mouth priority", default_eyes_mouth_prio, help_message='Helps to fix eye problems during training like "alien eyes" and wrong eyes direction. Also makes the detail of the teeth higher.')
            self.options['uniform_yaw'] = io.input_bool ("Uniform yaw distribution of samples", default_uniform_yaw, help_message='Helps to fix blurry side faces due to small amount of them in the faceset.')
            self.options['blur_out_mask'] = io.input_bool ("Blur out mask", default_blur_out_mask, help_message='Blurs nearby area outside of applied face mask of training samples. The result is the background near the face is smoothed and less noticeable on swapped face. The exact xseg mask in src and dst faceset is required.')

        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_gan_patch_size     = self.options['gan_patch_size']     = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
        default_gan_dims           = self.options['gan_dims']           = self.load_or_def_option('gan_dims', 16)

        if self.is_first_run() or ask_override:
            # Il testo TF ("place they on CPU to free up extra VRAM, thus set
            # bigger dimensions") descriveva il comportamento del grafo TF, dove
            # models_opt_device era uno scope per le sole *variabili* e le op
            # restavano sulla GPU: si perdeva velocita', si guadagnava VRAM, e
            # "dimensioni piu' grandi" era vero. In torch i parametri vivono dove
            # gira il modulo (vedi il commento a :765-771), quindi rispondere `n`
            # sposta l'intero training sulla CPU. Il testo dice cosa succede
            # davvero: la promessa vecchia lasciava l'utente con un trainer
            # inutilizzabile senza un avviso.
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. Answering n moves the whole model to the CPU: in this PyTorch port the weights and the computation cannot be split, so training then runs entirely on the CPU and is far slower. It frees all the VRAM, but it is not a way to fit bigger dimensions on the GPU. Outside training (merge and export) this option is ignored: use --cpu-only there.")

            self.options['adabelief'] = io.input_bool ("Use AdaBelief optimizer?", default_adabelief, help_message="Use AdaBelief optimizer. It requires more VRAM, but the accuracy and the generalization of the model is higher.")

            self.options['lr_dropout']  = io.input_str (f"Use learning rate dropout", default_lr_dropout, ['n','y','cpu'], help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations. Enabled it before `disable random warp` and before GAN. \nn - disabled.\ny - enabled\ncpu - enabled on CPU. This allows not to use extra VRAM, sacrificing 20% time of iteration.")

            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['random_hsv_power'] = np.clip ( io.input_number ("Random hue/saturation/light intensity", default_random_hsv_power, add_info="0.0 .. 0.3", help_message="Random hue/saturation/light intensity applied to the src face set only at the input of the neural network. Stabilizes color perturbations during face swapping. Reduces the quality of the color transfer by selecting the closest one in the src faceset. Thus the src faceset must be diverse enough. Typical fine value is 0.05"), 0.0, 0.3 )

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 5.0", help_message="Forces the neural network to learn small details of the face. Enable it only when the face is trained enough with lr_dropout(on) and random_warp(off), and don't disable. The higher the value, the higher the chances of artifacts. Typical fine value is 0.1"), 0.0, 5.0 )

            if self.options['gan_power'] != 0.0:
                gan_patch_size = np.clip ( io.input_int("GAN patch size", default_gan_patch_size, add_info="3-640", help_message="The higher patch size, the higher the quality, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is resolution / 8." ), 3, 640 )
                self.options['gan_patch_size'] = gan_patch_size

                gan_dims = np.clip ( io.input_int("GAN dimensions", default_gan_dims, add_info="4-512", help_message="The dimensions of the GAN network. The higher dimensions, the more VRAM is required. You can get sharper edges even at the lowest setting. Typical fine value is 16." ), 4, 512 )
                self.options['gan_dims'] = gan_dims

            if 'df' in self.options['archi']:
                self.options['true_face_power'] = np.clip ( io.input_number ("'True face' power.", default_true_face_power, add_info="0.0000 .. 1.0", help_message="Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01 . Comparison - https://i.imgur.com/czScS9q.png"), 0.0, 1.0 )
            else:
                self.options['true_face_power'] = 0.0

            self.options['face_style_power'] = np.clip ( io.input_number("Face style power", default_face_style_power, add_info="0.0..100.0", help_message="Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse."), 0.0, 100.0 )
            self.options['bg_style_power'] = np.clip ( io.input_number("Background style power", default_bg_style_power, add_info="0.0..100.0", help_message="Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. For whole_face you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0"), 0.0, 100.0 )

            self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. Try all modes to find the best.")
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain, help_message="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly. Forces random_warp=N, random_flips=Y, gan_power=0.0, lr_dropout=N, styles=0.0, uniform_yaw=Y")

        # Solo addestrando: il percorso di pretraining lo consuma unicamente
        # il ramo `if self.is_training:` piu' sotto (:1065-1066), per pescarci
        # i campioni. Un modello salvato in pretraining resta un modello, e
        # mergiarlo o esportarlo non chiede volti da nessuna parte -- ne'
        # `main.py merge` ne' `main.py exportdfm` hanno un argomento per
        # passarne una cartella, quindi senza questa condizione quei due
        # comandi sono semplicemente irraggiungibili per quel modello.
        if self.is_training and self.options['pretrain'] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        # NCHW sempre: nn.set_data_format solleva su qualunque altro valore.
        # Il TF ripiegava su NHWC in debug e senza GPU; quel ramo si perde
        # consapevolmente, perche' core/leras e' NCHW.
        self.model_data_format = "NCHW"
        nn.initialize(data_format=self.model_data_format)

        self.resolution = resolution = self.options['resolution']
        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]

        if 'eyes_prio' in self.options:
            self.options.pop('eyes_prio')

        eyes_mouth_prio = self.options['eyes_mouth_prio']

        archi_split = self.options['archi'].split('-')

        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        elif len(archi_split) == 1:
            archi_type, archi_opts = archi_split[0], None

        self.archi_type = archi_type

        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)

        adabelief = self.options['adabelief']

        use_fp16 = False
        if self.is_exporting:
            use_fp16 = io.input_bool ("Export quantized?", False, help_message='Makes the exported model faster. If you have problems, disable this option.')

        self.gan_power = gan_power = 0.0 if self.pretrain else self.options['gan_power']
        random_warp = False if self.pretrain else self.options['random_warp']
        random_src_flip = self.random_src_flip if not self.pretrain else True
        random_dst_flip = self.random_dst_flip if not self.pretrain else True
        random_hsv_power = self.options['random_hsv_power'] if not self.pretrain else 0.0
        blur_out_mask = self.options['blur_out_mask']

        if self.pretrain:
            self.options_show_override['lr_dropout'] = 'n'
            self.options_show_override['random_warp'] = False
            self.options_show_override['gan_power'] = 0.0
            self.options_show_override['random_hsv_power'] = 0.0
            self.options_show_override['face_style_power'] = 0.0
            self.options_show_override['bg_style_power'] = 0.0
            self.options_show_override['uniform_yaw'] = True

        masked_training = self.options['masked_training']
        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None


        # In TF `models_opt_device` era uno scope tf.device per le *variabili* e
        # `optimizer_vars_on_cpu` un secondo scope, indipendente, per gli
        # accumulatori. In torch i parametri vivono dove gira il modulo, quindi
        # i due scope non restano indipendenti: si applica la regola della
        # scelta di porting -- models_opt_on_gpu decide il device del
        # modulo, optimizer_vars_on_cpu continua a governare vars_on_cpu degli
        # accumulatori, che dal device del modulo e' indipendente davvero.
        #
        # Fuori dal training (merge ed export) il TF ignorava models_opt_on_gpu
        # per le *op*: le teneva su nn.tf_default_device_name se un device
        # c'era (Model.py:615 nella versione TF), mentre le variabili restavano
        # su /CPU:0 e la sessione le copiava a ogni frame. In torch i due non si
        # separano e la scelta e' una sola, quindi si segue il device delle op,
        # che e' quello che decide la velocita' del merge: l'opzione esiste per
        # liberare la VRAM dagli *accumulatori* ("Place models and optimizer on
        # GPU", :148), che fuori dal training non vengono nemmeno costruiti --
        # la memoria che il merge chiede sono i soli pesi. La conseguenza da
        # sapere: con models_opt_on_gpu=False nel model.dat il merge ora tiene i
        # pesi in VRAM invece che in RAM, e chi non ce li fa stare ha
        # `--cpu-only` (mainscripts/Merger.py:53), che era l'unica via anche col
        # TF.
        models_opt_on_gpu = False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = nn.device if models_opt_on_gpu or not self.is_training else torch.device('cpu')
        optimizer_vars_on_cpu = models_opt_device.type == 'cpu'

        input_ch=3
        self.model_filename_list = []

        # Niente placeholder: i tensori si creano dai numpy dentro le funzioni
        # di training (src_dst_train qui sotto).

        # Initializing model classes
        model_archi = nn.DeepFakeArchi(resolution, use_fp16=use_fp16, opts=archi_opts)

        if 'df' in archi_type:
            self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

            self.inter = model_archi.Inter (in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
            inter_out_ch = self.inter.get_out_ch()

            self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
            self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

            # I nomi sono quelli che saehd_flow indicizza, e sono anche i
            # prefissi delle chiavi dei pesi su disco: due Decoder identici hanno lo
            # stesso disk_key e senza il nome della rete davanti collidono.
            self.nets = {'encoder': self.encoder, 'inter': self.inter,
                         'decoder_src': self.decoder_src, 'decoder_dst': self.decoder_dst}

            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                if self.options['true_face_power'] != 0:
                    self.code_discriminator = nn.CodeDiscriminator(ae_dims, code_res=self.inter.get_out_res(), name='dis' )
                    self.nets['code_discriminator'] = self.code_discriminator
                    self.model_filename_list += [ [self.code_discriminator, 'code_discriminator.npy'] ]

        elif 'liae' in archi_type:
            self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch()*self.encoder.get_out_res(resolution)**2

            self.inter_AB = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_AB')
            self.inter_B  = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, name='inter_B')

            inter_out_ch = self.inter_AB.get_out_ch()
            inters_out_ch = inter_out_ch*2
            self.decoder = model_archi.Decoder(in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder')

            self.nets = {'encoder': self.encoder, 'inter_AB': self.inter_AB,
                         'inter_B': self.inter_B, 'decoder': self.decoder}

            self.model_filename_list += [ [self.encoder,  'encoder.npy'],
                                          [self.inter_AB, 'inter_AB.npy'],
                                          [self.inter_B , 'inter_B.npy'],
                                          [self.decoder , 'decoder.npy'] ]

        if self.is_training:
            if gan_power != 0:
                self.D_src = nn.UNetPatchDiscriminator(patch_size=self.options['gan_patch_size'], in_ch=input_ch, base_ch=self.options['gan_dims'], name="D_src")
                self.nets['D_src'] = self.D_src
                self.model_filename_list += [ [self.D_src, 'GAN.npy'] ]

        # build() e .to() prima di leggerne i pesi: get_weights() li enumera e
        # initialize_variables alloca gli accumulatori sul device del parametro
        # quando vars_on_cpu e' False.
        #
        # Qui dentro ci sono solo le reti: gli ottimizzatori entrano nella lista
        # piu' sotto, dopo questo ciclo, e i loro accumulatori li piazza
        # initialize_variables.
        for model, _ in self.model_filename_list:
            model.build()
            model.to(models_opt_device)

        def to_t(x):
            return torch.as_tensor(np.ascontiguousarray(x)).to(models_opt_device, nn.floatx)

        if self.is_training:
            # Initialize optimizers
            lr=5e-5
            if self.options['lr_dropout'] in ['y','cpu'] and not self.pretrain:
                lr_cos = 500
                lr_dropout = 0.3
            else:
                lr_cos = 0
                lr_dropout = 1.0
            OptimizerClass = nn.AdaBelief if adabelief else nn.RMSprop
            clipnorm = 1.0 if self.options['clipgrad'] else 0.0

            self.src_dst_saveable_weights, self.src_dst_trainable_weights = \
                saehd_src_dst_weights(self.nets, archi_type, random_warp)

            self.src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
            self.src_dst_opt.initialize_variables (self.src_dst_saveable_weights, vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
            self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

            if self.options['true_face_power'] != 0:
                self.D_code_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='D_code_opt')
                self.D_code_opt.initialize_variables ( self.code_discriminator.optimizer_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')
                self.model_filename_list += [ (self.D_code_opt, 'D_code_opt.npy') ]

            if gan_power != 0:
                self.D_src_dst_opt = OptimizerClass(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
                self.D_src_dst_opt.initialize_variables ( self.D_src.optimizer_weights(), vars_on_cpu=optimizer_vars_on_cpu, lr_dropout_on_cpu=self.options['lr_dropout']=='cpu')#+self.D_src_x2.get_weights()
                self.model_filename_list += [ (self.D_src_dst_opt, 'GAN_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            #
            # gpu_count resta nel calcolo del batch, ma il percorso e' a device
            # singolo: il loop `for gpu_id in range(gpu_count)` del TF, che
            # tagliava il batch e mediava i gradienti per GPU, non e' stato
            # portato: con piu' di una GPU si allena sulla prima sola, con
            # l'intero batch.
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            # I quattro power restano quelli che il grafo TF leggeva: gan_power
            # e' gia' azzerato dal pretrain (Model.py:230), true_face_power e i
            # due style no -- i loro rami hanno il proprio `not pretrain` dentro
            # (:471 e :475, per gli style) o nella chiamata del passo (:776, per
            # true_face). Le righe sono nella numerazione TF, come ovunque qui.
            train_cfg = {'archi_type'      : archi_type,
                         'resolution'      : resolution,
                         'masked_training' : masked_training,
                         'eyes_mouth_prio' : eyes_mouth_prio,
                         'blur_out_mask'   : blur_out_mask,
                         'gan_power'       : gan_power,
                         'true_face_power' : self.options['true_face_power'],
                         'face_style_power': self.options['face_style_power'],
                         'bg_style_power'  : self.options['bg_style_power'],
                         'pretrain'        : self.pretrain}

            def src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                              warped_dst, target_dst, target_dstm, target_dstm_em, ):
                s, d = saehd_train_step(self.nets, self.src_dst_opt,
                                        self.src_dst_trainable_weights,
                                        [to_t(x) for x in (warped_src, target_src, target_srcm, target_srcm_em,
                                                           warped_dst, target_dst, target_dstm, target_dstm_em)],
                                        train_cfg, gpu_count)
                return s.detach().cpu().numpy(), d.detach().cpu().numpy()
            self.src_dst_train = src_dst_train

            if self.options['true_face_power'] != 0:
                def D_train(warped_src, warped_dst):
                    saehd_D_code_train_step(self.nets, self.D_code_opt,
                                            [to_t(x) for x in (warped_src, warped_dst)],
                                            train_cfg, gpu_count)
                self.D_train = D_train

            if gan_power != 0:
                def D_src_dst_train(warped_src, target_src, target_srcm, target_srcm_em,  \
                                    warped_dst, target_dst, target_dstm, target_dstm_em, ):
                    saehd_D_src_dst_train_step(self.nets, self.D_src_dst_opt,
                                               [to_t(x) for x in (warped_src, target_src, target_srcm, target_srcm_em,
                                                                  warped_dst, target_dst, target_dstm, target_dstm_em)],
                                               train_cfg, gpu_count)
                self.D_src_dst_train = D_src_dst_train

            def AE_view(warped_src, warped_dst):
                return [x.cpu().numpy() for x in
                        saehd_view(self.nets, archi_type, to_t(warped_src), to_t(warped_dst))]
            self.AE_view = AE_view
        else:
            # Initializing merge function
            def AE_merge(warped_dst):
                return [x.cpu().numpy() for x in
                        saehd_merge(self.nets, archi_type, to_t(warped_dst))]
            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if 'df' in archi_type:
                    if model == self.inter:
                        do_init = True
                elif 'liae' in archi_type:
                    if model == self.inter_AB or model == self.inter_B:
                        do_init = True
            else:
                do_init = self.is_first_run()
                if self.is_training and gan_power != 0 and model == self.D_src:
                    if self.gan_model_changed:
                        do_init = True

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()


        ###############

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            random_ct_samples_path=training_data_dst_path if ct_mode is not None and not self.pretrain else None

            cpu_count = multiprocessing.cpu_count()
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            if ct_mode is not None:
                src_generators_count = int(src_generators_count * 1.5)

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, random_ct_samples_path=random_ct_samples_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=random_src_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,   'random_hsv_shift_amount' : random_hsv_power,                                        'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'ct_mode': ct_mode,                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=random_dst_flip),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':random_warp, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False                      , 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.EYES_MOUTH, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        uniform_yaw_distribution=self.options['uniform_yaw'] or self.pretrain,
                        generators_count=dst_generators_count )
                             ])

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    def export_dfm (self):
        # self.nets contiene D_src/code_discriminator solo se self.is_training
        # (Model.py:841-870): export_dfm gira con is_exporting=True e
        # is_training=False (ModelBase.__init__), quindi qui dentro non ci
        # sono mai -- ModuleDict non li porta dentro come inizializzatori
        # morti.
        output_path = self.get_strpath_storage_for_file('model.dfm')
        io.log_info(f'Dumping .dfm to {output_path}')

        out_names = ['out_face_mask:0','out_celeb_face:0','out_celeb_face_mask:0']

        # dynamo=False: l'esportatore nuovo e' il default in torch 2.13 e
        # ignora opset_version, scrivendo 18 dove il contratto vuole 12.
        torch.onnx.export(
            SAEHDExportModule(self.nets, self.archi_type).eval(),
            (torch.zeros(1, self.resolution, self.resolution, 3,
                         device=nn.device, dtype=nn.floatx),),
            output_path,
            input_names  = ['in_face:0'],
            output_names = out_names,
            opset_version = 12,
            dynamic_axes = {n: {0: 'batch'} for n in ['in_face:0'] + out_names},
            dynamo = False)

        # L'inferenza di forma di torch.onnx.export perde altezza e larghezza
        # attraversando la permute finale di SAEHDExportModule.forward
        # (NCHW->NHWC): il proto dichiarava
        # ['batch','batch','Transpose...dim_2',1] invece di [batch,96,96,1],
        # etichettando pure l'altezza col simbolo del batch -- un'uguaglianza
        # fra due assi diversi che non e' vera (misurato sul .dfm vero dello
        # smoke, task-9-brief.md). tf2onnx (riferimento T6) le dichiara
        # concrete; qui si rilegge il proto appena scritto e si riscrivono
        # altezza, larghezza e canali (assi 1, 2 e 3) sulle sole uscite,
        # lasciando l'asse 0 (batch) simbolico -- e' un difetto nei metadati
        # dichiarati, non nel calcolo: a run time il grafo era gia' corretto.
        # I canali si scrivono espliciti e non si lasciano al valore che
        # torch ha gia' inferito: sono gia' concreti qui (misurato), ma AMP
        # -- stesso pattern di export, gemello di questo -- perde anche
        # quell'asse sulla sola uscita che passa dallo slice dipendente da
        # morph_value; scrivere sempre i tre evita di dipendere da quale
        # asse l'inferenza di torch abbia deciso di perdere.
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
        if self.get_iter() == 0 and not self.pretrain and not self.pretrain_just_disabled:
            io.log_info('You are training the model from scratch. It is strongly recommended to use a pretrained model to speed up the training and improve the quality.\n')

        ( (warped_src, target_src, target_srcm, target_srcm_em), \
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        if self.options['true_face_power'] != 0 and not self.pretrain:
            self.D_train (warped_src, warped_dst)

        if self.gan_power != 0:
            self.D_src_dst_train (warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em)

        return ( ('src_loss', np.mean(src_loss) ), ('dst_loss', np.mean(dst_loss) ), )

    #override
    def onGetPreview(self, samples, for_history=False):
        ( (warped_src, target_src, target_srcm, target_srcm_em),
          (warped_dst, target_dst, target_dstm, target_dstm_em) ) = samples

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        if self.resolution <= 256:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]

                ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('SAEHD masked', np.concatenate (st_m, axis=0 )), ]
        else:
            result = []

            st = []
            for i in range(n_samples):
                ar = S[i], SS[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD src-src', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], DD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD dst-dst', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD pred', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                ar = S[i]*target_srcm[i], SS[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked src-src', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                ar = D[i]*target_dstm[i], DD[i]*DDM[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked dst-dst', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                ar = D[i]*target_dstm[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked pred', np.concatenate (st_m, axis=0 )), ]

        return result

    #override
    def get_preview_layout(self):
        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        def griglia(colonne, risultato):
            return { "righe": n_samples, "colonne": len(colonne),
                     "celle": [ list(colonne) for _ in range(n_samples) ],
                     "risultato": [0, colonne.index(risultato)],
                     "righe_sono_campioni": True }

        if self.resolution <= 256:
            colonne = ['src', 'src->src', 'dst', 'dst->dst', 'dst->src']
            return { 'SAEHD':        griglia(colonne, 'dst->src'),
                     'SAEHD masked': griglia(colonne, 'dst->src') }

        return { 'SAEHD src-src':        griglia(['src', 'src->src'], 'src->src'),
                 'SAEHD dst-dst':        griglia(['dst', 'dst->dst'], 'dst->dst'),
                 'SAEHD pred':           griglia(['dst', 'dst->src'], 'dst->src'),
                 'SAEHD masked src-src': griglia(['src', 'src->src'], 'src->src'),
                 'SAEHD masked dst-dst': griglia(['dst', 'dst->dst'], 'dst->dst'),
                 'SAEHD masked pred':    griglia(['dst', 'dst->src'], 'dst->src') }

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = SAEHDModel
