"""Il passo, la preview, il merge e l'export di H2. Le loss sono quelle di
SAEHD (saehd_src_dst_loss) sul flusso di H2: un solo inter, un decoder
letto con il vettore di src o di dst. Il ramo maschera legge il solo codice,
quindi pred_src_dstm e pred_dst_dstm coincidono: e' il prezzo del ramo
parallelo pre-allenato (documentazione del ciclo H2 sotto docs/)."""
import torch

from core.leras import nn
from models.Model_SAEHD.Model import saehd_blur_out_mask, saehd_src_dst_loss


def codice_doppio(nets, x):
    """z2 = cat([z, z]): il decoder del RTM ha 2*ae_out_ch canali in ingresso,
    ed e' cio' che liae gli da' sul ramo swap."""
    z = nets['inter'](nets['encoder'](x))
    return torch.cat([z, z], nn.conv2d_ch_axis)


def h2_flow(nets, warped_src, warped_dst):
    n = warped_src.shape[0]
    z_src, z_dst = codice_doppio(nets, warped_src), codice_doppio(nets, warped_dst)
    e_src, e_dst = nets['identita']('src', n), nets['identita']('dst', n)
    pred_src_src, pred_src_srcm = nets['decoder'](z_src, e_src)
    pred_dst_dst, pred_dst_dstm = nets['decoder'](z_dst, e_dst)
    pred_src_dst, pred_src_dstm = nets['decoder'](z_dst, e_src)
    return pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm


def h2_src_dst_weights(nets):
    """(saveable, trainable): l'identita' entra solo se allenabile (N2); con
    i vettori AdaFace congelati (N1) l'ottimizzatore non li vede."""
    ordine = ['encoder', 'inter', 'decoder'] + (['identita'] if nets['identita'].allenabile else [])
    return ([w for nome in ordine for w in nets[nome].optimizer_weights()],
            [w for nome in ordine for w in nets[nome].get_weights()])


def h2_train_step(nets, opt, trainable_weights, batch, cfg, gpu_count, loss_extra=None):
    """saehd_train_step con h2_flow: stesse loss, stesso hook, stesso
    aggiornamento. pred_src_dst_no_code_grad e' None perche' face_style e
    bg_style non sono opzioni di H2 e cfg le porta a 0."""
    warped_src, target_src, target_srcm, target_srcm_em, \
    warped_dst, target_dst, target_dstm, target_dstm_em = batch
    resolution = cfg['resolution']

    if cfg['blur_out_mask']:
        target_src = saehd_blur_out_mask(target_src, target_srcm, resolution)
        target_dst = saehd_blur_out_mask(target_dst, target_dstm, resolution)

    pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm = \
        h2_flow(nets, warped_src, warped_dst)

    src_loss, dst_loss, target_srcm_blur = saehd_src_dst_loss(
        target_src, target_srcm, target_srcm_em, pred_src_src, pred_src_srcm,
        target_dst, target_dstm, target_dstm_em, pred_dst_dst, pred_dst_dstm,
        pred_src_dst, pred_src_dstm, None,
        resolution, cfg['masked_training'], cfg['eyes_mouth_prio'], 0.0, 0.0, False)

    G_loss = src_loss + dst_loss
    if loss_extra is not None:
        G_loss = G_loss + loss_extra(dict(
            pred_src_src=pred_src_src, pred_dst_dst=pred_dst_dst,
            pred_src_dst=pred_src_dst, pred_src_dstm=pred_src_dstm,
            target_src=target_src, target_dst=target_dst,
            target_srcm=target_srcm, target_dstm=target_dstm,
            target_srcm_blur=target_srcm_blur))

    opt.step(nn.gradients(G_loss.sum() / gpu_count, trainable_weights))
    return src_loss, dst_loss


def h2_view(nets, warped_src, warped_dst):
    """Le cinque predizioni nell'ordine di saehd_view: onGetPreview di SAEHD
    le spacchetta posizionali."""
    with torch.no_grad():
        pred_src_src, _, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm = h2_flow(nets, warped_src, warped_dst)
    return [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm]


def h2_merge(nets, warped_dst, morph):
    """[pred_src_dst, pred_src_dstm, pred_src_dstm], l'ordine di predictor_func:
    le due maschere coincidono (il decoder ne calcola una sola), quindi si
    passa il medesimo tensore due volte. Il vettore e' l'interpolazione a
    `morph` (0 = dst, 1 = src)."""
    with torch.no_grad():
        z2 = codice_doppio(nets, warped_dst)
        e = nets['identita'].interpola(morph)[None].expand(z2.shape[0], -1)
        pred_src_dst, pred_src_dstm = nets['decoder'](z2, e)
    return [pred_src_dst, pred_src_dstm, pred_src_dstm]


class H2ExportModule(torch.nn.Module):
    """Il sottografo del .dfm: un ingresso immagine NHWC, morph_value (1,) letto
    come tensore (mai int()/float(): sotto il tracciamento diventerebbe una
    costante e l'ingresso sparirebbe dal grafo, la trappola di AMP), tre
    uscite nell'ordine di SAEHD: out_face_mask, out_celeb_face,
    out_celeb_face_mask."""
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleDict(nets)

    def forward(self, in_face, morph_value):
        x = in_face.permute(0, 3, 1, 2)
        z2 = codice_doppio(self.nets, x)
        e = self.nets['identita'].interpola(morph_value[0])[None].expand(z2.shape[0], -1)
        pred_src_dst, pred_src_dstm = self.nets['decoder'](z2, e)
        return (pred_src_dstm.permute(0, 2, 3, 1),
                pred_src_dst.permute(0, 2, 3, 1),
                pred_src_dstm.permute(0, 2, 3, 1))
