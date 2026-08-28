"""I termini di loss di H1: per campione, (N,), su BGR [0,1] NCHW.
Le reti sono congelate; il gradiente scorre solo verso `x`."""
import torch
import torch.nn.functional as F


MARGINE_BLEED = 0.2


def embedding_swap(encoder, x112):
    """L'embedding AdaFace normalizzato di x (BGR in [0,1]): unico punto che fa
    il forward sullo swap, cosi' id e bleed lo condividono quando sono accesi
    insieme invece di richiamare la rete due volte."""
    return F.normalize(encoder(x112 * 2 - 1).float(), dim=1)


def id_loss_da_embedding(e, e_rif):
    """1 - coseno fra un embedding gia' calcolato e il riferimento."""
    return 1.0 - (e * e_rif.to(e.device, e.dtype)).sum(1)


def id_loss(encoder, x112, e_rif):
    """1 - coseno fra l'embedding di x (AdaFace vuole BGR in [-1,1]) e il riferimento."""
    return id_loss_da_embedding(embedding_swap(encoder, x112), e_rif)


def bleed_da_embedding(e, e_dst):
    """Quanto un embedding gia' calcolato somiglia al riferimento dst oltre il
    margine di tolleranza: zero fino a MARGINE_BLEED, cresce linearmente sopra."""
    return F.relu((e * e_dst.to(e.device, e.dtype)).sum(1) - MARGINE_BLEED)


def bleed(encoder, x112, e_dst):
    """Repulsione dello swap dal riferimento dst: bleed_da_embedding sull'embedding di x."""
    return bleed_da_embedding(embedding_swap(encoder, x112), e_dst)


def ifsr(encoder, x112, y112):
    """L1 sulle feature intermedie dell'encoder fra x e y (y senza gradiente)."""
    with torch.no_grad():
        fy = encoder.stadi(y112 * 2 - 1)
    fx = encoder.stadi(x112 * 2 - 1)
    return sum((a.float() - b.float()).abs().mean((1, 2, 3)) for a, b in zip(fx, fy))


def _token(vit, t):
    t = t.flip(1).float()                                  # BGR -> RGB
    if t.shape[-2:] != (224, 224):
        t = F.interpolate(t, (224, 224), mode="bilinear", align_corners=False)
    return vit(t)[:, 1:]                                   # senza il cls


def dino_perceptual(vit, x, y):
    with torch.no_grad():
        ty = _token(vit, y)
    return (_token(vit, x).float() - ty.float()).abs().mean((1, 2))


def focal_frequency(x, y, alpha=1.0):
    """Focal Frequency Loss (Jiang et al. 2021): spettro ortonormale per canale,
    peso |d|^alpha normalizzato al massimo dell'immagine e staccato dal grafo."""
    d = torch.fft.fft2(x.float(), norm="ortho") - torch.fft.fft2(y.float(), norm="ortho")
    mag2 = d.real ** 2 + d.imag ** 2
    w = mag2.sqrt() ** alpha
    w = (w / w.amax((1, 2, 3), keepdim=True).clamp_min(1e-12)).detach()
    return (w * mag2).mean((1, 2, 3))
