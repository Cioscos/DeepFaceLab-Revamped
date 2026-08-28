"""LPIPS v0.1 su AlexNet: features fino a relu5, normalizzazione per canale,
5 strati lineari 1x1 appresi, media spaziale, somma sugli strati (Zhang 2018)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _alexnet_features():
    return nn.Sequential(
        nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))


_TAGLI = (2, 5, 8, 10, 12)          # gli indici dopo ciascun ReLU 1..5
_CANALI = (64, 192, 384, 256, 256)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _alexnet_features()
        self.lin = nn.ModuleList(nn.Conv2d(c, 1, 1, bias=False) for c in _CANALI)
        for m in self.lin:
            m.weight.data.abs_()   # pesi non negativi: la distanza cresce con la differenza di feature anche prima di caricare i pesi appresi
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("scale", torch.tensor([.458, .448, .450]).view(1, 3, 1, 1), persistent=False)

    def _strati(self, x):
        x = (x * 2 - 1 - self.shift) / self.scale
        out, inizio = [], 0
        for fine in _TAGLI:
            x = self.features[inizio:fine](x); inizio = fine
            out.append(x / (x.norm(dim=1, keepdim=True) + 1e-10))
        return out

    def forward(self, a, b):                                   # NCHW RGB in [0,1]
        d = torch.zeros(a.shape[0], device=a.device)
        for fa, fb, lin in zip(self._strati(a), self._strati(b), self.lin):
            d = d + lin((fa - fb) ** 2).mean(dim=(1, 2, 3))
        return d


@torch.no_grad()
def distanza(rete, a, b):
    dev = next(rete.parameters()).device
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x[..., ::-1])).permute(2, 0, 1)[None].float().to(dev)
    return float(rete(t(a), t(b))[0])
