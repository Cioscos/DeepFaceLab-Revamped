"""IR-101 come in `net.py` di AdaFace (mk-minchul/AdaFace, MIT): input_layer
(conv-BN-PReLU), body di bottleneck IR [3,13,30,3], output_layer (BN-Dropout-
Flatten-Linear-BN). Chiavi identiche al .ckpt ufficiale senza `model.`."""
import numpy as np
import torch
import torch.nn as nn


class BottleneckIR(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.shortcut_layer = (nn.MaxPool2d(1, stride) if i == o else
                               nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o)))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(i), nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.PReLU(o),
            nn.Conv2d(o, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o))

    def forward(self, x): return self.res_layer(x) + self.shortcut_layer(x)


def _blocchi(profondita):
    piani = [(64, 64, profondita[0]), (64, 128, profondita[1]), (128, 256, profondita[2]), (256, 512, profondita[3])]
    out = []
    for i, o, n in piani:
        out.append(BottleneckIR(i, o, 2))
        out += [BottleneckIR(o, o, 1) for _ in range(n - 1)]
    return out


class IR101(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        self.body = nn.Sequential(*_blocchi((3, 13, 30, 3)))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.4), nn.Flatten(),
                                          nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512, affine=False))

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))

    # Confini dei quattro gruppi di `_blocchi((3, 13, 30, 3))`: dopo i blocchi
    # 2, 15 e 45 le feature stanno a 56, 28 e 14; il quarto gruppo (7) e'
    # troppo vicino all'identita' per fare da vincolo di posa (IFSR, FaceDancer).
    CONFINI_STADI = (2, 15, 45)

    def stadi(self, x):
        x = self.input_layer(x); out = []
        for i, blocco in enumerate(self.body):
            x = blocco(x)
            if i in self.CONFINI_STADI:
                out.append(x)
        return out


def encoder(rete):
    """HWC BGR [0,1] (gia' 112x112) -> (512,). AdaFace vuole BGR in [-1,1]."""
    dev = next(rete.parameters()).device

    @torch.no_grad()
    def f(img):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(dev)
        return rete(t * 2 - 1)[0].cpu().numpy()
    return f
