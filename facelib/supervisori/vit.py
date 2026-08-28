"""ViT di DINOv2 (patch 14, cls token, pre-norm, MLP 4x), con le chiavi dello
state_dict ufficiale `dinov2_vits14_pretrain.pth`. Nessun peso qui dentro."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attenzione(nn.Module):
    def __init__(self, dim, teste):
        super().__init__(); self.teste = teste
        self.qkv = nn.Linear(dim, 3 * dim); self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, l, d = x.shape
        q, k, v = self.qkv(x).view(n, l, 3, self.teste, d // self.teste).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.teste), -1)
        return self.proj((a @ v).transpose(1, 2).reshape(n, l, d))


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.fc1 = nn.Linear(dim, 4 * dim); self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    """Il layer-scale di DINOv2: un gamma per canale (chiave `blocks.N.ls1.gamma`)."""
    def __init__(self, dim):
        super().__init__(); self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x): return x * self.gamma


class Blocco(nn.Module):
    def __init__(self, dim, teste):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6); self.attn = Attenzione(dim, teste); self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6); self.mlp = Mlp(dim); self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__(); self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(self, dim=384, profondita=12, teste=6, patch=14, lato_pretrain=518):
        super().__init__()
        self.patch = patch
        self.patch_embed = PatchEmbed(dim, patch)
        # il checkpoint ufficiale e' pre-addestrato a 518px: pos_embed ha
        # una voce per ognuna delle 37x37 patch piu' il token cls
        self.griglia_pretrain = lato_pretrain // patch
        n = self.griglia_pretrain ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))     # nel checkpoint, non nel forward
        self.blocks = nn.ModuleList(Blocco(dim, teste) for _ in range(profondita))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.register_buffer("media", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("scarto", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _pos_embed_per_griglia(self, gh, gw, dim):
        """Se la griglia dell'ingresso e' quella di pre-addestramento (37x37 a
        518px) usa pos_embed cosi' com'e'; altrimenti separa il token cls dalla
        parte patch e la interpola bicubicamente, come l'implementazione
        ufficiale di DINOv2 (`interpolate_pos_encoding`)."""
        if (gh, gw) == (self.griglia_pretrain, self.griglia_pretrain):
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, self.griglia_pretrain, self.griglia_pretrain, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, dim)
        return torch.cat([cls_pos, patch_pos], 1)

    def forward(self, x):                                  # x NCHW, RGB in [0,1]
        n, c, h, w = x.shape
        x = (x - self.media) / self.scarto
        x = self.patch_embed(x)
        pos = self._pos_embed_per_griglia(h // self.patch, w // self.patch, x.shape[-1])
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


def dinov2_vits14():
    return ViT(384, 12, 6, 14, 518)


def _nchw_rgb(a):
    """HWC BGR [0,1] -> NCHW RGB, ridimensionato a 224 se serve."""
    t = torch.from_numpy(np.ascontiguousarray(a[..., ::-1])).permute(2, 0, 1)[None].float()
    if t.shape[-2:] != (224, 224):
        t = F.interpolate(t, (224, 224), mode="bilinear", align_corners=False)
    return t


@torch.no_grad()
def token_perceptual(rete, a, b):
    dev = next(rete.parameters()).device
    ta, tb = rete(_nchw_rgb(a).to(dev))[:, 1:], rete(_nchw_rgb(b).to(dev))[:, 1:]
    return float((ta - tb).abs().mean())
