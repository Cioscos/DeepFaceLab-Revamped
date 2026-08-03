import torch

from core.leras import nn


class DenseNorm(nn.LayerBase):
    def __init__(self, dense=False, eps=1e-06, dtype=None, **kwargs):
        self.dense = dense
        if dtype is None:
            dtype = nn.floatx
        self.eps = eps

        super().__init__(**kwargs)

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)


nn.DenseNorm = DenseNorm
