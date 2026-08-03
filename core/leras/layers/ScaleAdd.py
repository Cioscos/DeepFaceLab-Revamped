import torch

from core.leras import nn


class ScaleAdd(nn.LayerBase):
    def __init__(self, ch, dtype=None, **kwargs):
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        self.ch = ch

        super().__init__(**kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(torch.zeros((self.ch,), dtype=self.dtype))

    def forward(self, inputs):
        shape = (1, self.ch, 1, 1)

        weight = torch.reshape(self.weight, shape)

        x0, x1 = inputs
        x = x0 + x1 * weight

        return x


nn.ScaleAdd = ScaleAdd
