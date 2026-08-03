import torch

from core.leras import nn


class TLU(nn.LayerBase):
    """
    Tensorflow implementation of
    Filter Response Normalization Layer: Eliminating Batch Dependence in theTraining of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        self.tau = torch.nn.Parameter(torch.zeros((self.in_ch,), dtype=self.dtype))

    def forward(self, x):
        shape = (1, self.in_ch, 1, 1)

        tau = torch.reshape(self.tau, shape)
        return torch.maximum(x, tau)


nn.TLU = TLU
