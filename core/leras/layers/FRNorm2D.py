import torch

from core.leras import nn


class FRNorm2D(nn.LayerBase):
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
        self.weight = torch.nn.Parameter(torch.ones ((self.in_ch,), dtype=self.dtype))
        self.bias   = torch.nn.Parameter(torch.zeros((self.in_ch,), dtype=self.dtype))
        self.eps    = torch.nn.Parameter(torch.full((1,), 1e-6, dtype=self.dtype))

    def forward(self, x):
        shape  = (1, self.in_ch, 1, 1)
        weight = torch.reshape(self.weight, shape)
        bias   = torch.reshape(self.bias,   shape)

        nu2 = torch.mean(torch.square(x), dim=nn.conv2d_spatial_axes, keepdim=True)
        x   = x * (1.0 / torch.sqrt(nu2 + torch.abs(self.eps)))

        return x * weight + bias


nn.FRNorm2D = FRNorm2D
