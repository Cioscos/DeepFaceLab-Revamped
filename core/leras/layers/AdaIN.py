import torch

from core.leras import nn


class AdaIN(nn.LayerBase):
    """
    """
    def __init__(self, in_ch, mlp_ch, kernel_initializer=None, dtype=None, **kwargs):
        self.in_ch = in_ch
        self.mlp_ch = mlp_ch
        self.kernel_initializer = kernel_initializer

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        kernel_initializer = self.kernel_initializer

        # weight1/weight2 are (mlp_ch, in_ch), used as mlp @ weight, so no
        # disk<->RAM layout conversion is needed (same reasoning as Dense).
        self.weight1 = torch.nn.Parameter(torch.empty((self.mlp_ch, self.in_ch), dtype=self.dtype))
        self.bias1   = torch.nn.Parameter(torch.zeros((self.in_ch,), dtype=self.dtype))
        self.weight2 = torch.nn.Parameter(torch.empty((self.mlp_ch, self.in_ch), dtype=self.dtype))
        self.bias2   = torch.nn.Parameter(torch.zeros((self.in_ch,), dtype=self.dtype))

        if kernel_initializer is not None:
            kernel_initializer(self.weight1.data)
            kernel_initializer(self.weight2.data)
        else:
            torch.nn.init.kaiming_normal_(self.weight1.data)
            torch.nn.init.kaiming_normal_(self.weight2.data)

    def forward(self, inputs):
        x, mlp = inputs

        gamma = torch.matmul(mlp, self.weight1)
        gamma = gamma + torch.reshape(self.bias1, (1, self.in_ch))

        beta = torch.matmul(mlp, self.weight2)
        beta = beta + torch.reshape(self.bias2, (1, self.in_ch))

        shape = (-1, self.in_ch, 1, 1)

        x_mean = torch.mean(x, dim=nn.conv2d_spatial_axes, keepdim=True)
        # Population std (ddof=0), +1e-5 on the std itself, same as InstanceNorm2D.
        x_std  = torch.std(x, dim=nn.conv2d_spatial_axes, keepdim=True, unbiased=False) + 1e-5

        x = (x - x_mean) / x_std
        x = x * torch.reshape(gamma, shape)
        x = x + torch.reshape(beta, shape)

        return x


nn.AdaIN = AdaIN
