import torch

from core.leras import nn


class InstanceNorm2D(nn.LayerBase):
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(torch.empty((self.in_ch,), dtype=self.dtype))
        # xavier_uniform_ needs >=2 dims to compute fan_in/fan_out; reshape(1,-1)
        # is a view into the same storage, so this still initializes self.weight
        # in place. Loaded weights overwrite this value anyway.
        torch.nn.init.xavier_uniform_(self.weight.data.reshape(1, -1))
        self.bias   = torch.nn.Parameter(torch.zeros((self.in_ch,), dtype=self.dtype))

    def forward(self, x):
        shape = (1, self.in_ch, 1, 1)

        weight = torch.reshape(self.weight, shape)
        bias   = torch.reshape(self.bias,   shape)

        x_mean = torch.mean(x, dim=nn.conv2d_spatial_axes, keepdim=True)
        # tf.math.reduce_std is the population std (ddof=0); the +1e-5 is
        # added to the std itself, not to the variance under the sqrt.
        x_std  = torch.std(x, dim=nn.conv2d_spatial_axes, keepdim=True, unbiased=False) + 1e-5

        x = (x - x_mean) / x_std
        x = x * weight
        x = x + bias

        return x


nn.InstanceNorm2D = InstanceNorm2D
