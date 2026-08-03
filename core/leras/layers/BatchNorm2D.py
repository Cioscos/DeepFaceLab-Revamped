import torch

from core.leras import nn


class BatchNorm2D(nn.LayerBase):
    """
    currently not for training
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, **kwargs):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        self.weight       = torch.nn.Parameter(torch.ones ((self.dim,), dtype=self.dtype))
        self.bias         = torch.nn.Parameter(torch.zeros((self.dim,), dtype=self.dtype))
        # running_mean/running_var are not trainable but were saved variables
        # in the TF build (inference-only: this layer never updates them, no
        # momentum/training-mode behaviour). They must stay Parameters, not
        # buffers, so named_parameters()/save_weights() still see them.
        self.running_mean = torch.nn.Parameter(torch.zeros((self.dim,), dtype=self.dtype), requires_grad=False)
        self.running_var  = torch.nn.Parameter(torch.zeros((self.dim,), dtype=self.dtype), requires_grad=False)

    def forward(self, x):
        shape = (1, self.dim, 1, 1)

        weight       = torch.reshape(self.weight,       shape)
        bias         = torch.reshape(self.bias,         shape)
        running_mean = torch.reshape(self.running_mean, shape)
        running_var  = torch.reshape(self.running_var,  shape)

        x = (x - running_mean) / torch.sqrt(running_var + self.eps)
        x = x * weight
        x = x + bias
        return x


nn.BatchNorm2D = BatchNorm2D
