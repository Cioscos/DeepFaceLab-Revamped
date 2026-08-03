import numpy as np
import torch

from core.leras import nn


class Dense(nn.LayerBase):
    def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        """
        use_wscale          enables weight scale (equalized learning rate)
                            if kernel_initializer is None, it will be forced to random_normal

        maxout_ch     https://link.springer.com/article/10.1186/s40537-019-0233-0
                            typical 2-4 if you want to enable DenseMaxout behaviour
        """
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = maxout_ch
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        if dtype is None:
            dtype = nn.floatx

        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        if self.maxout_ch > 1:
            weight_shape = (self.in_ch, self.out_ch * self.maxout_ch)
        else:
            weight_shape = (self.in_ch, self.out_ch)

        kernel_initializer = self.kernel_initializer

        if self.use_wscale:
            gain   = 1.0
            fan_in = np.prod(weight_shape[:-1])
            he_std = gain / np.sqrt(fan_in) # He init
            self.wscale = torch.tensor(he_std, dtype=self.dtype)     # explicit cast, not numpy weak-typing
            if kernel_initializer is None:
                kernel_initializer = lambda t: torch.nn.init.normal_(t, 0.0, 1.0)

        # weight stays (in_ch, out_ch) on disk and in RAM: forward keeps the
        # original x @ W convention instead of nn.Linear's (out, in) layout.
        self.weight = torch.nn.Parameter(torch.empty(weight_shape, dtype=self.dtype),
                                         requires_grad=self.trainable)
        if kernel_initializer is not None:
            kernel_initializer(self.weight.data)
        else:
            torch.nn.init.xavier_uniform_(self.weight.data)

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros((self.out_ch,), dtype=self.dtype),
                                           requires_grad=self.trainable)
            if self.bias_initializer is not None:
                self.bias_initializer(self.bias.data)

    #override
    def weight_layouts(self):
        return {}       # (in,out) on disk and in RAM: no permutation needed

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        x = torch.matmul(x, weight)

        if self.maxout_ch > 1:
            x = x.reshape(-1, self.out_ch, self.maxout_ch)
            x = torch.max(x, dim=-1).values

        if self.use_bias:
            x = x + self.bias.reshape(1, self.out_ch)

        return x


nn.Dense = Dense
