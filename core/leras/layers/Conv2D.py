import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


class Conv2D(nn.LayerBase):
    """
    use_wscale  bool enables equalized learning rate, if kernel_initializer is None, it will be forced to random_normal

    kernel_initializer / bias_initializer are `f(torch_tensor) -> None`: they
    fill the tensor in place, in *torch* layout (out,in,kh,kw), the way
    torch.nn.init.* does. This is not the TF contract, which was
    `f(shape, dtype) -> tensor` in TF layout, and the change is not merely
    cosmetic: `nn.initializers.ca` -- which the original version of this
    docstring advertised as the default -- satisfies *neither* half of it any
    more. It lost its `__call__` when the lazy "_cai_" placeholder mechanism was
    removed (so `kernel_initializer(self.weight.data)` would raise TypeError),
    and `CAInitializerSubprocessor.generate` returns (kh,kw,in,out) arrays. So
    CA is NOT the default here and cannot be passed as one: with
    kernel_initializer=None the default is xavier_uniform_ (glorot_uniform,
    which is what tf.get_variable fell back to as well -- verified fan-equivalent
    on every live layer). Nothing in the codebase passes a kernel_initializer
    today; whoever revives CA has to give it a torch-layout in-place __call__
    first.
    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        if not isinstance(strides, int):
            raise ValueError ("strides must be an int type")
        if not isinstance(dilations, int):
            raise ValueError ("dilations must be an int type")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx

        if isinstance(padding, str):
            if padding == "SAME":
                padding = ( (kernel_size - 1) * dilations + 1 ) // 2
            elif padding == "VALID":
                padding = None
            else:
                raise ValueError ("Wrong padding type. Should be VALID SAME or INT or 4x INTs")
        else:
            padding = int(padding)



        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        kernel_initializer = self.kernel_initializer
        if self.use_wscale:
            gain   = 1.0 if self.kernel_size == 1 else np.sqrt(2)
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=self.dtype)     # explicit cast, not numpy weak-typing
            if kernel_initializer is None:
                kernel_initializer = lambda t: torch.nn.init.normal_(t, 0.0, 1.0)

        self.weight = torch.nn.Parameter(
            torch.empty((self.out_ch, self.in_ch, self.kernel_size, self.kernel_size),
                        dtype=self.dtype), requires_grad=self.trainable)
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
        return {"weight": (3, 2, 0, 1)}        # (kh,kw,in,out) -> (out,in,kh,kw)

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        padding = self.padding
        if padding is not None:
            x = F.pad(x, (padding, padding, padding, padding), mode='constant')

        return F.conv2d(x, weight, self.bias if self.use_bias else None,
                        stride=self.strides, padding=0, dilation=self.dilations)

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "

        return r


nn.Conv2D = Conv2D
