import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


class DepthwiseConv2D(nn.LayerBase):
    """
    default kernel_initializer - CA
    use_wscale  bool enables equalized learning rate, if kernel_initializer is None, it will be forced to random_normal
    """
    def __init__(self, in_ch, kernel_size, strides=1, padding='SAME', depth_multiplier=1, dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
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
                padding = 0
            else:
                raise ValueError ("Wrong padding type. Should be VALID SAME or INT or 4x INTs")

        if isinstance(padding, int):
            padding = padding if padding != 0 else None

        self.in_ch = in_ch
        self.depth_multiplier = depth_multiplier
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

        # RAM layout is the grouped-conv shape torch expects for
        # F.conv2d(..., groups=in_ch): (in_ch*depth_multiplier, 1, kh, kw).
        self.weight = torch.nn.Parameter(
            torch.empty((self.in_ch * self.depth_multiplier, 1, self.kernel_size, self.kernel_size),
                        dtype=self.dtype), requires_grad=self.trainable)
        if kernel_initializer is not None:
            kernel_initializer(self.weight.data)
        else:
            torch.nn.init.xavier_uniform_(self.weight.data)

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros((self.in_ch * self.depth_multiplier,), dtype=self.dtype),
                                           requires_grad=self.trainable)
            if self.bias_initializer is not None:
                self.bias_initializer(self.bias.data)

    #override
    def weight_from_disk(self, param_path, arr):
        # Not a pure permutation: disk (kh,kw,in,mult) packs (in,mult) into a
        # single grouped-conv output axis, so it needs a reshape as well as a
        # transpose. weight_layouts() is deliberately not declared for this
        # layer.
        if param_path.rsplit('.', 1)[-1] != "weight":
            return arr
        kh, kw, in_ch, mult = arr.shape
        return np.transpose(arr, (2, 3, 0, 1)).reshape(in_ch * mult, 1, kh, kw)

    #override
    def weight_to_disk(self, param_path, tensor):
        arr = tensor.detach().cpu().numpy()
        if param_path.rsplit('.', 1)[-1] != "weight":
            return arr
        _, _, kh, kw = arr.shape
        return np.transpose(arr.reshape(self.in_ch, self.depth_multiplier, kh, kw), (2, 3, 0, 1))

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        if self.padding is not None:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant')

        # Deliberately NOT passing dilation=self.dilations here: the original
        # TF build called tf.nn.depthwise_conv2d(x, weight, self.strides,
        # 'VALID', ...) with no `rate=` argument, so `dilations` only ever
        # sized the 'SAME' padding in __init__ and was never applied to the
        # convolution itself. That is almost certainly a bug in the original
        # (a `dilations` parameter that silently does nothing to the conv),
        # but this migration's contract is behavioural fidelity, bugs
        # included — see total_variation_mse for the same rule applied
        # elsewhere. Do not "fix" this by adding dilation= back.
        return F.conv2d(x, weight, self.bias if self.use_bias else None,
                        stride=self.strides, padding=0, groups=self.in_ch)

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} depth_multiplier:{self.depth_multiplier} "
        return r


nn.DepthwiseConv2D = DepthwiseConv2D
