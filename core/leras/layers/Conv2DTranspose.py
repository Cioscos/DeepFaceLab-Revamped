import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


class Conv2DTranspose(nn.LayerBase):
    """
    use_wscale      enables weight scale (equalized learning rate)
                    if kernel_initializer is None, it will be forced to random_normal
    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs ):
        if not isinstance(strides, int):
            raise ValueError ("strides must be an int type")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
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
            he_std = gain / np.sqrt(fan_in) # He init
            self.wscale = torch.tensor(he_std, dtype=self.dtype)     # explicit cast, not numpy weak-typing
            if kernel_initializer is None:
                kernel_initializer = lambda t: torch.nn.init.normal_(t, 0.0, 1.0)

        self.weight = torch.nn.Parameter(
            torch.empty((self.in_ch, self.out_ch, self.kernel_size, self.kernel_size),
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
        return {"weight": (3, 2, 0, 1)}        # (kh,kw,out,in) -> (in,out,kh,kw)

    def _conv_transpose_crop(self):
        # deconv_length (original TF build) defines output length as in*s for
        # 'SAME' and in*s + max(k-s, 0) for 'VALID'. F.conv_transpose2d with
        # padding=0/output_padding=0 always produces the "full" length
        # (in-1)*s + k.
        #
        # A first attempt equated the two lengths and solved for a single
        # symmetric F.conv_transpose2d(padding=p, output_padding=op) — that
        # reproduces the right *shape* but the wrong *values*: F.conv_transpose2d's
        # `padding` trims symmetrically (p off each side), while TF's real
        # SAME deconvolution is the adjoint of a forward SAME conv2d, whose
        # padding split (pad_before = pad_total // 2, pad_after = pad_total -
        # pad_before) is generally asymmetric. Confirmed against the reference
        # implementation for k=3, s=2: the symmetric-crop formula produced an
        # output shifted by one pixel from it (max diff ~4.2), while
        # cropping the full-length output asymmetrically (0 off the front, 1
        # off the back) matches to float32 precision.
        #
        # So for SAME this crops the full-length output directly (asymmetric,
        # matching the forward-SAME padding split) instead of routing through
        # F.conv_transpose2d's own (symmetric) padding/output_padding. `grow`
        # covers the k < s case, where the full length is already short of
        # the target and needs zeros appended via output_padding instead of a
        # crop.
        #
        # VALID needs the same k<s growth: deconv_length defines its target
        # as n*s + max(k-s, 0), which for k<s is just n*s, while the
        # full-length output (padding=0, output_padding=0) is only
        # (n-1)*s + k = n*s - (s-k). Both branches share the same `grow`
        # formula; VALID never needs a crop since for k>=s the full length
        # already equals the VALID target exactly.
        k, s = self.kernel_size, self.strides
        if self.padding == 'SAME':
            pad_total  = max(k - s, 0)
            pad_before = pad_total // 2
            pad_after  = pad_total - pad_before
            grow       = max(s - k, 0)
        elif self.padding == 'VALID':
            pad_before, pad_after, grow = 0, 0, max(s - k, 0)
        else:
            raise ValueError(f"unsupported padding: {self.padding}")
        return pad_before, pad_after, grow

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        pad_before, pad_after, grow = self._conv_transpose_crop()
        x = F.conv_transpose2d(x, weight, self.bias if self.use_bias else None,
                               stride=self.strides, padding=0, output_padding=grow)
        if pad_before or pad_after:
            h, w = x.shape[2], x.shape[3]
            x = x[:, :, pad_before:h - pad_after, pad_before:w - pad_after]
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "

        return r


nn.Conv2DTranspose = Conv2DTranspose
