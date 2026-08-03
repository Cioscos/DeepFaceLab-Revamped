import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


def torch_gradients(loss, vars):
    grads = torch.autograd.grad(loss, vars, allow_unused=True)
    gv = [*zip(grads, vars)]
    for i, (g, v) in enumerate(gv):
        if g is None:
            raise Exception(f"Variable {i} with shape {tuple(v.shape)} is declared as trainable, but no tensors flow through it.")
    return gv
nn.gradients = torch_gradients


def average_gv_list(grad_var_list, device=None):
    if len(grad_var_list) == 1:
        return grad_var_list[0]

    result = []
    for i, gv in enumerate(grad_var_list):
        for j, (g, v) in enumerate(gv):
            if i == 0:
                result += [[[g], v]]
            else:
                result[j][0] += [g]

    for i, (gs, v) in enumerate(result):
        result[i] = (torch.mean(torch.stack(gs, 0), 0), v)
    return result
nn.average_gv_list = average_gv_list


def average_tensor_list(tensors_list, device=None):
    if len(tensors_list) == 1:
        return tensors_list[0]
    return torch.mean(torch.stack(tensors_list, 0), 0)
nn.average_tensor_list = average_tensor_list


def concat(tensors_list, axis):
    """
    Better version.
    """
    if len(tensors_list) == 1:
        return tensors_list[0]
    return torch.cat(tensors_list, axis)
nn.concat = concat


def flatten(x):
    return torch.reshape(x, (-1, int(np.prod(x.shape[1:]))))
nn.flatten = flatten


def max_pool(x, kernel_size=2, strides=2):
    # TF's 'SAME' padding distributes asymmetrically (more on bottom/right)
    # when the input isn't evenly divisible by the stride; F.max_pool2d's
    # symmetric `padding=` argument can't express that, so pad explicitly.
    ih, iw = x.shape[nn.conv2d_spatial_axes[0]], x.shape[nn.conv2d_spatial_axes[1]]
    oh, ow = -(-ih // strides), -(-iw // strides)  # ceil division, TF 'SAME' output size
    pad_h  = max((oh - 1) * strides + kernel_size - ih, 0)
    pad_w  = max((ow - 1) * strides + kernel_size - iw, 0)
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
              value=float("-inf"))
    return F.max_pool2d(x, kernel_size, strides, padding=0)
nn.max_pool = max_pool


def reshape_4D(x, w, h, c):
    return torch.reshape(x, (-1, c, h, w))
nn.reshape_4D = reshape_4D


def upsample2d(x, size=2):
    return F.interpolate(x, scale_factor=size, mode='nearest')
nn.upsample2d = upsample2d


def resize2d_bilinear(x, size=2):
    h = x.shape[nn.conv2d_spatial_axes[0]]
    w = x.shape[nn.conv2d_spatial_axes[1]]

    if size > 0:
        new_h, new_w = h * size, w * size
    else:
        new_h, new_w = h // -size, w // -size

    # nn.py aliases `tf = tensorflow.compat.v1` with `disable_v2_behavior()`,
    # so the original tf.image.resize(..., method=BILINEAR) call runs the
    # TF1 legacy kernel: align_corners=False WITHOUT half-pixel centers, i.e.
    # src_coord = out_idx * (in_size / out_size), no +0.5/-0.5 offset. That is
    # NOT what torch's F.interpolate(align_corners=False) computes (torch's
    # align_corners=False uses half-pixel centers, matching TF2's default
    # instead) -- confirmed against the original kernel, where e.g. an upsample
    # by 2 places the original samples exactly on even output indices. So
    # this is done by hand via explicit gather rather than F.interpolate.
    return _tf1_legacy_bilinear_resize(x, new_h, new_w)
nn.resize2d_bilinear = resize2d_bilinear


def _tf1_legacy_bilinear_resize(x, out_h, out_w):
    b, c, h, w = x.shape
    dtype, device = x.dtype, x.device
    scale_h, scale_w = h / out_h, w / out_w

    # The sampling grid is built in float32 regardless of x's dtype, then the
    # interpolation weights are cast down. TF computed resize coordinates in
    # float always, and under float16 torch.arange cannot represent output
    # indices above 2048 exactly, so floor() would yield *wrong gather indices*
    # -- not rounding error in the result but samples taken from the wrong
    # pixels (measured: 3 to 512 wrong indices depending on resolution, deltas
    # up to 4.8e-1). Unlike total_variation_mse's axes or DepthwiseConv2D's
    # dilations, this is not a TF quirk to preserve: it is a divergence from
    # TF introduced by the port.
    sy = torch.arange(out_h, dtype=torch.float32, device=device) * scale_h
    y0 = torch.floor(sy).long()
    y1 = torch.clamp(y0 + 1, max=h - 1)
    fy = (sy - y0.to(torch.float32)).to(dtype).view(1, 1, out_h, 1)
    y0 = torch.clamp(y0, 0, h - 1)

    sx = torch.arange(out_w, dtype=torch.float32, device=device) * scale_w
    x0 = torch.floor(sx).long()
    x1 = torch.clamp(x0 + 1, max=w - 1)
    fx = (sx - x0.to(torch.float32)).to(dtype).view(1, 1, 1, out_w)
    x0 = torch.clamp(x0, 0, w - 1)

    row = x.index_select(2, y0) * (1 - fy) + x.index_select(2, y1) * fy
    return row.index_select(3, x0) * (1 - fx) + row.index_select(3, x1) * fx


def resize2d_nearest(x, size=2):
    if size in [-1, 0, 1]:
        return x

    if size > 0:
        raise Exception("")

    return x[:, :, ::-size, ::-size]
nn.resize2d_nearest = resize2d_nearest


def space_to_depth(x, size):
    # WARNING: this is NOT F.pixel_unshuffle, for the mirror-image reason
    # depth_to_space below is not F.pixel_shuffle. TF composes the output
    # channel axis as (i, j, c) -- hence the permute putting the two block axes
    # ahead of c -- while pixel_unshuffle composes it as (c, i, j). The two
    # produce the same shape and a permutation of the same values, so a
    # substitution here would pass every shape assertion and silently break
    # compatibility with existing pretrained weights.
    b, c, h, w = x.shape
    oh, ow = h // size, w // size
    x = torch.reshape(x, (-1, c, oh, size, ow, size))
    x = x.permute(0, 3, 5, 1, 2, 4)
    return torch.reshape(x, (-1, c * size * size, oh, ow))
nn.space_to_depth = space_to_depth


def depth_to_space(x, size):
    # WARNING: this is NOT F.pixel_shuffle. TF decomposes the channel axis as
    # (i, j, c), torch's pixel_shuffle as (c, i, j). Substituting it would
    # break compatibility with existing pretrained weights.
    b, c, h, w = x.shape
    oh, ow = h * size, w * size
    oc = c // (size * size)

    x = torch.reshape(x, (-1, size, size, oc, h, w))
    x = x.permute(0, 3, 4, 1, 5, 2)
    return torch.reshape(x, (-1, oc, oh, ow))
nn.depth_to_space = depth_to_space


def gaussian_blur(input, radius=2.0):
    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

    def make_kernel(sigma):
        kernel_size = max(3, int(2 * 2 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
        kernel = np_kernel / np.sum(np_kernel)
        return kernel, kernel_size

    gauss_kernel, kernel_size = make_kernel(radius)
    padding = kernel_size // 2

    x  = input
    ch = x.shape[nn.conv2d_ch_axis]
    k  = torch.as_tensor(gauss_kernel, dtype=x.dtype, device=x.device)
    k  = k[None, None, :, :].repeat(ch, 1, 1, 1)              # (C,1,kh,kw)

    if padding != 0:
        x = F.pad(x, (padding, padding, padding, padding))
    return F.conv2d(x, k, stride=1, groups=ch)
nn.gaussian_blur = gaussian_blur


def style_loss(target, style, gaussian_blur_radius=0.0, loss_weight=1.0, step_size=1):
    # step_size is vestigial: the TF original never referenced it in the body
    # either, and no caller passes anything but the default. Kept because the
    # signature is public leras API -- do not wire it up to anything.
    def sd(content, style, loss_weight):
        content_nc = content.shape[nn.conv2d_ch_axis]
        style_nc   = style.shape[nn.conv2d_ch_axis]
        if content_nc != style_nc:
            raise Exception("style_loss() content_nc != style_nc")

        c_mean = torch.mean(content, dim=nn.conv2d_spatial_axes, keepdim=True)
        s_mean = torch.mean(style,   dim=nn.conv2d_spatial_axes, keepdim=True)
        c_var  = torch.var(content, dim=nn.conv2d_spatial_axes, keepdim=True, unbiased=False)
        s_var  = torch.var(style,   dim=nn.conv2d_spatial_axes, keepdim=True, unbiased=False)
        c_std, s_std = torch.sqrt(c_var + 1e-5), torch.sqrt(s_var + 1e-5)

        mean_loss = torch.sum(torch.square(c_mean - s_mean), dim=[1, 2, 3])
        std_loss  = torch.sum(torch.square(c_std - s_std),   dim=[1, 2, 3])
        return (mean_loss + std_loss) * (loss_weight / content_nc)

    if gaussian_blur_radius > 0.0:
        target = gaussian_blur(target, gaussian_blur_radius)
        style  = gaussian_blur(style,  gaussian_blur_radius)

    return sd(target, style, loss_weight=loss_weight)
nn.style_loss = style_loss


def dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if img1.dtype != img2.dtype:
        raise ValueError("img1.dtype != img2.dtype")

    not_float32 = img1.dtype != torch.float32
    if not_float32:
        img_dtype = img1.dtype
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

    filter_size = max(1, filter_size)

    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1) / 2.0
    kernel = kernel ** 2
    kernel *= (-0.5 / (filter_sigma ** 2))
    kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))
    kernel = torch.as_tensor(np.reshape(kernel, (1, -1)), dtype=torch.float32, device=img1.device)
    kernel = torch.softmax(kernel, dim=-1)
    kernel = torch.reshape(kernel, (1, 1, filter_size, filter_size))

    ch     = img1.shape[nn.conv2d_ch_axis]
    kernel = kernel.repeat(ch, 1, 1, 1)                        # (C,1,fs,fs)

    def reducer(x):
        return F.conv2d(x, kernel, stride=1, groups=ch)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0  = mean0 * mean1 * 2.0
    den0  = torch.square(mean0) + torch.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(torch.square(img1) + torch.square(img2))
    c2  *= 1.0  # compensation factor
    cs   = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim_val = torch.mean(luminance * cs, dim=nn.conv2d_spatial_axes)
    result   = (1.0 - ssim_val) / 2.0

    if not_float32:
        result = result.to(img_dtype)
    return result
nn.dssim = dssim


def sigmoid_cross_entropy_with_logits(labels, logits):
    """
    tf.nn.sigmoid_cross_entropy_with_logits: elemento per elemento, senza
    riduzione. I nomi degli argomenti sono quelli di TF, cosi' i chiamanti
    (models/Model_XSeg/Model.py) restano leggibili come prima.

    F.binary_cross_entropy_with_logits con reduction='none' calcola la stessa
    quantita' nella stessa forma stabile -- max(x,0) - x*z + log(1+exp(-|x|)) --
    e in particolare non passa mai da un sigmoid esplicito, che saturerebbe
    per logit grandi in valore assoluto.
    """
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
nn.sigmoid_cross_entropy_with_logits = sigmoid_cross_entropy_with_logits


def gelu(x):
    cdf = 0.5 * (1.0 + torch.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf
nn.gelu = gelu


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    if dtype is None:
        dtype = torch.float32

    gen = None
    if seed is not None:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))

    # NOTE: the TF original samples the uniform draw in float16
    # (random_ops.random_uniform(shape, dtype=tf.float16, seed=seed)), which
    # quantizes the comparison against p to float16 granularity. That is not
    # reproduced here -- float32 is used throughout -- since it only shifts
    # results at the level of individual coin flips near float16 ULPs, not a
    # behaviour anything downstream depends on.
    u = torch.rand(shape, generator=gen, dtype=torch.float32)
    return torch.where(u < p, torch.ones(shape, dtype=dtype), torch.zeros(shape, dtype=dtype))
nn.random_binomial = random_binomial


def rgb_to_lab(srgb):
    srgb_pixels      = torch.reshape(srgb, (-1, 3))
    linear_mask      = (srgb_pixels <= 0.04045).to(torch.float32)
    exponential_mask = (srgb_pixels >  0.04045).to(torch.float32)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + \
                 (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ], dtype=rgb_pixels.dtype, device=rgb_pixels.device)
    xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz)

    xyz_normalized_pixels = xyz_pixels * torch.tensor(
        [1 / 0.950456, 1.0, 1 / 1.088754], dtype=xyz_pixels.dtype, device=xyz_pixels.device)

    epsilon          = 6 / 29
    linear_mask      = (xyz_normalized_pixels <= (epsilon ** 3)).to(torch.float32)
    exponential_mask = (xyz_normalized_pixels >  (epsilon ** 3)).to(torch.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + \
                    (xyz_normalized_pixels ** (1 / 3)) * exponential_mask

    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0],  # fx
        [116.0, -500.0,  200.0],  # fy
        [  0.0,    0.0, -200.0],  # fz
    ], dtype=fxfyfz_pixels.dtype, device=fxfyfz_pixels.device)
    lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + \
                 torch.tensor([-16.0, 0.0, 0.0], dtype=fxfyfz_pixels.dtype,
                              device=fxfyfz_pixels.device)
    return torch.reshape(lab_pixels, srgb.shape)
nn.rgb_to_lab = rgb_to_lab


def total_variation_mse(images):
    """
    Same as generic total_variation, but MSE diff instead of MAE

    NOTE: axes 1 and 2 are indexed literally as in the TF original, which was
    written for NHWC. With NCHW tensors the first difference ends up along
    the channel axis, not the row axis. This is the behaviour every existing
    model was trained with, and it must not be "fixed".
    """
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

    # Reduced one axis at a time, innermost axis first, rather than a single
    # torch.sum(..., dim=[1,2,3]) call. Both compute the same mathematical
    # sum, but round differently in float32: the flattened form misses the
    # reference values (taken from TF on GPU) by ~1.2e-4 on a value around 810,
    # just over atol=1e-6+rtol=1e-7*|value|; this nested form lands within
    # ~6.1e-5, inside tolerance.
    #
    # EMPIRICAL, not verified against TF's reduce_sum source: TF is not
    # available in this environment, so this is inferred, not read out of
    # TF's kernel. The working theory is that TF-GPU's multi-axis reduce_sum
    # reduces the memory-contiguous (last) axis first, rather than flattening
    # all reduced axes into one pass the way torch.sum(dim=[...]) does, and
    # that running the same op on a local GPU (a different architecture than
    # the one the reference values came from) landed on nearly those exact
    # values supports that theory -- but it is still a theory fitted to a
    # single case, not a guarantee that holds for every input. If a future
    # measurement makes this fail, don't assume the theory still holds;
    # re-derive from scratch.
    def _reduce_hwc(t):
        return torch.sum(torch.sum(torch.sum(t, dim=-1), dim=-1), dim=-1)

    tot_var = _reduce_hwc(torch.square(pixel_dif1)) + _reduce_hwc(torch.square(pixel_dif2))
    return tot_var
nn.total_variation_mse = total_variation_mse


def pixel_norm(x, axes):
    return x * torch.rsqrt(torch.mean(torch.square(x), dim=axes, keepdim=True) + 1e-06)
nn.pixel_norm = pixel_norm


def _get_pixel_value(img, x, y):
    batch_size, height, width = x.shape[0], x.shape[1], x.shape[2]

    b = torch.arange(0, batch_size, device=img.device).reshape(batch_size, 1, 1)
    b = b.expand(batch_size, height, width)

    return img[b, y, x]


def bilinear_sampler(img, x, y):
    # NOTE: unlike everything else in this file, this works in NHWC. Its
    # only caller (TanhPolar) transposes around the call; do not "correct"
    # it to NCHW.
    H_MAX = img.shape[1] - 1
    W_MAX = img.shape[2] - 1

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.floor(x).to(torch.int64)
    x1 = x0 + 1
    y0 = torch.floor(y).to(torch.int64)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, 0, W_MAX)
    x1 = torch.clamp(x1, 0, W_MAX)
    y0 = torch.clamp(y0, 0, H_MAX)
    y1 = torch.clamp(y1, 0, H_MAX)

    # get pixel value at corner coords
    Ia = _get_pixel_value(img, x0, y0)
    Ib = _get_pixel_value(img, x0, y1)
    Ic = _get_pixel_value(img, x1, y0)
    Id = _get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = x0.to(img.dtype)
    x1 = x1.to(img.dtype)
    y0 = y0.to(img.dtype)
    y1 = y1.to(img.dtype)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return (wa[..., None] * Ia + wb[..., None] * Ib +
            wc[..., None] * Ic + wd[..., None] * Id)
nn.bilinear_sampler = bilinear_sampler
