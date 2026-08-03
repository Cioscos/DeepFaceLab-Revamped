import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


class TanhPolar(nn.LayerBase):
    """
    RoI Tanh-polar Transformer Network for Face Parsing in the Wild
    https://github.com/hhj1897/roi_tanh_warping
    """

    def __init__(self, width, height, angular_offset_deg=270, **kwargs):
        self.width = width
        self.height = height

        warp_gridx, warp_gridy = TanhPolar._get_tanh_polar_warp_grids(width,height,angular_offset_deg=angular_offset_deg)
        restore_gridx, restore_gridy = TanhPolar._get_tanh_polar_restore_grids(width,height,angular_offset_deg=angular_offset_deg)

        self.warp_gridx = warp_gridx[None, ...]
        self.warp_gridy = warp_gridy[None, ...]
        self.restore_gridx = restore_gridx[None, ...]
        self.restore_gridy = restore_gridy[None, ...]

        super().__init__(**kwargs)

    def build_weights(self):
        # tf.constant in the original: fixed sampling-grid tables derived
        # only from width/height/angular_offset_deg, recomputable and never
        # part of the on-disk weight file -- register_buffer, not Parameter.
        self.register_buffer('warp_gridx_t', torch.as_tensor(self.warp_gridx))
        self.register_buffer('warp_gridy_t', torch.as_tensor(self.warp_gridy))
        self.register_buffer('restore_gridx_t', torch.as_tensor(self.restore_gridx))
        self.register_buffer('restore_gridy_t', torch.as_tensor(self.restore_gridy))

    def warp(self, inp_t):
        batch = inp_t.shape[0]
        warp_gridx_t = self.warp_gridx_t.repeat(batch, 1, 1)
        warp_gridy_t = self.warp_gridy_t.repeat(batch, 1, 1)

        # nn.bilinear_sampler works in NHWC -- keep the transpose around the
        # call rather than "correcting" the sampler (see its docstring).
        inp_t = inp_t.permute(0, 2, 3, 1)

        out_t = nn.bilinear_sampler(inp_t, warp_gridx_t, warp_gridy_t)

        out_t = out_t.permute(0, 3, 1, 2)

        return out_t

    def restore(self, inp_t):
        batch = inp_t.shape[0]
        restore_gridx_t = self.restore_gridx_t.repeat(batch, 1, 1)
        restore_gridy_t = self.restore_gridy_t.repeat(batch, 1, 1)

        # Original pads in NHWC with tf.pad(..., [(0,0),(1,1),(1,0),(0,0)],
        # "SYMMETRIC") after the transpose below. Padding H/W is independent
        # of channel position, so doing it here in NCHW (H,W last) is
        # equivalent and is what F.pad's spatial-only modes require.
        # mode='replicate' stands in for TF's "SYMMETRIC": every non-zero
        # width here is 1, and mirroring one element while including the
        # boundary is definitionally the same as duplicating the edge
        # element. It stops holding for pad widths > 1.
        inp_t = F.pad(inp_t, (1, 0, 1, 1), mode='replicate')

        inp_t = inp_t.permute(0, 2, 3, 1)

        out_t = nn.bilinear_sampler(inp_t, restore_gridx_t, restore_gridy_t)

        out_t = out_t.permute(0, 3, 1, 2)

        return out_t

    @staticmethod
    def _get_tanh_polar_warp_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)
        normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / W),np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / H)), axis=-1)
        radii = normalised_dest_indices[..., 0]
        orientation_x = np.cos(normalised_dest_indices[..., 1])
        orientation_y = np.sin(normalised_dest_indices[..., 1])

        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
        src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    @staticmethod
    def _get_tanh_polar_restore_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)

        dest_indices = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(float)
        normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                                [sin_offset, cos_offset]]))
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                        roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]

        src_radii = np.tanh(radii)


        src_x_indices = src_radii * W + 1.0
        src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                                2.0 / np.pi) * H, H) + 1.0

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)


nn.TanhPolar = TanhPolar
