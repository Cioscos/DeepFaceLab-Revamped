import operator
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


class S3FDExtractor(object):
    def __init__(self, place_model_on_cpu=False):
        nn.initialize()

        model_path = Path(__file__).parent / "S3FD.npy"
        if not model_path.exists():
            raise Exception("Unable to load S3FD.npy")

        class L2Norm(nn.LayerBase):
            def __init__(self, n_channels, **kwargs):
                self.n_channels = n_channels
                super().__init__(**kwargs)

            def build_weights(self):
                # (1,C,1,1), not (C,): it is what the TF variable's (1,1,1,C)
                # becomes under the NHWC->NCHW permutation, it broadcasts
                # against NCHW without a reshape in forward, and it keeps the
                # on-disk shape exact through a save/load round-trip.
                self.weight = torch.nn.Parameter(
                    torch.ones((1, self.n_channels, 1, 1), dtype=nn.floatx))

            def weight_layouts(self):
                return {"weight": (0, 3, 1, 2)}        # (1,1,1,C) -> (1,C,1,1)

            def forward(self, x):
                norm = torch.sqrt(torch.sum(x ** 2, dim=nn.conv2d_ch_axis, keepdim=True))
                return x / (norm + 1e-10) * self.weight

        class S3FD(nn.ModelBase):
            def __init__(self):
                super().__init__(name='S3FD')

            def on_build(self):
                # A tf.constant, so a buffer: as a Parameter it would add a
                # 66th key to the 65-key file on disk. (1,3,1,1) is the NCHW
                # broadcast shape of the TF (3,) constant against NHWC.
                self.register_buffer('minus',
                    torch.tensor([104, 117, 123], dtype=nn.floatx).reshape(1, 3, 1, 1))
                self.conv1_1 = nn.Conv2D(3, 64, kernel_size=3, strides=1, padding='SAME')
                self.conv1_2 = nn.Conv2D(64, 64, kernel_size=3, strides=1, padding='SAME')

                self.conv2_1 = nn.Conv2D(64, 128, kernel_size=3, strides=1, padding='SAME')
                self.conv2_2 = nn.Conv2D(128, 128, kernel_size=3, strides=1, padding='SAME')

                self.conv3_1 = nn.Conv2D(128, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_2 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3 = nn.Conv2D(256, 256, kernel_size=3, strides=1, padding='SAME')

                self.conv4_1 = nn.Conv2D(256, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.conv5_1 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_2 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3 = nn.Conv2D(512, 512, kernel_size=3, strides=1, padding='SAME')

                self.fc6 = nn.Conv2D(512, 1024, kernel_size=3, strides=1, padding=3)
                self.fc7 = nn.Conv2D(1024, 1024, kernel_size=1, strides=1, padding='SAME')

                self.conv6_1 = nn.Conv2D(1024, 256, kernel_size=1, strides=1, padding='SAME')
                self.conv6_2 = nn.Conv2D(256, 512, kernel_size=3, strides=2, padding='SAME')

                self.conv7_1 = nn.Conv2D(512, 128, kernel_size=1, strides=1, padding='SAME')
                self.conv7_2 = nn.Conv2D(128, 256, kernel_size=3, strides=2, padding='SAME')

                self.conv3_3_norm = L2Norm(256)
                self.conv4_3_norm = L2Norm(512)
                self.conv5_3_norm = L2Norm(512)


                self.conv3_3_norm_mbox_conf = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')
                self.conv3_3_norm_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv4_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv4_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv5_3_norm_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv5_3_norm_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.fc7_mbox_conf = nn.Conv2D(1024, 2, kernel_size=3, strides=1, padding='SAME')
                self.fc7_mbox_loc = nn.Conv2D(1024, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv6_2_mbox_conf = nn.Conv2D(512, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv6_2_mbox_loc = nn.Conv2D(512, 4, kernel_size=3, strides=1, padding='SAME')

                self.conv7_2_mbox_conf = nn.Conv2D(256, 2, kernel_size=3, strides=1, padding='SAME')
                self.conv7_2_mbox_loc = nn.Conv2D(256, 4, kernel_size=3, strides=1, padding='SAME')

            def forward(self, x):
                x = x - self.minus
                x = F.relu(self.conv1_1(x))
                x = F.relu(self.conv1_2(x))
                x = F.max_pool2d(x, 2, 2)          # tf.nn.max_pool VALID 2x2

                x = F.relu(self.conv2_1(x))
                x = F.relu(self.conv2_2(x))
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.conv3_1(x))
                x = F.relu(self.conv3_2(x))
                x = F.relu(self.conv3_3(x))
                f3_3 = x
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.conv4_1(x))
                x = F.relu(self.conv4_2(x))
                x = F.relu(self.conv4_3(x))
                f4_3 = x
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.conv5_1(x))
                x = F.relu(self.conv5_2(x))
                x = F.relu(self.conv5_3(x))
                f5_3 = x
                x = F.max_pool2d(x, 2, 2)

                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                ffc7 = x

                x = F.relu(self.conv6_1(x))
                x = F.relu(self.conv6_2(x))
                f6_2 = x

                x = F.relu(self.conv7_1(x))
                x = F.relu(self.conv7_2(x))
                f7_2 = x

                f3_3 = self.conv3_3_norm(f3_3)
                f4_3 = self.conv4_3_norm(f4_3)
                f5_3 = self.conv5_3_norm(f5_3)

                cls1 = self.conv3_3_norm_mbox_conf(f3_3)
                reg1 = self.conv3_3_norm_mbox_loc(f3_3)

                cls2 = torch.softmax(self.conv4_3_norm_mbox_conf(f4_3), dim=nn.conv2d_ch_axis)
                reg2 = self.conv4_3_norm_mbox_loc(f4_3)

                cls3 = torch.softmax(self.conv5_3_norm_mbox_conf(f5_3), dim=nn.conv2d_ch_axis)
                reg3 = self.conv5_3_norm_mbox_loc(f5_3)

                cls4 = torch.softmax(self.fc7_mbox_conf(ffc7), dim=nn.conv2d_ch_axis)
                reg4 = self.fc7_mbox_loc(ffc7)

                cls5 = torch.softmax(self.conv6_2_mbox_conf(f6_2), dim=nn.conv2d_ch_axis)
                reg5 = self.conv6_2_mbox_loc(f6_2)

                cls6 = torch.softmax(self.conv7_2_mbox_conf(f7_2), dim=nn.conv2d_ch_axis)
                reg6 = self.conv7_2_mbox_loc(f7_2)

                # max-out background label
                bmax = torch.maximum(torch.maximum(cls1[:, 0:1], cls1[:, 1:2]), cls1[:, 2:3])

                cls1 = torch.cat([bmax, cls1[:, 3:4]], dim=nn.conv2d_ch_axis)
                cls1 = torch.softmax(cls1, dim=nn.conv2d_ch_axis)

                return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

        # TF wrapped only S3FD()+load_weights in the CPU device scope, so
        # place_model_on_cpu kept the weights in host memory while
        # build_for_run's convolutions still ran on the GPU. A torch module's
        # parameters live wherever it runs, so that split cannot be
        # reproduced: this network runs entirely on `device`, weights
        # included. We deliberately kept CPU rather than matching TF's GPU
        # compute here, because place_model_on_cpu is set from
        # `cpu_only or total_mem_gb < 4` (mainscripts/Extractor.py) -- moving
        # ~90MB of resident weights onto a sub-4GB card risks turning "slow
        # extraction" into an OOM with no extraction at all. This is a
        # deliberate divergence from TF's behaviour, not an oversight.
        self.device = torch.device('cpu') if place_model_on_cpu else nn.device

        self.model = S3FD()
        self.model.build()
        self.model.to(self.device)
        self.model.load_weights(model_path)

    def _net_run(self, img):
        """
        NHWC numpy in, twelve NHWC numpy arrays out, batch axis kept.

        refine() indexes these as [h, w, c] and unpacks the batch itself, so
        the NCHW->NHWC transpose belongs here: core/leras is NCHW-only.
        """
        x = torch.as_tensor(np.ascontiguousarray(img)).to(self.device, nn.floatx)
        with torch.no_grad():
            outs = self.model(x.permute(0, 3, 1, 2))
        return [o.permute(0, 2, 3, 1).cpu().numpy() for o in outs]

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, is_bgr=True, is_remove_intersects=False):

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize (input_image, ( int(w/input_scale), int(h/input_scale) ), interpolation=cv2.INTER_LINEAR)

        olist = self._net_run(input_image[None,...])

        detected_faces = []
        for ltrb in self.refine (olist):
            l,t,r,b = [ x*input_scale for x in ltrb]
            bt = b-t
            if min(r-l,bt) < 40: #filtering faces < 40pix by any side
                continue
            b += bt*0.1 #enlarging bottom line a bit for 2DFAN-4, because default is not enough covering a chin
            detected_faces.append ( [int(x) for x in (l,t,r,b) ] )

        #sort by largest area first
        detected_faces = [ [(l,t,r,b), (r-l)*(b-t) ]  for (l,t,r,b) in detected_faces ]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True )
        detected_faces = [ x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1 = detected_faces[i]
                l0,t0,r0,b0 = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)

        return detected_faces

    def refine(self, olist):
        bboxlist = []
        for i, ((ocls,), (oreg,)) in enumerate ( zip ( olist[::2], olist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls[...,1] > 0.05)):
                score = ocls[hindex, windex, 1]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        # np.int was removed in numpy 2 (it was only ever an alias of the
        # builtin, so this is identical in behaviour).
        bboxlist = [ x[:-1].astype(int) for x in bboxlist if x[-1] >= 0.5]
        return bboxlist

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
