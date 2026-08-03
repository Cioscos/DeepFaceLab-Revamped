import numpy as np
import torch
import torch.nn.functional as F

from core.leras import nn


class BlurPool(nn.LayerBase):
    def __init__(self, filt_size=3, stride=2, **kwargs ):

        self.strides = stride

        self.filt_size = filt_size
        pad = [ int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ]

        # TF's nested-list padding [[0,0],[0,0],pad,pad] (NCHW), flattened to
        # F.pad's (left, right, top, bottom) form. pad[0] != pad[1] for even
        # filt_size (XSeg uses filt_size=4), so the two halves must stay
        # distinct here, not collapsed into a single symmetric value.
        self.padding = (pad[0], pad[1], pad[0], pad[1])

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a[:,None]*a[None,:]
        a = a / np.sum(a)
        a = a[:,:,None,None]
        self.a = a
        super().__init__(**kwargs)

    def build_weights(self):
        # tf.constant in the original: a recomputable constant, never part
        # of the on-disk weight file. register_buffer, not Parameter -- see
        # Saveable's docstring.
        self.register_buffer('k', torch.as_tensor(self.a, dtype=nn.floatx))

    def forward(self, x):
        # int() e non la dimensione tracciata: sotto torch.onnx.export
        # x.shape[1] e' simbolica, il kernel repeat()-ato eredita una forma
        # ignota e l'export della convoluzione fallisce con
        # "ONNX export of convolution for kernel of unknown shape".
        # Il canale non e' mai un asse dinamico -- l'unico dichiarato tale in
        # export_dfm e' il batch -- quindi congelarlo e' corretto, e in eager
        # int() di un int e' l'identita'. Senza questa riga XSeg non si
        # esporta affatto: e' l'unica rete della migrazione che usa BlurPool.
        ch = int(x.shape[nn.conv2d_ch_axis])
        k  = self.k.permute(2, 3, 0, 1).repeat(ch, 1, 1, 1)      # (kh,kw,1,1) -> (C,1,kh,kw)
        x  = F.pad(x, self.padding)
        return F.conv2d(x, k, stride=self.strides, groups=ch)
nn.BlurPool = BlurPool
