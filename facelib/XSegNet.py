from pathlib import Path

import numpy as np
import torch

from core.interact import interact as io
from core.leras import nn


class XSegNet(object):
    VERSION = 1

    def __init__ (self, name,
                        resolution=256,
                        load_weights=True,
                        weights_file_root=None,
                        training=False,
                        place_model_on_cpu=False,
                        run_on_cpu=False,
                        optimizer=None,
                        data_format="NHWC",
                        raise_on_no_model_files=False):

        self.resolution = resolution
        self.weights_file_root = Path(weights_file_root) if weights_file_root is not None else Path(__file__).parent

        nn.initialize()

        # `data_format` is stored and never read. net_run/extract always treat
        # numpy in/out as NHWC regardless of this value -- it does not select
        # anything. mainscripts/XSegUtil.py:56 and models/Model_XSeg/Model.py
        # both pass nn.data_format, which is unconditionally "NCHW" now that
        # core/leras is NCHW-only; XSegUtil still feeds net_run genuinely NHWC
        # arrays, and that only works because this field is inert.
        # `flow()` below bypasses the wrapper's transpose and hands the torch
        # model NCHW tensors directly. Its caller is Model_XSeg's training
        # step, which is torch and NCHW, and builds its
        # tensors from numpy that samplelib already produced in nn.data_format.
        # So the two callers pass this field the same value and differ in the
        # layout of the data they actually feed -- net_run genuine NHWC,
        # flow() genuine NCHW -- which is consistent only because each entry
        # point fixes its own layout and none of them consults this field.
        # Left inert rather than asserting
        # data_format=="NHWC" here: that assert is the textbook fix but it
        # would raise on both live callers, which pass "NCHW" and are correct.
        self.data_format = data_format

        # TF placed variables and operations with two independent tf.device
        # scopes: place_model_on_cpu kept the weights in host memory while the
        # ops still ran on the GPU. A torch module's parameters live wherever
        # it runs, so the two scopes cannot be kept independent; we chose to
        # preserve the execution device and let place_model_on_cpu become a
        # no-op for inference placement (it still gates vars_on_cpu for the
        # optimizer below, which is unaffected). The cost: mainscripts/Merger.py
        # hardcodes place_model_on_cpu=True, so under TF the weights lived off
        # the GPU there; under torch they no longer do, i.e. more VRAM used
        # than before -- the opposite of what an earlier version of this
        # comment claimed.
        self.device = torch.device('cpu') if run_on_cpu else nn.device

        model_name = f'{name}_{resolution}'
        self.model_filename_list = []

        self.model = nn.XSeg(3, 32, 1, name=name)
        self.model.build()
        self.model.to(self.device)
        self.model_weights = self.model.get_weights()

        if training:
            if optimizer is None:
                raise ValueError("Optimizer should be provided for training mode.")
            self.opt = optimizer
            # optimizer_weights(), non get_weights(): initialize_variables ha
            # bisogno delle quadruple (name, param, owner, param_path) -- il
            # nome scope-qualificato per la chiave su disco e il proprietario
            # per il layout dell'accumulatore. Il TF originale passava
            # get_weights() perche' una tf.Variable porta il proprio `.name`
            # addosso; un Parameter torch no.
            self.opt.initialize_variables (self.model.optimizer_weights(), vars_on_cpu=place_model_on_cpu)
            self.model_filename_list += [ [self.opt, f'{model_name}_opt.npy' ] ]

        self.model_filename_list += [ [self.model, f'{model_name}.npy'] ]

        self.initialized = True
        # Loading/initializing all models/optimizers weights
        for model, filename in self.model_filename_list:
            do_init = not load_weights

            if not do_init:
                model_file_path = self.weights_file_root / filename
                do_init = not model.load_weights( model_file_path )
                if do_init:
                    if raise_on_no_model_files:
                        raise Exception(f'{model_file_path} does not exists.')
                    if not training:
                        self.initialized = False
                        break

            if do_init:
                model.init_weights()

    def get_resolution(self):
        return self.resolution

    def flow(self, x, pretrain=False):
        return self.model(x, pretrain=pretrain)

    def get_weights(self):
        return self.model_weights

    def save_weights(self):
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Saving", leave=False):
            model.save_weights( self.weights_file_root / filename )

    def net_run(self, input_np):
        """
        NHWC numpy in, NHWC numpy out.

        The transpose lives here rather than in core/leras: leras is NCHW-only,
        and this wrapper is the boundary where cv2's layout meets it.
        """
        x = torch.as_tensor(np.ascontiguousarray(input_np)).to(self.device, nn.floatx)
        with torch.no_grad():
            _, pred = self.model(x.permute(0, 3, 1, 2))
        return pred.permute(0, 2, 3, 1).cpu().numpy()

    def extract (self, input_image):
        if not self.initialized:
            # nn.floatx is a torch.dtype now, so the TF `.as_numpy_dtype` is
            # gone. This is the only call site of it in live code.
            np_floatx = np.float16 if nn.floatx == torch.float16 else np.float32
            return 0.5*np.ones ( (self.resolution, self.resolution, 1), np_floatx )

        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None,...]

        result = np.clip ( self.net_run(input_image), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result
