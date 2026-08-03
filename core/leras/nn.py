"""
Leras.

like lighter keras.
A lightweight neural network library built on top of PyTorch.

The classes in here derive from torch.nn.Module: leras provides the API
(build_weights/forward/get_weights) and the on-disk weight format, while
execution, autograd and device handling are torch's.

NCHW is the only supported data_format: it is torch's native layout.
"""
import numpy as np
import torch

from core.interact import interact as io

from .device import Devices


class nn():
    current_DeviceConfig = None

    # Set by initialize() and read by NOTHING inside the package. No layer
    # passes device= to torch.empty/zeros/ones, so every parameter and buffer is
    # created on the CPU and placement is left entirely to the caller's
    # `module.to(device)`. That is the deliberate Phase 1 scope, but it is a
    # decision worth stating: a new layer copied from Conv2D inherits the CPU
    # default without knowing it. Note also that a plain tensor attribute (the
    # `wscale` in Conv2D/Conv2DTranspose/DepthwiseConv2D/Dense) is neither a
    # Parameter nor a buffer, so `module.to(device)` would not move it either --
    # harmless today only because it is 0-dim and torch special-cases 0-dim CPU
    # operands. Whoever wires up real device placement has to handle both.
    device              = None

    data_format         = "NCHW"
    conv2d_ch_axis      = 1
    conv2d_spatial_axes = [2, 3]

    floatx              = None

    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NCHW"):
        if device_config is None:
            device_config = nn.getCurrentDeviceConfig()
        nn.setCurrentDeviceConfig(device_config)

        if len(device_config.devices) == 0:
            nn.device = torch.device('cpu')
        else:
            nn.device = torch.device(f'cuda:{device_config.devices[0].index}')

        if floatx == "float32":
            floatx = torch.float32
        elif floatx == "float16":
            floatx = torch.float16
        else:
            raise ValueError(f"unsupported floatx {floatx}")
        nn.set_floatx(floatx)
        nn.set_data_format(data_format)

    @staticmethod
    def initialize_main_env():
        Devices.initialize_main_env()

    @staticmethod
    def set_floatx(torch_dtype):
        """
        set default float type for all layers when dtype is None for them
        """
        nn.floatx = torch_dtype

    @staticmethod
    def set_data_format(data_format):
        if data_format != "NCHW":
            raise ValueError(f"unsupported data_format {data_format}: only NCHW is supported")
        nn.data_format = data_format

    @staticmethod
    def get4Dshape ( w, h, c ):
        """
        returns 4D shape based on current data_format
        """
        return (None,c,h,w)

    @staticmethod
    def to_data_format( x, to_data_format, from_data_format):
        if to_data_format == from_data_format:
            return x

        if to_data_format == "NHWC":
            return np.transpose(x, (0,2,3,1) )
        elif to_data_format == "NCHW":
            return np.transpose(x, (0,3,1,2) )
        else:
            raise ValueError(f"unsupported to_data_format {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = nn.DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        nn.current_DeviceConfig = device_config

    @staticmethod
    def release_session():
        """
        Release GPU memory between one model and the next.
        Replaces reset_session()/close_session(), which existed to tear down the
        TF graph: in eager mode there is no graph to tear down.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
        devices = Devices.getDevices()
        if len(devices) == 0:
            return []

        all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False

        if suggest_all_gpu:
            best_device_indexes = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device()) ]
        else:
            best_device_indexes = [ devices.get_best_device().index ]
        best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        io.log_info ("")
        if choose_only_one:
            io.log_info ("Choose one GPU idx.")
        else:
            io.log_info ("Choose one or several GPU idxs (separated by comma).")
        io.log_info ("")

        if allow_cpu:
            io.log_info ("[CPU] : CPU")
        for device in devices:
            io.log_info (f"  [{device.index}] : {device.name}")

        io.log_info ("")

        while True:
            try:
                if choose_only_one:
                    choosed_idxs = io.input_str("Which GPU index to choose?", best_device_indexes)
                else:
                    choosed_idxs = io.input_str("Which GPU indexes to choose?", best_device_indexes)

                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break

                choosed_idxs = [ int(x) for x in choosed_idxs.split(',') ]

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all( [idx in all_devices_indexes for idx in choosed_idxs] ):
                        break
            except:
                pass
        io.log_info ("")

        return choosed_idxs

    class DeviceConfig():
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            return nn.DeviceConfig.GPUIndexes( nn.ask_choose_device_idxs(*args,**kwargs) )

        def __init__ (self, devices=None):
            devices = devices or []

            if not isinstance(devices, Devices):
                devices = Devices(devices)

            self.devices = devices
            self.cpu_only = len(devices) == 0

        @staticmethod
        def BestGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_best_device()])

        @staticmethod
        def WorstGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_worst_device()])

        @staticmethod
        def GPUIndexes(indexes):
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []

            return nn.DeviceConfig(devices)

        @staticmethod
        def CPU():
            return nn.DeviceConfig([])
