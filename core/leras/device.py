import sys

import torch


class Device(object):
    def __init__(self, index, name, total_mem, free_mem):
        self.index         = index
        self.name          = name

        self.total_mem     = total_mem
        self.total_mem_gb  = total_mem / 1024**3
        self.free_mem      = free_mem
        self.free_mem_gb   = free_mem / 1024**3

    def __str__(self):
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb :.3}]"

class Devices(object):
    all_devices = None

    def __init__(self, devices):
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, key):
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        for device in self.devices:
            yield device

    def get_best_device(self):
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_device_by_index(self, idx):
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append (device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb*(1024**3):
                result.append (device)
        return Devices(result)

    @staticmethod
    def initialize_main_env():
        # In TF's graph mode this needed a spawned subprocess to import TF and
        # read device_lib. torch enumerates the GPUs in the current process with
        # no side effects, so invalidating the cache is enough here: getDevices()
        # recomputes it on the next call.
        Devices.all_devices = None

    @staticmethod
    def getDevices():
        if Devices.all_devices is None:
            devices = []
            for i in range(torch.cuda.device_count()):
                props    = torch.cuda.get_device_properties(i)
                free_mem = torch.cuda.mem_get_info(i)[0]
                devices.append (Device(index=i,
                                        name=props.name,
                                        total_mem=props.total_memory,
                                        free_mem=free_mem))
            Devices.all_devices = Devices(devices)

        return Devices.all_devices
