import math

import torch

from core.leras import nn


class RMSprop(nn.OptimizerBase):
    def __init__(self, lr=0.001, rho=0.9, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None, **kwargs):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr_dropout = lr_dropout
        self.lr_cos     = lr_cos
        self.lr         = lr
        self.rho        = rho
        self.clipnorm   = clipnorm

        # See AdaBelief.py for why the attribute is `iters` (not
        # `iterations`) and why the shape is (1,) (not ()) -- the latter is
        # load-bearing: Saveable.load_weights would raise on any 0-d
        # parameter, this class's own saved files included.
        self.iters = torch.nn.Parameter(torch.zeros(1, dtype=torch.int64), requires_grad=False)

        # Named `acc` (not `accumulators_dict`) so torch_name_to_disk_key's
        # container-join produces the TF `f'acc_{v.name}'.replace(':','_')`
        # disk key from self.acc['encoder/weight_0'] -- see AdaBelief.py.
        self.acc = torch.nn.ParameterDict()

        # id(param) -> (disk key, param); see AdaBelief.py for why the
        # strong reference to the parameter matters (avoids an id() reuse
        # collision misattributing accumulators after a parameter is freed).
        self._keys = {}

        # See AdaBelief.py: the device of the lr_dropout mask draw in
        # step() -- False samples on the weight's own device, True on the
        # host, mirroring the TF graph placement of the sampling op.
        self._lr_dropout_on_cpu = False

    @property
    def iterations(self):
        return self.iters

    #override
    def get_weights(self):
        # Stable order the optimizer itself defines: [iterations, acc].
        return [self.iters] + list(self.acc.values())

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        # See AdaBelief.initialize_variables for the
        # (name, parameter, owner, param_path) contract.
        self._lr_dropout_on_cpu = lr_dropout_on_cpu

        for name, v, owner, param_path in trainable_weights:
            key    = f'{name}:0'.replace(':', '_')
            device = torch.device('cpu') if vars_on_cpu else v.device

            self.acc[key] = torch.nn.Parameter(
                torch.zeros(v.shape, dtype=v.dtype, device=device), requires_grad=False)

            self._keys[id(v)]  = (key, v)
            self._mirrors[key] = (owner, param_path)

    def step(self, grads_vars):
        """
        Apply the update immediately. See AdaBelief.step for the
        `get_update_op` -> `step` rationale and the iterations
        increment-then-read convention (identical here).
        """
        with torch.no_grad():
            if self.clipnorm > 0.0:
                norm = torch.sqrt(sum(torch.sum(torch.square(g.to(torch.float32)))
                                       for g, v in grads_vars))

            self.iters += 1

            for g, v in grads_vars:
                key, registered_v = self._keys[id(v)]
                assert registered_v is v, (
                    "id(v) collision in RMSprop._keys: this parameter was "
                    "not the one registered under this id() by "
                    "initialize_variables(). Was step() called with a "
                    "parameter that was never passed to "
                    "initialize_variables()?")

                if self.clipnorm > 0.0:
                    g = self.clip_norm(g, self.clipnorm, norm.to(g.dtype))

                a = self.acc[key]

                g_c = g.to(dtype=a.dtype, device=v.device)
                a_c = a.to(device=v.device)

                new_a = self.rho * a_c + (1.0 - self.rho) * torch.square(g_c)

                lr = torch.as_tensor(self.lr, dtype=g_c.dtype, device=v.device)
                if self.lr_cos != 0:
                    # iters lives on the CPU; read it on the weight's device
                    # -- see AdaBelief.step.
                    lr = lr * (torch.cos(self.iters.to(dtype=g_c.dtype, device=v.device) *
                                          (2 * math.pi / float(self.lr_cos))) + 1.0) / 2.0

                v_diff = -lr * g_c / (torch.sqrt(new_a) + torch.finfo(g_c.dtype).resolution)
                if self.lr_dropout != 1.0:
                    # Resampled every step, drawn on the weight's device
                    # unless lr_dropout_on_cpu asked for the host -- see
                    # AdaBelief.step.
                    mask_device = torch.device('cpu') if self._lr_dropout_on_cpu else v.device
                    mask = nn.random_binomial(v.shape, p=self.lr_dropout, dtype=v.dtype,
                                              device=mask_device)
                    v_diff = v_diff * mask.to(device=v.device, dtype=v_diff.dtype)

                a.copy_(new_a.to(dtype=a.dtype, device=a.device))
                v.add_(v_diff.to(dtype=v.dtype))


nn.RMSprop = RMSprop
