import math

import torch

from core.leras import nn


class AdaBelief(nn.OptimizerBase):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None, **kwargs):
        super().__init__(name=name)

        if name is None:
            raise ValueError('name must be defined.')

        self.lr         = lr
        self.beta_1     = beta_1
        self.beta_2     = beta_2
        self.lr_dropout = lr_dropout
        self.lr_cos     = lr_cos
        self.clipnorm   = clipnorm

        # Registered as `iters` (not `iterations`): the TF build wrote this
        # variable's name (`iters:0`, no scope prefix -- Saveable strips the
        # optimizer's own scope on save) to disk, and the on-disk format is
        # frozen. `iterations` below is a read-only convenience alias.
        #
        # Shape (1,), not (): Saveable.load_weights runs every loaded array
        # through np.ascontiguousarray, which promotes a 0-d array to shape
        # (1,) unconditionally (numpy's ascontiguousarray is documented to
        # produce "at least 1-d"). copy_-ing that (1,) result onto a 0-d
        # parameter raises `RuntimeError: output with shape [] doesn't match
        # the broadcast shape [1]` -- on every optimizer file, including one
        # this class wrote itself. A (1,) parameter sidesteps it entirely:
        # loading a genuinely 0-d array (e.g. a pre-existing TensorFlow-era
        # file, which wrote iters as a bare scalar) hits Saveable's
        # squeeze-shape legacy-reshape path (squeeze(()) == squeeze((1,)) ==
        # ()), so those files still load correctly.
        self.iters = torch.nn.Parameter(torch.zeros(1, dtype=torch.int64), requires_grad=False)

        # ms/vs are ParameterDicts named after the TF accumulator prefixes
        # themselves (`ms`/`vs`), *not* `ms_dict`/`vs_dict`: torch_name_to_disk_key
        # joins a container's own attribute name with its element key using
        # '_', so self.ms['encoder/weight_0'] serialises to the disk key
        # 'ms_encoder/weight_0:0' -- exactly the TF `f'ms_{v.name}'.replace(':','_')`
        # mangling, reproduced without hand-building the 'ms_' prefix twice.
        self.ms = torch.nn.ParameterDict()
        self.vs = torch.nn.ParameterDict()

        # id(param) -> (disk key, param), so step() can find each
        # parameter's accumulators without relying on grads_vars being in
        # the exact same order initialize_variables() was called with.
        # The tuple keeps a *strong* reference to the parameter itself, not
        # just its id(): id() is a memory address, and if nothing else kept
        # the parameter alive, CPython is free to garbage-collect it and
        # later reuse that exact address for an unrelated object, which
        # would make step() silently apply some other parameter's
        # accumulators to it -- precisely the wrong-accumulator failure mode
        # this whole task exists to prevent. Holding the reference here
        # makes that collision impossible for as long as this optimizer is
        # alive; the identity check in step() turns it into a loud
        # AssertionError instead of silent corruption if it ever did happen.
        self._keys = {}

        # lr_dropout_on_cpu placed the random_uniform sampling *op* on CPU
        # in the TF graph (as opposed to vars_on_cpu, which places the
        # accumulator *storage*); it did not fix the sampled mask, which was
        # still resampled every sess.run. step() honours it the same way, as
        # the device of the lr_dropout mask draw: False samples on the
        # weight's own device, True samples on the host -- no device memory
        # for the draw, at the price of a host->device copy of the mask per
        # parameter per step.
        self._lr_dropout_on_cpu = False

        # The _foreach_ path groups the same operations into one kernel per
        # group instead of ~12 per tensor. It only turns on when every
        # accumulator lives on its own weight's device and the dtypes match:
        # _foreach_ ops never cross devices, and vars_on_cpu makes them
        # diverge by construction. initialize_variables decides.
        self._fused_path = False

    @property
    def iterations(self):
        return self.iters

    #override
    def get_weights(self):
        # Stable order the optimizer itself defines: [iterations, ms, vs].
        return [self.iters] + list(self.ms.values()) + list(self.vs.values())

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        """
        `trainable_weights` is an iterable of
        (name, torch.nn.Parameter, owner, param_path) tuples, as produced by
        `Saveable.optimizer_weights()` -- concatenate them across models the way
        the TF call sites concatenated `get_weights()`.

        `name` must be the TF-equivalent scope-qualified name including the
        model's own scope (e.g. 'encoder/down1/conv/weight') -- a torch Parameter
        carries no name of its own, so it is supplied. It is used only to build
        the ms_*/vs_* disk keys the TF build already wrote; it plays no role in
        the arithmetic.

        `owner`/`param_path` identify the Saveable that owns the mirrored
        parameter, so the accumulators go to disk in the same layout the
        parameter does. Pass owner=None (and param_path=None) for a parameter
        that has no owning layer and therefore no layout, e.g. a bare
        torch.nn.Parameter.
        """
        self._lr_dropout_on_cpu = lr_dropout_on_cpu

        for name, v, owner, param_path in trainable_weights:
            key    = f'{name}:0'.replace(':', '_')
            device = torch.device('cpu') if vars_on_cpu else v.device

            self.ms[key] = torch.nn.Parameter(
                torch.zeros(v.shape, dtype=v.dtype, device=device), requires_grad=False)
            self.vs[key] = torch.nn.Parameter(
                torch.zeros(v.shape, dtype=v.dtype, device=device), requires_grad=False)

            self._keys[id(v)]  = (key, v)
            self._mirrors[key] = (owner, param_path)

        self._fused_path = not vars_on_cpu

    def step(self, grads_vars):
        """
        Apply the update immediately. In graph mode `get_update_op` built an
        op executed later via `sess.run`; here the update happens as soon as
        `step` returns.

        `iterations` is incremented *before* it is read for `lr_cos` in this
        same step (so the very first call to step() already uses
        iterations==1 for the cosine schedule). The TF graph had no control
        dependency forcing this order either way -- `lr_cos`'s read of
        `iterations` and the `assign_add` were independent nodes in the same
        `sess.run` -- so the original values themselves reflect whichever
        order the TF executor happened to pick; the difference between them
        is a single integer offset inside a cosine, on the order of 2e-9
        absolute here, far under the 1e-6 tolerance used to compare. Increment-then-read
        is chosen because it is what a straightforward eager port reads as at
        a glance, and it matches this module's own step-1 pseudocode.
        """
        with torch.no_grad():
            if self.clipnorm > 0.0:
                norm = torch.sqrt(sum(torch.sum(torch.square(g.to(torch.float32)))
                                       for g, v in grads_vars))

            self.iters += 1

            if self._fused_path and self.lr_dropout == 1.0 and self.clipnorm == 0.0:
                grads, params, ms_list, vs_list = [], [], [], []
                for g, v in grads_vars:
                    key, registered_v = self._keys[id(v)]
                    assert registered_v is v
                    grads.append(g.to(dtype=v.dtype))
                    params.append(v)
                    ms_list.append(self.ms[key])
                    vs_list.append(self.vs[key])

                # m_t = b1*ms + (1-b1)*g
                torch._foreach_mul_(ms_list, self.beta_1)
                torch._foreach_add_(ms_list, grads, alpha=1.0 - self.beta_1)

                # v_t = b2*vs + (1-b2)*(g - m_t)^2
                diff = torch._foreach_sub(grads, ms_list)
                torch._foreach_mul_(diff, diff)
                torch._foreach_mul_(vs_list, self.beta_2)
                torch._foreach_add_(vs_list, diff, alpha=1.0 - self.beta_2)

                lr = self.lr
                if self.lr_cos != 0:
                    lr = lr * (math.cos(float(self.iters.item()) *
                                        (2 * math.pi / float(self.lr_cos))) + 1.0) / 2.0

                denom = torch._foreach_sqrt(vs_list)
                torch._foreach_add_(denom,
                                    torch.finfo(params[0].dtype).resolution)
                update = torch._foreach_div(ms_list, denom)
                torch._foreach_add_(params, update, alpha=-lr)
                return

            for g, v in grads_vars:
                key, registered_v = self._keys[id(v)]
                assert registered_v is v, (
                    "id(v) collision in AdaBelief._keys: this parameter was "
                    "not the one registered under this id() by "
                    "initialize_variables(). Was step() called with a "
                    "parameter that was never passed to "
                    "initialize_variables()?")

                if self.clipnorm > 0.0:
                    g = self.clip_norm(g, self.clipnorm, norm.to(g.dtype))

                ms = self.ms[key]
                vs = self.vs[key]

                # Accumulators may live on a different device than v/g when
                # vars_on_cpu was set at initialize_variables() time; compute
                # on v's device and write the result back onto the
                # accumulator's own storage.
                g_c  = g.to(dtype=ms.dtype, device=v.device)
                ms_c = ms.to(device=v.device)
                vs_c = vs.to(device=v.device)

                m_t = self.beta_1 * ms_c + (1.0 - self.beta_1) * g_c
                v_t = self.beta_2 * vs_c + (1.0 - self.beta_2) * torch.square(g_c - m_t)

                lr = torch.as_tensor(self.lr, dtype=g_c.dtype, device=v.device)
                if self.lr_cos != 0:
                    # iters is a saved parameter and lives on the CPU; read
                    # it on the weight's device, or the cosine product mixes
                    # devices and raises on the first masked GPU step.
                    lr = lr * (torch.cos(self.iters.to(dtype=g_c.dtype, device=v.device) *
                                          (2 * math.pi / float(self.lr_cos))) + 1.0) / 2.0

                v_diff = -lr * m_t / (torch.sqrt(v_t) + torch.finfo(g_c.dtype).resolution)
                if self.lr_dropout != 1.0:
                    # Resampled every step -- in TF this was a random_uniform
                    # op re-evaluated on every sess.run, not a value fixed
                    # once at graph-build time. Fixing it once would change
                    # training dynamics. Drawn on the weight's device unless
                    # lr_dropout_on_cpu asked for the host (see __init__).
                    mask_device = torch.device('cpu') if self._lr_dropout_on_cpu else v.device
                    mask = nn.random_binomial(v.shape, p=self.lr_dropout, dtype=v.dtype,
                                              device=mask_device)
                    v_diff = v_diff * mask.to(device=v.device, dtype=v_diff.dtype)

                ms.copy_(m_t.to(dtype=ms.dtype, device=ms.device))
                vs.copy_(v_t.to(dtype=vs.dtype, device=vs.device))
                v.add_(v_diff.to(dtype=v.dtype))


nn.AdaBelief = AdaBelief
