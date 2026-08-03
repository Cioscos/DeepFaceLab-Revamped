import torch

from core.leras import nn


class OptimizerBase(nn.Saveable):
    def __init__(self, name=None):
        super().__init__(name=name)

        # accumulator ParameterDict key -> (owning Saveable, torch path) of the
        # parameter it mirrors, filled in by initialize_variables(). It exists
        # only so weight_to_disk/weight_from_disk below can delegate the layout
        # conversion to the layer that owns the mirrored weight; it holds no
        # state that needs persisting and is rebuilt on every construction,
        # since initialize_variables() always runs before any save/load.
        self._mirrors = {}

    #override
    def weight_to_disk(self, param_path, tensor):
        """
        An accumulator goes to disk in the same layout as the weight it mirrors.

        The TF build created each accumulator with `tf.get_variable(f'ms_{v.name}',
        v.shape)` where `v` was the *TF* variable, so a convolution's accumulator
        was HWIO on disk exactly like the convolution's own kernel. Here the
        accumulator is allocated with the torch parameter's shape, so without
        this delegation it would be written OIHW: every convolution accumulator
        in an existing `*_opt.npy` mismatches on resume, and files this build
        writes cannot be read by the TF build. The frozen on-disk format covers
        the optimizer files too.

        Delegating to the owner rather than permuting here is what makes it
        correct for every layout at once — see Saveable.optimizer_weights.
        """
        owner, mirrored_path = self._mirrors.get(param_path.rsplit('.', 1)[-1], (None, None))
        if owner is None:
            # `iters` (shape (1,), key 'iters:0'), and any parameter registered
            # without an owner. No layout either way.
            return tensor.detach().cpu().numpy()
        return owner.weight_to_disk(mirrored_path, tensor)

    #override
    def weight_from_disk(self, param_path, arr):
        owner, mirrored_path = self._mirrors.get(param_path.rsplit('.', 1)[-1], (None, None))
        if owner is None:
            return arr
        return owner.weight_from_disk(mirrored_path, arr)

    #override
    def _ensure_built(self):
        """
        Nothing to build: an optimizer's accumulators are created (and zeroed)
        by `initialize_variables()`, which every call site runs long before any
        save/load. The no-op is deliberate, not an omission — see the
        persistence-entry-point rules in `Saveable`'s docstring.
        """
        pass

    #override
    def init_weights(self):
        """
        `Saveable.init_weights` (used by every model's first-run branch, e.g.
        `if do_init: model.init_weights()` in ModelBase) delegates to
        `self.build_weights()` by default -- that is the right thing for a
        layer, which builds its weights lazily on first use, but an
        optimizer has none of that: its accumulators are already
        zero-initialised in `initialize_variables()`, called well before
        `init_weights()` could ever run. So there is genuinely nothing to
        do here; without this override, `init_weights()` on any optimizer
        raises AttributeError (no `build_weights`), breaking first-run
        training the same way a broken `load_weights()` breaks resume.
        """
        pass

    def clip_norm(self, g, c, n):
        """Clip the gradient `g` if the L2 norm `n` exceeds `c`.

        # Arguments
            g: Tensor, the gradient tensor
            c: float >= 0. Gradients will be clipped
                when their L2 norm exceeds this value.
            n: Tensor, actual norm of `g` (same dtype as `g`).
        # Returns
            Tensor, the gradient clipped if required.

        The TF original was `tf.cond(n >= c, lambda: g * (c / n), lambda: g)`.
        `g * clamp(c / n, max=1.0)` is the same piecewise function without a
        graph-mode conditional: when n >= c it scales by c/n <= 1 (the clipped
        branch), otherwise c/n > 1 and the clamp caps it at 1.0, leaving g
        unchanged (the pass-through branch). It also sidesteps the
        div-by-zero-in-the-untaken-branch trap `tf.cond` avoided lazily: if
        n == 0 then c/n is +inf, clamp(+inf, max=1.0) == 1.0, and g is all
        zero anyway (a zero-norm gradient has no nonzero components), so the
        result is exactly g -- no NaN is produced.
        """
        if c <= 0:  # if clipnorm == 0 no need to clip
            return g

        scale = torch.clamp(c / n, max=1.0)
        return g * scale


nn.OptimizerBase = OptimizerBase
