from core.leras import nn


class LayerBase(nn.Saveable):
    # Class attribute, not set in __init__: LayerBase defines no __init__ (the
    # layers call super().__init__(**kwargs) straight through to Saveable), and
    # ModelBase.build() is the only thing that flips it, per instance.
    _built = False

    #override
    def build_weights(self):
        pass

    #override
    def _ensure_built(self):
        """
        Fail loudly, rather than auto-build, when a standalone layer's weights
        were never created.

        Auto-building here would be wrong: `build_weights()` also *initialises*,
        so calling it on a layer that was built manually — the pattern every
        layer test and every `ModelBase.build()` subtree walk uses — would
        replace the layer's weights with fresh random ones, silently, right
        before they are saved. The TF build raised too, just less explicitly
        (`Conv2D.get_weights()` referenced `self.weight` before it existed).

        "Never built" is detected as: the layer declares weights (it overrides
        the `build_weights` stub) but owns no parameters and no buffers yet.
        A layer that genuinely has none (DenseNorm) keeps the stub and passes;
        a built BlurPool owns its kernel buffer and passes.

        The ownership check is deliberately non-recursive (`_parameters` /
        `_buffers`, not `parameters()` / `buffers()`). The recursive form would
        let an unbuilt layer that happens to contain a *built* sub-layer pass
        this guard, and `save_weights` would then write a partial dict missing
        the outer layer's own keys — the exact silent failure this guard exists
        to prevent, reintroduced inside the guard itself. No current layer nests
        a sub-layer, so the recursive form was not observably wrong; it was
        waiting for the first one that does.
        """
        if type(self).build_weights is LayerBase.build_weights:
            return
        if not self._parameters and not self._buffers:
            raise RuntimeError(
                f"{type(self).__name__} owns no weights: build_weights() has "
                f"never been called on it. A standalone layer must be built "
                f"before its weights can be enumerated, saved or loaded "
                f"(ModelBase.build() does this for every layer in a model's "
                f"subtree).")

    #override
    def forward(self, *args, **kwargs):
        pass


nn.LayerBase = LayerBase
