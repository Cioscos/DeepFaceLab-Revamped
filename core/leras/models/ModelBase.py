import types

import numpy as np

from core.interact import interact as io
from core.leras import nn


class ModelBase(nn.Saveable):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.built  = False
        self.args   = args
        self.kwargs = kwargs

    def build(self):
        # torch.nn.Module registers submodules assigned as attributes, so
        # vars(self) no longer needs to be scanned: just running on_build is
        # enough. The generator support stays because it determines the
        # order in which submodules are created, and therefore the names of
        # their weights on disk.
        generator = self.on_build(*self.args, **self.kwargs)
        if isinstance(generator, types.GeneratorType):
            for _ in generator:
                pass

        self._reject_plain_containers()

        # Snapshot the subtree before mutating it. self.modules() is a lazy
        # generator that descends into a child *after* yielding it, so
        # calling child.build() mid-iteration -- which registers the
        # child's own submodules -- would mutate a structure the generator
        # is still walking. An unbuilt child ModelBase has no children yet
        # (its on_build has not run), so it is not enough to just look for
        # LayerBase instances here: TF's _build_sub called layer.build() on
        # model children, and that recursive call is what has to happen for
        # a child model's own subtree (including its own nested models) to
        # be built at all.
        #
        # The snapshot is also sufficient, not just safe: a grandchild under
        # an unbuilt child model does not need to appear in *this* snapshot,
        # because the child's own build() (called below) walks its own
        # subtree with its own snapshot. And a ModuleList holding model
        # children is covered for free, because self.modules() already
        # yields the list's elements (they exist at snapshot time, unlike an
        # unbuilt model's not-yet-created children).
        for m in [m for m in self.modules() if m is not self]:
            if isinstance(m, ModelBase):
                if not m.built:
                    m.build()
            elif isinstance(m, nn.LayerBase) and not m._built:
                # The _built guard matters here: without it, a subtree built
                # lazily via __call__ (or built by an ancestor's recursive
                # build() before this one runs) would get build_weights()
                # called a second time, silently resetting its weights.
                m.build_weights()
                m._built = True

        self.built = True

    def _reject_plain_containers(self):
        """
        torch.nn.Module only registers submodules assigned directly to an
        attribute. A plain list or dict of layers is invisible to .modules(),
        so its weights would never be built, never saved and never trained --
        with no error anywhere. The TF _build_sub walked lists and dicts
        explicitly, so the pattern was legitimate there and is still in the
        code waiting to be ported (DeepFakeArchi's `self.downs = []`). Fail
        loudly instead of dropping them.

        Attributes whose name starts with '_' are skipped: those are torch's
        own bookkeeping dicts (_modules, _parameters, _buffers), which
        legitimately hold Saveables. 'args' and 'kwargs' are ModelBase's own
        bookkeeping (the constructor arguments replayed into on_build) and
        are skipped by name for the same reason -- a Saveable passed through
        the constructor would otherwise false-positive here, since it lands
        in self.args/self.kwargs, a plain tuple/dict, as an artifact of
        __init__ rather than a container of layers the model owns. No
        current call site does this, so it is untested; a layer passed
        through the constructor is therefore not checked by this guard.
        """
        _own_bookkeeping = {'args', 'kwargs', 'name', 'built'}
        for attr, value in vars(self).items():
            if attr.startswith('_') or attr in _own_bookkeeping:
                continue
            if isinstance(value, (list, tuple)):
                items, kind = value, 'ModuleList'
            elif isinstance(value, dict):
                items, kind = value.values(), 'ModuleDict'
            else:
                continue
            if any(isinstance(v, nn.Saveable) for v in items):
                raise TypeError(
                    f"{type(self).__name__}.{attr} is a plain "
                    f"{type(value).__name__} holding layers, which torch would "
                    f"not register as submodules. Use torch.nn.{kind}.")

    def __call__(self, *args, **kwargs):
        # The TF ModelBase built on first call; torch.nn.Module.__call__ does
        # not. Delegating to super().__call__ keeps torch's hook machinery.
        if not self.built:
            self.build()
        return super().__call__(*args, **kwargs)

    #override
    def on_build(self, *args, **kwargs):
        """
        init model layers here

        return 'yield' if build is not finished
                    therefore dependency models will be initialized
        """
        pass

    #override
    def forward(self, *args, **kwargs):
        #flow layers/models/tensors here
        pass

    #override
    def _ensure_built(self):
        # The auto-build is load-bearing, not a convenience: a model builds its
        # subtree lazily, so every persistence entry point on Saveable
        # (get_weights, set_weights, save_weights, load_weights,
        # optimizer_weights) would otherwise see an empty parameter list —
        # get_weights() returning [] silently disarms the optimizer,
        # save_weights() writes {} over a trained file, load_weights() reports
        # success having loaded nothing. The TF ModelBase.get_weights() did
        # exactly this; hosting it here applies it to all of them at once
        # instead of one method at a time.
        if not self.built:
            self.build()

    #override
    def init_weights(self):
        # Saveable.init_weights() delegates to self.build_weights(), which
        # ModelBase does not define -- it has no weights of its own, only a
        # subtree of layers/models that own theirs. The TF original was
        # nn.init_weights(self.get_weights()): it created variables
        # uninitialised and then ran the initialiser ops on demand. In torch,
        # build_weights() both creates and initialises each layer's
        # parameters in one step, so ensuring the whole subtree is built is
        # the equivalent -- this is what every first-run training path calls
        # on a model (e.g. models/Model_SAEHD/Model.py, facelib/XSegNet.py).
        if not self.built:
            self.build()

    def get_layer_by_name(self, name):
        return dict(self.named_children()).get(name, None)

    def get_layers(self):
        if not self.built:
            self.build()
        return [m for m in self.modules() if isinstance(m, nn.LayerBase)]

    def summary(self):
        layers = self.get_layers()
        layers_names = []
        layers_params = []

        max_len_str = 0
        max_len_param_str = 0
        delim_str = "-"

        total_params = 0

        #Get layers names and str lenght for delim
        for l in layers:
            if len(str(l))>max_len_str:
                max_len_str = len(str(l))
            layers_names+=[str(l).capitalize()]

        #Get params for each layer
        layers_params = [ sum(p.numel() for p in l.parameters()) for l in layers ]
        total_params = np.sum(layers_params)

        #Get str lenght for delim
        for p in layers_params:
            if len(str(p))>max_len_param_str:
                max_len_param_str=len(str(p))

        #Set delim
        for i in range(max_len_str+max_len_param_str+3):
            delim_str += "-"

        output = "\n"+delim_str+"\n"

        #Format model name str
        model_name_str = "| "+self.name.capitalize()
        len_model_name_str = len(model_name_str)
        for i in range(len(delim_str)-len_model_name_str):
            model_name_str+= " " if i!=(len(delim_str)-len_model_name_str-2) else " |"

        output += model_name_str +"\n"
        output += delim_str +"\n"


        #Format layers table
        for i in range(len(layers_names)):
            output += delim_str +"\n"

            l_name = layers_names[i]
            l_param = str(layers_params[i])
            l_param_str = ""
            if len(l_name)<=max_len_str:
                for i in range(max_len_str - len(l_name)):
                    l_name+= " "

            if len(l_param)<=max_len_param_str:
                for i in range(max_len_param_str - len(l_param)):
                    l_param_str+= " "

            l_param_str += l_param


            output +="| "+l_name+"|"+l_param_str+"| \n"

        output += delim_str +"\n"

        #Format sum of params
        total_params_str = "| Total params count: "+str(total_params)
        len_total_params_str = len(total_params_str)
        for i in range(len(delim_str)-len_total_params_str):
            total_params_str+= " " if i!=(len(delim_str)-len_total_params_str-2) else " |"

        output += total_params_str +"\n"
        output += delim_str +"\n"

        io.log_info(output)

nn.ModelBase = ModelBase
