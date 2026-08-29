from pathlib import Path

import numpy as np
import torch

from core.leras import nn
from core.leras.weight_io import (MissingWeightsError, apply_layout, invert_layout,
                                  read_weights_file, resolve_path_component,
                                  scrittore_corrente, torch_name_to_disk_key,
                                  write_weights_file)


def _squeeze_shape(shape):
    """
    Drop size-1 dimensions, keeping the order of the rest.

    Used by load_weights to decide whether a shape mismatch is a harmless
    legacy broadcast wrapper (S3FD ships bias as (1,1,1,64) against a (64,)
    parameter) or a genuine layout bug. Matching element count alone is not
    enough: a (4,2,3,3) parameter and a (3,3,2,4) TensorFlow-layout array
    have the same count (72) but are not the same tensor with some 1s
    inserted — accepting that reshape would silently load transposed
    garbage. Squeeze-equality rejects that case while still accepting
    (1,1,1,64) vs (64,), because squeezing only ever removes 1s, never
    reorders the remaining dimensions.
    """
    return tuple(s for s in shape if s != 1)


class Saveable(torch.nn.Module):
    """
    Base class for everything that persists to the on-disk weight format.

    What gets saved: only `torch.nn.Parameter`s (via `get_weights()` /
    `named_parameters()`), including non-trainable ones
    (`requires_grad=False`) — the TF build persisted every saved variable,
    trainable or not (e.g. BatchNorm2D's running_mean/running_var).

    What does NOT get saved: `register_buffer()`. Buffers are for values that
    are recomputable/constant and were never part of the TF weight file to
    begin with — e.g. BlurPool's blur kernel, a tf.constant in the TF build,
    which is why BlurPool must save an *empty* dict. Do not "fix" this by
    iterating named_buffers() alongside named_parameters() in save_weights:
    that would add keys the TensorFlow build does not recognise and break
    the frozen on-disk format this class exists to preserve.

    THE PERSISTENCE ENTRY POINTS, AND THE TWO RULES ALL OF THEM MUST FOLLOW.
    Five methods here enumerate this Saveable's parameters:

        get_weights, get_weights_np, set_weights, save_weights, load_weights

    plus `optimizer_weights`, which enumerates them for a *mirroring* caller.
    Every defect found at this seam so far has been "the rule was applied to
    some of them but not all" — `init_weights` first, and later
    `save_weights`/`load_weights`. So, when adding or
    editing any of them:

      1. Build first. Call `self._ensure_built()` before enumerating, never
         `named_parameters()` straight away. An unbuilt module has no
         parameters, so skipping this makes `save_weights` write `{}` over a
         user's trained file and `load_weights` return True having loaded
         nothing.
      2. Be strict about shapes. Only a squeeze-compatible reshape is a legal
         mismatch (see `_squeeze_shape`); equal element count is not enough.
    """
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    #override
    def _ensure_built(self):
        """
        Make this Saveable's parameters exist, before anything enumerates them.

        The default is a no-op: a plain Saveable creates its parameters in
        __init__. LayerBase, ModelBase and OptimizerBase each override it —
        see rule 1 in the class docstring for why every entry point calls it.
        """
        pass

    #override
    def weight_layouts(self):
        """
        Maps parameter name -> disk->memory permutation (None if identical).
        Declared by the layer that owns the weight, not by a central table.

        The default `{}` *is* a declaration, not an omission: it means "every
        weight of this layer has the same layout on disk as in RAM", which is
        true of all nine 1-D layers and of Dense. Silence here does not mean
        nobody checked. The one layer whose conversion is not a permutation at
        all (DepthwiseConv2D) leaves this empty too and overrides
        weight_from_disk/weight_to_disk instead.
        """
        return {}

    #override
    def weight_from_disk(self, param_path, arr):
        leaf = param_path.rsplit('.', 1)[-1]
        return apply_layout(arr, self.weight_layouts().get(leaf))

    #override
    def weight_to_disk(self, param_path, tensor):
        leaf = param_path.rsplit('.', 1)[-1]
        return invert_layout(tensor.detach().cpu().numpy(), self.weight_layouts().get(leaf))

    def _owner_of(self, param_path):
        """
        The nearest enclosing Saveable that owns the parameter, i.e. the one
        whose weight_layouts() applies to it.

        A parameter can sit inside a plain container (ParameterList,
        ParameterDict) that is not itself a Saveable and has no
        weight_layouts() of its own — e.g. `self.ws = nn.ParameterList([...])`
        on a Saveable, where the parameter's path is "ws.0". Stopping at that
        container (the naive "last resolved node") would silently skip the
        owner's declared layout, since a plain container has no
        weight_to_disk/weight_from_disk to fall back to. So this keeps
        walking and remembers the last Saveable seen, not just the last node.

        Walks the tree with the same rules `torch_name_to_disk_key` uses to
        build the disk key, via the shared `resolve_path_component` helper —
        the two must never disagree about what a given path component means.
        """
        owner   = self
        current = self
        for part in param_path.split('.')[:-1]:
            current = resolve_path_component(current, part, param_path)
            if isinstance(current, Saveable):
                owner = current
        return owner

    #override
    def get_weights(self):
        self._ensure_built()
        return [p for _, p in self.named_parameters()]

    def get_weights_np(self):
        return [p.detach().cpu().numpy() for p in self.get_weights()]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError('len of lists mismatch')

        with torch.no_grad():
            for w, new_w in zip(weights, new_weights):
                new_w = np.asarray(new_w)
                if new_w.shape != tuple(w.shape):
                    new_w = new_w.reshape(tuple(w.shape))
                w.copy_(torch.as_tensor(new_w, dtype=w.dtype))

    def optimizer_weights(self):
        """
        The (name, param, owner, param_path) tuples an optimizer needs to mirror
        this Saveable's parameters into accumulators.

        `name` is the TF-equivalent scope-qualified variable name without the
        ':0' suffix, with this Saveable's own `name` as the outermost scope —
        i.e. exactly `v.name[:-2]` in the TF graph, which is what the ms_*/vs_*/
        acc_* accumulator disk keys were mangled from. Building it here rather
        than at the call site is deliberate: forgetting the `<model_name>/`
        prefix silently renames every accumulator key on disk and no existing
        training session resumes.

        `owner` and `param_path` are what `owner.weight_to_disk(param_path, ...)`
        needs, so the accumulator lands on disk in the same layout as the
        parameter it mirrors. The owning layer is the only correct source of
        that layout: it covers the permutations (Conv2D, Conv2DTranspose), the
        non-permutations (DepthwiseConv2D, which reshapes as well as
        transposes) and the identities (Dense, every bias) alike, and an
        optimizer that derived a permutation from tensor rank instead would be
        wrong for at least the second of those.

        Returns a list, not a generator: the TF call sites concatenate these
        across models and then filter them (models/Model_SAEHD/Model.py:352-359).
        """
        if self.name is None:
            raise Exception("name must be defined.")

        self._ensure_built()

        return [(f'{self.name}/{torch_name_to_disk_key(self, path)[:-2]}',
                 param, self._owner_of(path), path)
                for path, param in self.named_parameters()]

    def save_weights(self, filename, force_dtype=None):
        if self.name is None:
            raise Exception("name must be defined.")

        self._ensure_built()

        d = {}
        for path, param in self.named_parameters():
            owner = self._owner_of(path)
            w_val = owner.weight_to_disk(path, param) if hasattr(owner, 'weight_to_disk') \
                    else param.detach().cpu().numpy()
            if force_dtype is not None:
                w_val = w_val.astype(force_dtype)
            w_val = np.ascontiguousarray(w_val)
            # Una copia sola, e solo quando serve. Da una GPU `.cpu()` ha gia'
            # creato un tensore nuovo che nessun altro tocca: l'array che ne
            # viene e' lo snapshot, e copiarlo di nuovo era la seconda copia
            # intera che il salvataggio pagava (+0,8 GB su un H2 a 224). Su
            # CPU invece `numpy()` condivide la memoria del parametro vivo
            # (OWNDATA falso e nessuna trasposizione in mezzo): li' la copia
            # e' obbligatoria, perche' `d` puo' essere scritto in sfondo
            # mentre il training continua ad aggiornare quel parametro.
            if not w_val.flags['OWNDATA'] and param.device.type == 'cpu':
                w_val = w_val.copy()
            d[torch_name_to_disk_key(self, path)] = w_val

        scrittore = scrittore_corrente()
        if scrittore is not None:
            scrittore.accoda(filename, d)
        else:
            write_weights_file(filename, d)

    def load_weights(self, filename):
        """
        Returns True if the file exists and every parameter loaded.

        Unlike the TF version, this does not swallow errors: a missing key or
        an incompatible shape raises MissingWeightsError instead of silently
        restarting the model from random weights. A missing file is still the
        legitimate False case (first run).
        """
        filepath = Path(filename)
        if not filepath.exists():
            return False

        if self.name is None:
            raise Exception("name must be defined.")

        self._ensure_built()

        d          = read_weights_file(filepath)
        missing    = []
        mismatched = []
        tuples     = []

        for path, param in self.named_parameters():
            key   = torch_name_to_disk_key(self, path)
            w_val = d.get(key, None)
            if w_val is None:
                missing.append(key)
                continue

            owner = self._owner_of(path)
            w_val = owner.weight_from_disk(path, w_val) if hasattr(owner, 'weight_from_disk') \
                    else w_val

            if w_val.shape != tuple(param.shape):
                if _squeeze_shape(w_val.shape) != _squeeze_shape(tuple(param.shape)):
                    mismatched.append((key, w_val.shape, tuple(param.shape)))
                    continue
                w_val = np.reshape(w_val, tuple(param.shape))    # legacy file, e.g. (1,1,1,C) -> (C,)

            tuples.append((param, w_val))

        if missing or mismatched:
            raise MissingWeightsError(missing=missing, mismatched=mismatched)

        with torch.no_grad():
            for param, w_val in tuples:
                param.copy_(torch.as_tensor(np.ascontiguousarray(w_val), dtype=param.dtype))

        return True

    def init_weights(self):
        self.build_weights()


nn.Saveable = Saveable
