"""
Translation between the on-disk weight format and in-memory tensors.

The on-disk format is the one produced by the TensorFlow build and it does not
change: a pickle of dict {tf_variable_name: np.ndarray}, where names follow the
variable_scope hierarchy and keep the ':0' suffix, and tensors are in TF layout
(HWIO for convolutions).

Keeping it unchanged means the same workspace/model directory opens with both
the TF build and this one, and the pretrained weights shipped in the package
need no conversion.
"""
import pickle
from pathlib import Path

import numpy as np
import torch

from core import pathex

_CONTAINER_TYPES = (torch.nn.ModuleList, torch.nn.ModuleDict,
                    torch.nn.ParameterList, torch.nn.ParameterDict)


class MissingWeightsError(Exception):
    """
    Raised when a weights file does not cover every parameter of the module.

    The TF version swallowed these cases with a bare except and silently
    restarted from random weights: a naming bug turned into "training
    restarted from scratch and nobody knows why". This error names the
    offending keys instead.
    """
    def __init__(self, missing=None, mismatched=None):
        self.missing    = missing or []
        self.mismatched = mismatched or []

        parts = []
        if self.missing:
            parts.append("keys missing from file: " + ", ".join(self.missing))
        if self.mismatched:
            parts.append("shape mismatch: " + ", ".join(
                f"{k} (file {file_shape}, expected {expected_shape})"
                for k, file_shape, expected_shape in self.mismatched))
        if not parts:
            # Raised with neither list populated: a caller bug, not a real
            # weight-loading failure. Still give a non-blank message so this
            # never shows up as an empty exception in a traceback.
            parts.append("no missing keys or shape mismatches were given "
                          "(this exception was raised without details)")
        super().__init__("; ".join(parts))


def resolve_path_component(current, part, param_path):
    """
    Resolve one dotted component of a named_parameters() path against the
    live module tree, the same way for every caller that needs to walk it.

    `current` is the node reached so far; `part` is the next path component
    (e.g. "0", "ab", "conv1"); `param_path` is only used to build a useful
    error message. ModuleList/ParameterList index by int, ModuleDict/
    ParameterDict index by key, anything else is a plain attribute.

    Shared by `torch_name_to_disk_key` (which also needs to know, at each
    step, whether `current` *was* a container — that decides the '/' vs '_'
    join) and `Saveable._owner_of` (which only needs the final node). Keeping
    one walk means the two can never resolve the same path differently.
    """
    if isinstance(current, (torch.nn.ModuleList, torch.nn.ParameterList)):
        return current[int(part)]
    if isinstance(current, (torch.nn.ModuleDict, torch.nn.ParameterDict)):
        return current[part]
    if not hasattr(current, part):
        # type name + direct child names, never repr(current): a torch module's
        # repr is its whole subtree, measured at 34 KB for one real encoder, and
        # this runs once per parameter.
        children = ", ".join(sorted(dict(current.named_children()))) \
                   if isinstance(current, torch.nn.Module) else ""
        raise AttributeError(
            f"cannot resolve component {part!r} of path {param_path!r} — "
            f"{type(current).__name__} has no attribute {part!r} "
            f"(direct children: {children or 'none'}). This usually means "
            f"`root` does not match the module `param_path` was taken from "
            f"(e.g. root is missing a prefix that named_parameters() "
            f"included).")
    return getattr(current, part)


def torch_name_to_disk_key(root, param_path):
    """
    Convert a named_parameters() path into the key used on disk.

    leras names list elements f"{name}_{i}" and dict elements f"{name}_{subname}",
    while torch uses "name.i" and "name.subname". The rule: a path component
    whose *container* is a ModuleList/ModuleDict/ParameterList/ParameterDict
    joins to the previous component with '_'; every other component joins with
    '/'. The whole thing gets ':0' appended.

        down1.downs.0.conv1.weight  ->  down1/downs_0/conv1/weight:0

    The tree is walked component by component (rather than pattern-matching the
    string) because whether a given "0" or "ab" is a list index, a dict key, or
    a plain attribute name can only be known by looking at what actually holds
    it in the live module tree.

    Note: the key comes from the *attribute* name, so a nested layer's own
    `name=` is now ignored for keying. TF used the variable_scope, which was
    `layer.name` whenever one was passed explicitly (`_build_sub` only fell back
    to the attribute name `if layer.name is None`), so
    `self.foo = nn.Conv2D(..., name='bar')` keyed `bar/weight:0` there and keys
    `foo/weight:0` here. `name=` still matters on the save/load *root*
    (Saveable.save_weights requires it). No shipped layer passes name= to a
    nested child, so this is latent -- but it is a silent frozen-format
    divergence one keyword argument away, so do not start passing name= to
    children expecting it to reach disk.

    Note: if `param_path`'s last component is itself a direct element of a
    ParameterList/ParameterDict (e.g. "pl.0" with no further attribute after
    the index), the result has no '/' at all (e.g. "pl_0:0"). No key shipped
    in the current build has that shape — leras always nests a named tensor
    (weight/bias/...) one level below any list or dict — but a future layer
    that mirrors a bare parameter list would produce a key nothing on disk
    matches. If that ever happens, this function will need a rule for it.
    """
    parts   = param_path.split('.')
    out     = []
    current = root

    for part in parts:
        if isinstance(current, _CONTAINER_TYPES) and out:
            out[-1] = f"{out[-1]}_{part}"
        else:
            out.append(part)

        current = resolve_path_component(current, part, param_path)

    return "/".join(out) + ":0"


def apply_layout(arr, perm):
    """
    Disk layout -> in-memory layout.

    Returns a transposed *view*, not a copy (and the very same object when
    `perm is None`). Callers must not mutate the result in place — doing so
    would corrupt the original array, including the dict handed back by
    `read_weights_file`. Copy explicitly (`.copy()`) if in-place mutation is
    needed.
    """
    if perm is None:
        return arr
    return np.transpose(arr, perm)


def invert_layout(arr, perm):
    """
    In-memory layout -> disk layout. The exact inverse of apply_layout: for
    any permutation `perm`, invert_layout(apply_layout(arr, perm), perm)
    reconstructs `arr`. Same view/no-copy caveat as apply_layout.
    """
    if perm is None:
        return arr
    return np.transpose(arr, np.argsort(perm))


def read_weights_file(path):
    return pickle.loads(Path(path).read_bytes())


def write_weights_file(path, d):
    pathex.write_bytes_safe(Path(path), pickle.dumps(d, 4))
