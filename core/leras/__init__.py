from .nn import nn

# During the port this was kept at just `from .nn import nn`, because the
# sub-packages were still TensorFlow and importing them eagerly would have
# masked real bugs in the layers already converted. All of core/leras is on
# torch now, so the
# original wiring is restored and `nn.Conv2D` & co. are available from a plain
# `import core.leras`.
#
# The order is not arbitrary. Each of these modules attaches its symbols to the
# `nn` facade at import time, so a module that subclasses one of those symbols
# *at class-definition time* cannot be imported before the one that registers
# it. Verified by reordering, the two hard constraints are:
#   - layers before optimizers: OptimizerBase subclasses nn.Saveable at class
#     scope, so `optimizers` first raises AttributeError: nn has no 'Saveable'.
#   - layers before models:     ModelBase does the same, and fails identically.
#
# `archis` is deliberately kept last for readability, but it imposes no
# import-time constraint of its own: DeepFakeArchi subclasses nn.ArchiBase,
# which its own package registers first, and its nn.ModelBase subclasses
# (Encoder/Inter/Decoder/...) are declared inside DeepFakeArchi.__init__, so
# nn.ModelBase is resolved when an archi is instantiated, not when it is
# imported. Do not "tighten" this into a claim that archis needs models.
from . import ops
from . import initializers
from . import layers
from . import optimizers
from . import models
from . import archis
