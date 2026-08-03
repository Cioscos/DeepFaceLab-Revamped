from core.leras import nn

from .CA import CAInitializerSubprocessor

class initializers():
    class ca:
        """
        Convolution Aware Initialization https://arxiv.org/abs/1702.06295

        The TF build resolved this lazily: __call__ returned a placeholder
        tf.zeros(..., name="_cai_") and a separate nn.init_weights() pass
        scanned every Variable for that "_cai_" marker, batch-generated real
        values via CAInitializerSubprocessor, and assigned them in. Layers
        now initialize synchronously inside build_weights(), so that
        indirection is gone: generate()/generate_batch() just return numpy
        arrays directly.

        CONSEQUENCE, since it is easy to miss: this class is therefore NOT a
        usable `kernel_initializer` any more. A layer's kernel_initializer is
        now `f(torch_tensor) -> None`, filling in place in torch layout
        (out,in,kh,kw for Conv2D); `ca` has no `__call__` at all and
        `generate` returns (kh,kw,in,out) numpy arrays. Nothing passes it
        today. Reviving it means giving it an in-place __call__ that permutes
        generate()'s output into torch layout -- not simply wiring the class
        into a kernel_initializer argument.
        """
        generate = staticmethod(CAInitializerSubprocessor.generate)

        @staticmethod
        def generate_batch( data_list, eps_std=0.05 ):
            # list of (shape, np.dtype)
            return CAInitializerSubprocessor (data_list).run()

nn.initializers = initializers
