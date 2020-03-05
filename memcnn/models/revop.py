# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import warnings
from memcnn.models.additive import AdditiveCoupling
from memcnn.models.affine import AffineCoupling
from memcnn.models.utils import pytorch_version_one_and_above


class InvertibleCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, num_inputs, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.num_inputs = num_inputs

        input_t = inputs_and_weights[:num_inputs]
        ctx.input_requires_grad = [element.requires_grad for element in input_t]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = [element.detach() for element in input_t]
            output = ctx.fn(*x)


        if not isinstance(output, tuple):
            output = (output,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_output = tuple([element.detach_() for element in output])

        # store these tensor nodes for backward pass
        ctx.input_t = [input_t] * num_bwd_passes
        ctx.output_t = [detached_output] * num_bwd_passes

        return detached_output

    @staticmethod
    def backward(ctx, *grad_output):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.output_t) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpointFunction for more than "
                               "{} times! Try raising `num_bwd_passes` by one.".format(ctx.num_bwd_passes))
        input_t = ctx.input_t.pop()
        output = ctx.output_t.pop()

        # recompute input if necessary
        if not ctx.keep_input:
            with torch.no_grad():
                input_inverted = ctx.fn_inverse(*output)
                if not isinstance(input_inverted, tuple):
                    input_inverted = (input_inverted,)
                if pytorch_version_one_and_above:
                    for element_original, element_inverted in zip(input_t, input_inverted):
                        element_original.storage().resize_(int(np.prod(element_original.size())))
                        element_original.set_(element_inverted)
                else:
                    for element_original, element_inverted in zip(input_t, input_inverted):
                        element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_input = tuple([element.detach().requires_grad_() for element in input_t])
            temp_output = ctx.fn(*detached_input)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        gradients = torch.autograd.grad(outputs=temp_output, inputs=detached_input + ctx.weights, grad_outputs=grad_output)

        return (None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1, disable=False):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.

        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.

            disable : :obj:`bool`, optional
                This will disable the detached graph approach with the backward hook.
                Essentially this renders the function as `y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        Raises
        ------
        NotImplementedError
            If an unknown coupling or implementation is given.

        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self._fn = fn

    def forward(self,*xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.

        """
        if not self.disable:
            if not isinstance(xin, tuple):
                xin = (xin,)
            y = InvertibleCheckpointFunction.apply(
                self._fn.forward,
                self._fn.inverse,
                self.keep_input,
                self.num_bwd_passes,
                len(xin),
                *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
            if not self.keep_input:
                if not pytorch_version_one_and_above:
                    # PyTorch 0.4 way to clear storage
                    for element in xin:
                        element.data.set_()
                else:
                    # PyTorch 1.0+ way to clear storage
                    for element in xin:
                        element.storage().resize_(0)
        else:
            y = self._fn(*xin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            y = y[0]

        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.

        """
        if not self.disable:
            if not isinstance(yin, tuple):
                yin = (yin,)
            x = InvertibleCheckpointFunction.apply(
                self._fn.inverse,
                self._fn.forward,
                self.keep_input_inverse,
                self.num_bwd_passes,
                len(yin),
                *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
            if not self.keep_input_inverse:
                if not pytorch_version_one_and_above:
                    # PyTorch 0.4 way to clear storage
                    for element in yin:
                        element.data.set_()
                else:
                    # PyTorch 1.0+ way to clear storage
                    for element in yin:
                        element.storage().resize_(0)
        else:
            x = self._fn.inverse(*yin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        return x


class ReversibleBlock(InvertibleModuleWrapper):
    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, keep_input_inverse=False,
                 implementation_fwd=-1, implementation_bwd=-1, adapter=None):
        """The ReversibleBlock

        Warning
        -------
        This class has been deprecated. Use the more flexible InvertibleModuleWrapper class.

        Note
        ----
        The `implementation_fwd` and `implementation_bwd` parameters can be set to one of the following implementations:

        * -1 Naive implementation without reconstruction on the backward pass.
        * 0  Memory efficient implementation, compute gradients directly.
        * 1  Memory efficient implementation, similar to approach in Gomez et al. 2017.


        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function

            Gm : :obj:`torch.nn.Module`, optional
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)

            coupling : :obj:`str`, optional
                Type of coupling ['additive', 'affine']. Default = 'additive'

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            implementation_fwd : :obj:`int`, optional
                Switch between different Operation implementations for forward training (Default = 1).
                If using the naive implementation (-1) then `keep_input` should be True.

            implementation_bwd : :obj:`int`, optional
                Switch between different Operation implementations for backward training (Default = 1).
                If using the naive implementation (-1) then `keep_input_inverse` should be True.

            adapter : :obj:`class`, optional
                Only relevant when using the 'affine' coupling.
                Should be a class of type :obj:`torch.nn.Module` that serves as an
                optional wrapper class A for Fm and Gm which must output
                s, t = A(x) with shape(s) = shape(t) = shape(x).
                s, t are respectively the scale and shift tensors for the affine coupling.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        Raises
        ------
        NotImplementedError
            If an unknown coupling or implementation is given.

        """
        warnings.warn("This class has been deprecated. Use the more flexible InvertibleModuleWrapper class", DeprecationWarning)
        fn = create_coupling(Fm=Fm, Gm=Gm, coupling=coupling,
                             implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd,
                             adapter=adapter)
        super(ReversibleBlock, self).__init__(fn, keep_input=keep_input, keep_input_inverse=keep_input_inverse)


def create_coupling(Fm, Gm=None, coupling='additive', implementation_fwd=-1, implementation_bwd=-1, adapter=None):
    if coupling == 'additive':
        fn = AdditiveCoupling(Fm, Gm,
                              implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
    elif coupling == 'affine':
        fn = AffineCoupling(Fm, Gm, adapter=adapter,
                            implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
    else:
        raise NotImplementedError('Unknown coupling method: %s' % coupling)
    return fn


def is_invertible_module(module_in, test_input_shape, test_input_dtype=torch.float32, atol=1e-6):
    """Test if a :obj:`torch.nn.Module` is invertible

    Parameters
    ----------
    module_in : :obj:`torch.nn.Module`
        A torch.nn.Module to test.
    test_input_shape : :obj:`tuple`
        Dimensions of test tensor object to perform the test with.
    test_input_dtype : :obj:`torch.dtype`, optional
        Data type of test tensor object to perform the test with.
    atol : :obj:`float`, optional
        Tolerance value used for comparing the outputs

    Returns
    -------
        :obj:`bool`
            True if the input module is invertible, False otherwise.

    """
    if isinstance(module_in, InvertibleModuleWrapper):
        module_in = module_in._fn
    test_input = torch.rand(test_input_shape, dtype=test_input_dtype)
    if not hasattr(module_in, "inverse"):
        return False
    with torch.no_grad():
        if not torch.allclose(module_in.inverse(module_in(test_input)), test_input, atol=atol):
            return False
        if not torch.allclose(module_in(module_in.inverse(test_input)), test_input, atol=atol):
            return False
        if test_input is module_in(test_input):
            return False
        if test_input is module_in.inverse(test_input):
            return False
    return True
