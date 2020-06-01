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
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, num_inputs, preserve_rng_state, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.num_inputs = num_inputs
        ctx.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*inputs_and_weights)

        inputs = inputs_and_weights[:num_inputs]
        ctx.input_requires_grad = [element.requires_grad for element in inputs]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = [element.detach() for element in inputs]
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_outputs = tuple([element.detach_() for element in outputs])

        # clear memory from inputs
        if not ctx.keep_input:
            if not pytorch_version_one_and_above:
                # PyTorch 0.4 way to clear storage
                for element in inputs:
                    element.data.set_()
            else:
                # PyTorch 1.0+ way to clear storage
                for element in inputs:
                    element.storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpointFunction for more than "
                               "{} times! Try raising `num_bwd_passes` by one.".format(ctx.num_bwd_passes))
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input if necessary
        if not ctx.keep_input:
            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                with torch.no_grad():
                    inputs_inverted = ctx.fn_inverse(*outputs)
                    if not isinstance(inputs_inverted, tuple):
                        inputs_inverted = (inputs_inverted,)
                    if pytorch_version_one_and_above:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.storage().resize_(int(np.prod(element_original.size())))
                            element_original.set_(element_inverted)
                    else:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = tuple([element.detach().requires_grad_() for element in inputs])
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        gradients = torch.autograd.grad(outputs=temp_output, inputs=detached_inputs + ctx.weights, grad_outputs=grad_outputs)

        # Setting the gradients manually on the inputs and outputs (mimic backwards)
        for element, element_grad in zip(inputs, gradients[:ctx.num_inputs]):
            element.grad = element_grad

        for element, element_grad in zip(outputs, grad_outputs):
            element.grad = element_grad

        return (None, None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1,
                 disable=False, preserve_rng_state=True):
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

            preserve_rng_state : :obj:`bool`, optional
                Setting this will ensure that the same RNG state is used during reconstruction of the inputs.
                I.e. if keep_input = False on forward or keep_input_inverse = False on inverse. By default
                this is True.

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
        self.preserve_rng_state = preserve_rng_state
        self._fn = fn

    def forward(self, *xin):
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
            y = InvertibleCheckpointFunction.apply(
                self._fn.forward,
                self._fn.inverse,
                self.keep_input,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(xin),
                *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            y = self._fn(*xin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
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
            x = InvertibleCheckpointFunction.apply(
                self._fn.inverse,
                self._fn.forward,
                self.keep_input_inverse,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(yin),
                *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            x = self._fn.inverse(*yin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
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


def is_invertible_module(module_in, test_input_shape, test_input_dtype=torch.float32, atol=1e-6, random_seed=42):
    """Test if a :obj:`torch.nn.Module` is invertible

    Parameters
    ----------
    module_in : :obj:`torch.nn.Module`
        A torch.nn.Module to test.
    test_input_shape : :obj:`tuple` of :obj:`int` or :obj:`tuple` of :obj:`tuple` of :obj:`int`
        Dimensions of test tensor(s) object to perform the test with.
    test_input_dtype : :obj:`torch.dtype`, optional
        Data type of test tensor object to perform the test with.
    atol : :obj:`float`, optional
        Tolerance value used for comparing the outputs.
    random_seed : :obj:`int`, optional
        Use this value to seed the pseudo-random test_input_shapes with different numbers.

    Returns
    -------
        :obj:`bool`
            True if the input module is invertible, False otherwise.

    """
    if isinstance(module_in, InvertibleModuleWrapper):
        module_in = module_in._fn

    if not hasattr(module_in, "inverse"):
        return False

    def _type_check_input_shape(test_input_shape):
        if isinstance(test_input_shape, (tuple, list)):
            if all([isinstance(e, int) for e in test_input_shape]):
                return True
            elif all([isinstance(e, (tuple, list)) for e in test_input_shape]):
                return all([isinstance(ee, int) for e in test_input_shape for ee in e])
            else:
                return False
        else:
            return False

    if not _type_check_input_shape(test_input_shape):
        raise ValueError("test_input_shape should be of type Tuple[int, ...] or "
                         "Tuple[Tuple[int, ...], ...], but {} found".format(type(test_input_shape)))

    if not isinstance(test_input_shape[0], (tuple, list)):
        test_input_shape = (test_input_shape,)

    def _check_inputs_allclose(inputs, reference, atol):
        for inp, ref in zip(inputs, reference):
            if not torch.allclose(inp, ref, atol=atol):
                return False
        return True

    def _pack_if_no_tuple(x):
        if not isinstance(x, tuple):
            return (x, )
        return x

    with torch.no_grad():
        torch.manual_seed(random_seed)
        test_inputs = tuple([torch.rand(shape, dtype=test_input_dtype) for shape in test_input_shape])
        if any([torch.equal(torch.zeros_like(e), e) for e in test_inputs]):  # pragma: no cover
            warnings.warn("Some inputs were detected to be all zeros, you might want to set a different random_seed.")

        if not _check_inputs_allclose(_pack_if_no_tuple(module_in.inverse(*_pack_if_no_tuple(module_in(*test_inputs)))), test_inputs, atol=atol):
            return False

        test_outputs = _pack_if_no_tuple(module_in(*test_inputs))
        if any([torch.equal(torch.zeros_like(e), e) for e in test_outputs]):  # pragma: no cover
            warnings.warn("Some outputs were detected to be all zeros, you might want to set a different random_seed.")

        if not _check_inputs_allclose(_pack_if_no_tuple(module_in(*_pack_if_no_tuple(module_in.inverse(*test_outputs)))), test_outputs, atol=atol):  # pragma: no cover
            return False

        test_reconstructed_inputs = _pack_if_no_tuple(module_in.inverse(*test_outputs))

    def _test_shared(inputs, outputs, msg):
        shared = set(inputs)
        shared_outputs = set(outputs)
        if len(inputs) != len(shared):  # pragma: no cover
            warnings.warn("Some inputs (*x) share the same tensor, are you sure this is what you want? ({})".format(msg))
        if len(outputs) != len(shared_outputs):
            warnings.warn("Some outputs (*y) share the same tensor, are you sure this is what you want? ({})".format(msg))
        if any([inp in shared for inp in shared_outputs]):
            warnings.warn("Some inputs (*x) and outputs (*y) share the same tensor, this is typically not a "
                          "good function to use with memcnn.InvertibleModuleWrapper as it might increase memory usage. "
                          "E.g. an identity function. ({})".format(msg))

    _test_shared(test_inputs, test_outputs, msg="forward")
    _test_shared(test_reconstructed_inputs, test_outputs, msg="inverse")

    return True
