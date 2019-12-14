# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import warnings
from memcnn.models.additive import AdditiveCoupling
from memcnn.models.affine import AffineCoupling
from memcnn.models.utils import pytorch_version_one_and_above


warnings.filterwarnings(action='ignore', category=UserWarning)


def signal_hook(grad_output, valid_states, state_index):  # pragma: no cover
    state = valid_states[state_index]
    valid_states[state_index] = not state


def backward_hook(grad_output, keep_input, compute_input_fn, compute_output_fn,
                  input_tensor, output_tensor, valid_states, state_index):  # pragma: no cover
    perform_action = valid_states[state_index]
    valid_states[state_index] = not perform_action
    if perform_action:
        # restore input
        if not keep_input:
            with torch.no_grad():
                input_inverted = compute_input_fn(output_tensor)
                if pytorch_version_one_and_above:
                    input_tensor.storage().resize_(int(np.prod(input_tensor.size())))
                    input_tensor.set_(input_inverted)
                else:
                    input_tensor.set_(input_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            temp_output = compute_output_fn(input_tensor)
            with torch.no_grad():
                temp_output.set_(output_tensor)
            temp_output.backward(gradient=grad_output)


class ReversibleModule(nn.Module):
    def __init__(self, fn, keep_input=False, keep_input_inverse=False, disable=False):
        """The ReversibleModule

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
        super(ReversibleModule, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self._fn = fn
        self._valid_states = []
        self._state_counter = 0


    def forward(self, xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            xin : :obj:`torch.Tensor`
                Input torch tensor.

        Returns
        -------
            :obj:`torch.Tensor`
                Output torch tensor y.

        """
        if not self.disable:
            x = xin.detach()  # Makes a detached copy which shares the storage
            y = self._fn(x)
            input_tensor = xin
            output_tensor = y
            # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
            if not self.keep_input:
                if not pytorch_version_one_and_above:
                    # PyTorch 0.4 way to clear storage
                    input_tensor.data.set_()
                else:
                    # PyTorch 1.0+ way to clear storage
                    input_tensor.storage().resize_(0)
            if self.training:
                self._valid_states.append(True)
                xin.register_hook(hook=partial(signal_hook, valid_states=self._valid_states, state_index=self._state_counter))
                y.register_hook(hook=partial(backward_hook, keep_input=self.keep_input,
                                             compute_input_fn=self._fn.inverse, compute_output_fn=self._fn.forward,
                                             valid_states=self._valid_states, state_index=self._state_counter,
                                             input_tensor=input_tensor, output_tensor=output_tensor))
                self._state_counter += 1

            y.detach_()  # Detaches y in-place (inbetween computations can now be discarded)
            y.requires_grad = self.training
        else:
            y = self._fn(xin)
        return y

    def inverse(self, yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            yin : :obj:`torch.Tensor`
                Input torch tensor.

        Returns
        -------
            :obj:`torch.Tensor`
                Output torch tensor x.

        """
        if not self.disable:
            y = yin.detach()  # Makes a detached copy which shares the storage
            x = self._fn.inverse(y)
            input_tensor = yin
            output_tensor = x
            # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
            if not self.keep_input_inverse:
                if not pytorch_version_one_and_above:
                    # PyTorch 0.4 way to clear storage
                    input_tensor.data.set_()
                else:
                    # PyTorch 1.0+ way to clear storage
                    input_tensor.storage().resize_(0)
            if self.training:
                self._valid_states.append(True)
                yin.register_hook(hook=partial(signal_hook, valid_states=self._valid_states, state_index=self._state_counter))
                x.register_hook(hook=partial(backward_hook, keep_input=self.keep_input_inverse,
                                             compute_input_fn=self._fn.forward, compute_output_fn=self._fn.inverse,
                                             valid_states=self._valid_states, state_index=self._state_counter,
                                             input_tensor=input_tensor, output_tensor=output_tensor))
                self._state_counter += 1
            x.detach_()  # Detaches x in-place (inbetween computations can now be discarded)
            x.requires_grad = self.training
        else:
            x = self._fn.inverse(yin)
        return x


class ReversibleBlock(ReversibleModule):
    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, keep_input_inverse=False,
                 implementation_fwd=-1, implementation_bwd=-1, adapter=None):
        """The ReversibleBlock

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
        warnings.warn("This class has been deprecated. Use the more flexible ReversibleModule class", DeprecationWarning)
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


def is_invertible_module(module_in, test_input, atol=1e-6):
    test_input = torch.rand(test_input.shape, dtype=test_input.dtype)
    if not hasattr(module_in, "inverse"):
        return False
    with torch.no_grad():
        if not torch.allclose(module_in.inverse(module_in(test_input)), test_input, atol=atol):
            return False
        if test_input is module_in(test_input):
            return False
        if test_input is module_in.inverse(test_input):
            return False
    return True
