# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import warnings
from memcnn.models.additive import AdditiveBlock
from memcnn.models.affine import AffineBlock
from memcnn.models.utils import pytorch_version_one_and_above


warnings.filterwarnings(action='ignore', category=UserWarning)


def signal_hook(grad_output, rev_block, direction):
    state = rev_block._bwd_state[direction] == 0
    # print("SIGNAL {} {}".format(direction, state))
    rev_block._bwd_state[direction] = 1 if state else 0


def backward_hook(grad_output, keep_input, rev_block, compute_input_fn, compute_output_fn, direction, input_tensor, output_tensor):
    # print("BW hook runs {} {} {} {} {} {}".format(direction, grad_output.shape, keep_input, None, compute_input_fn.__name__, compute_output_fn.__name__))
    perform_action = rev_block._bwd_state[direction] == 0
    # print("HOOK {} {} in:{} out:{}".format(direction, perform_action, input_tensor.storage().size(), output_tensor.storage().size()))
    rev_block._bwd_state[direction] = 1 if perform_action else 0
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
        self._bwd_state = {"forward":0, "inverse":0}


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
                xin.register_hook(hook=partial(signal_hook, rev_block=self, direction="forward"))
                y.register_hook(hook=partial(backward_hook, keep_input=self.keep_input, rev_block=self,
                                             compute_input_fn=self._fn.inverse, compute_output_fn=self._fn.forward,
                                             direction="forward", input_tensor=input_tensor, output_tensor=output_tensor))

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
                yin.register_hook(hook=partial(signal_hook, rev_block=self, direction="inverse"))
                x.register_hook(hook=partial(backward_hook, keep_input=self.keep_input_inverse, rev_block=self,
                                             compute_input_fn=self._fn.forward, compute_output_fn=self._fn.inverse,
                                             direction="inverse", input_tensor=input_tensor, output_tensor=output_tensor))
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
        if coupling == 'additive':
            fn = AdditiveBlock(Fm, Gm,
                               implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
        elif coupling == 'affine':
            fn = AffineBlock(Fm, Gm, adapter=adapter,
                             implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
        else:
            raise NotImplementedError('Unknown coupling method: %s' % coupling)
        super(ReversibleBlock, self).__init__(fn, keep_input=keep_input, keep_input_inverse=keep_input_inverse)
