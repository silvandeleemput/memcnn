# -*- coding: utf-8 -*-

import torch.nn as nn
import warnings
from memcnn.models.additive import AdditiveBlock
from memcnn.models.affine import AffineBlock
from memcnn.models.utils import pytorch_version_one_and_above


warnings.filterwarnings(action='ignore', category=UserWarning)


class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, keep_input_inverse=False,
                 implementation_fwd=1, implementation_bwd=1, adapter=None):
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
        super(ReversibleBlock, self).__init__()
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        if coupling == 'additive':
            self.rev_block = AdditiveBlock(Fm, Gm,
                                           implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
        elif coupling == 'affine':
            self.rev_block = AffineBlock(Fm, Gm, adapter=adapter,
                                         implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
        else:
            raise NotImplementedError('Unknown coupling method: %s' % coupling)

    def forward(self, x):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            x : :obj:`torch.Tensor`
                Input torch tensor.

        Returns
        -------
            :obj:`torch.Tensor`
                Output torch tensor y.

        """
        y = self.rev_block(x)
        # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
        if not self.keep_input:
            if not pytorch_version_one_and_above:
                # PyTorch 0.4 way to clear storage
                x.data.set_()
            else:
                # PyTorch 1.0+ way to clear storage
                x.storage().resize_(0)

        return y

    def inverse(self, y):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            y : :obj:`torch.Tensor`
                Input torch tensor.

        Returns
        -------
            :obj:`torch.Tensor`
                Output torch tensor x.

        """
        x = self.rev_block.inverse(y)
        # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
        if not self.keep_input_inverse:
            if not pytorch_version_one_and_above:
                # PyTorch 0.4 way to clear storage
                y.data.set_()
            else:
                # PyTorch 1.0+ way to clear storage
                y.storage().resize_(0)

        return x
