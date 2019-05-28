import torch
import torch.nn as nn
import warnings
from memcnn.models.additive import AdditiveBlock
from memcnn.models.affine import AffineBlock
from memcnn.models.utils import pytorch_version_one_and_above


warnings.filterwarnings(action='ignore', category=UserWarning)


class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, implementation_fwd=1, implementation_bwd=1):
        """The ReversibleBlock

        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function

            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)

            coupling: str
                Type of coupling ['additive', 'affine']. Default = 'additive'

            keep_input : bool
                Retain the input information, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            implementation_fwd : int
                Switch between different Operation implementations for forward training. Default = 1
                -1 : Naive implementation without reconstruction on the backward pass (keep_input should be True)
                 0 : Memory efficient implementation, compute gradients directly on y
                 1 : Memory efficient implementation, similar to approach in Gomez et al. 2017

            implementation_bwd : int
                Switch between different Operation implementations for backward training. Default = 1
                -1 : Naive implementation without reconstruction on the backward pass (keep_input should be True)
                 0 : Memory efficient implementation, compute gradients directly on y
                 1 : Memory efficient implementation, similar to approach in Gomez et al. 2017

        """
        super(ReversibleBlock, self).__init__()
        self.keep_input = keep_input
        if coupling == 'additive':
            self.rev_block = AdditiveBlock(Fm, Gm, implementation_fwd, implementation_bwd)
        elif coupling == 'affine':
            self.rev_block = AffineBlock(Fm, Gm, implementation_fwd, implementation_bwd)
        else:
            raise NotImplementedError('Unknown coupling method: %s' % coupling)

    def forward(self, x):
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
        x = self.rev_block.inverse(y)
        # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
        if not self.keep_input:
            if not pytorch_version_one_and_above:
                # PyTorch 0.4 way to clear storage
                y.data.set_()
            else:
                # PyTorch 1.0+ way to clear storage
                y.storage().resize_(0)

        return x
