import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from contextlib import contextmanager
import warnings
from memcnn.models.additive import AdditiveBlock
from memcnn.models.affine import AffineBlock


warnings.filterwarnings(action='ignore', category=UserWarning)


use_context_mans = int(torch.__version__[0]) * 100 + int(torch.__version__[2]) - \
                   (1 if 'a' in torch.__version__ else 0) > 3




class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, implementation_fwd=1, implementation_bwd=1):
        """The ReversibleBlock

        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function

            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Gm is used as a Module)

            coupling: str
                Type of coupling ['additive', 'affine']. Default = 'additive'

            keep_input : bool
                Retain the input information, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            implementation_fwd : int
                Switch between different Operation implementations for forward training. Default = 1

            implementation_bwd : int
                Switch between different Operation implementations for backward training. Default = 1

        """
        super(ReversibleBlock, self).__init__()

        if coupling == 'additive':
            self.rev_block = AdditiveBlock(Fm, Gm, keep_input, implementation_fwd, implementation_bwd)
        elif coupling == 'affine':
            self.rev_block = AffineBlock(Fm, Gm, keep_input, implementation_fwd, implementation_bwd)
        else:
            raise NotImplementedError('Unknown coupling method: %s' % coupling)

    def forward(self, x):
        return self.rev_block(x)

    def inverse(self, y):
        return self.rev_block.inverse(y)




