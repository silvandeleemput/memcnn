import torch
import torch.nn as nn
from torch.nn.modules.module import Module


def _assert_no_grad(variable):
    msg = "nn criterions don't compute the gradient w.r.t. targets - please " \
          "mark these variables as not requiring gradients"
    assert not variable.requires_grad, msg  # nosec


class CrossEntropyLossTF(Module):
    def __init__(self):
        super(CrossEntropyLossTF, self).__init__()

    def forward(self, Ypred, Y, W=None):
        _assert_no_grad(Y)
        lsm = nn.Softmax(dim=1)
        y_onehot = torch.zeros(Ypred.shape[0], Ypred.shape[1], dtype=torch.float32, device=Ypred.device)
        y_onehot.scatter_(1, Y.data.view(-1, 1), 1)
        if W is not None:
            y_onehot = y_onehot * W
        return torch.mean(-y_onehot * torch.log(lsm(Ypred))) * Ypred.shape[1]
