import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import Module


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class CrossEntropyLossTF(Module):
    def __init__(self):
        super(CrossEntropyLossTF, self).__init__()

    def forward(self, Ypred, Y, W=None):
        _assert_no_grad(Y)
        lsm = nn.Softmax(dim=1)
        y_onehot = torch.FloatTensor(Ypred.shape[0], Ypred.shape[1])
        if Ypred.is_cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, Y.data.view(-1,1), 1)
        if W is not None:
            y_onehot = y_onehot * torch.from_numpy(W)
        y_onehot = Variable(y_onehot)
        return torch.mean(-y_onehot * torch.log(lsm(Ypred))) * Ypred.shape[1]
