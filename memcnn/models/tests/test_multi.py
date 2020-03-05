import pytest
import torch
from memcnn.models.revop import InvertibleModuleWrapper


class SplitChannels(torch.nn.Module):
    def __init__(self, split_location):
        self.split_location = split_location
        super(SplitChannels, self).__init__()

    def forward(self, x):
        return (x[:, :self.split_location].clone(),
                x[:, self.split_location:].clone())

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


class ConcatenateChannels(torch.nn.Module):
    def __init__(self, split_location):
        self.split_location = split_location
        super(ConcatenateChannels, self).__init__()

    def forward(self, x, y):
        return torch.cat([x, y], dim=1)

    def inverse(self, x):
        return (x[:, :self.split_location].clone(),
                x[:, self.split_location:].clone())

@pytest.mark.parametrize('disable', [True, False])
def test_multi(disable):
    split = InvertibleModuleWrapper(SplitChannels(2), disable = disable)
    concat = InvertibleModuleWrapper(ConcatenateChannels(2), disable = disable)

    conv_a = torch.nn.Conv2d(2, 2, 3)
    conv_b = torch.nn.Conv2d(1, 1, 3)

    x = torch.rand(1, 3, 32, 32)
    x.requires_grad = True

    a, b = split(x)
    a, b = conv_a(a), conv_b(b)
    y = concat(a, b)
    loss = torch.sum(y)
    loss.backward()

