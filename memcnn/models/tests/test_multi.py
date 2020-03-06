import pytest
import torch
from memcnn.models.revop import InvertibleModuleWrapper, is_invertible_module
from memcnn.models.tests.test_models import SplitChannels, ConcatenateChannels


@pytest.mark.parametrize('disable', [True, False])
def test_multi(disable):
    split = InvertibleModuleWrapper(SplitChannels(2), disable = disable)
    concat = InvertibleModuleWrapper(ConcatenateChannels(2), disable = disable)

    assert is_invertible_module(split, test_input_shape=(1, 3, 32, 32))
    assert is_invertible_module(concat, test_input_shape=((1, 2, 32, 32), (1, 1, 32, 32)))

    conv_a = torch.nn.Conv2d(2, 2, 3)
    conv_b = torch.nn.Conv2d(1, 1, 3)

    x = torch.rand(1, 3, 32, 32)
    x.requires_grad = True

    a, b = split(x)
    a, b = conv_a(a), conv_b(b)
    y = concat(a, b)
    loss = torch.sum(y)
    loss.backward()
