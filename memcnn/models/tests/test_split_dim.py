from typing import Union

import pytest
import torch

from memcnn import AdditiveCoupling, AffineCoupling, InvertibleModuleWrapper


class Check(torch.nn.Module):
    def __init__(self, dim, target_size):
        super(Check, self).__init__()
        self.dim = dim
        self.target_size = target_size

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        assert fn_input.size(self.dim) == self.target_size
        return fn_input


@pytest.mark.parametrize('dimension', [None, 1, 2])
@pytest.mark.parametrize('coupling', [AdditiveCoupling, AffineCoupling])
def test_resnet(dimension: int, coupling: Union[AdditiveCoupling, AffineCoupling]):
    dim = 1 if dimension is None else dimension
    module = Check(dim, 2)
    model = (coupling(module) if dimension is None
             else coupling(module, split_dim=dimension))
    model = InvertibleModuleWrapper(model)
    inp = torch.randn((2, 4, 4), requires_grad=False)
    out = model(inp)
