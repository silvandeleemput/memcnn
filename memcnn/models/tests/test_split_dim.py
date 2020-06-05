
import pytest
import torch

from memcnn import AdditiveCoupling, AffineAdapterNaive, AffineCoupling


class Check(torch.nn.Module):
    def __init__(self, dim, target_size):
        super(Check, self).__init__()
        self.dim = dim
        self.target_size = target_size

    def forward(self, fn_input):
        assert fn_input.size(self.dim) == self.target_size
        return fn_input


@pytest.mark.parametrize('dimension', [None, 0, 1, 2])
@pytest.mark.parametrize('coupling', [AdditiveCoupling, AffineCoupling])
@pytest.mark.parametrize('input_size', [(2, 2, 2), (2, 4, 8, 12)])
def test_split_dim(dimension, coupling, input_size):
    dim = 1 if dimension is None else dimension
    module = Check(dim, input_size[dim] // 2)
    coupling_args = dict(adapter=AffineAdapterNaive) if coupling.__name__ == 'AffineCoupling' else dict()
    if dimension is not None:
        coupling_args["split_dim"] = dimension
    model = coupling(module, **coupling_args)
    inp = torch.randn(input_size, requires_grad=False)
    output = model(inp)
    assert inp.shape == output.shape
