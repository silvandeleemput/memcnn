import pytest
import torch

from memcnn import is_invertible_module, InvertibleModuleWrapper, AdditiveCoupling
from memcnn.models.tests.test_models import IdentityInverse, MultiSharedOutputs, SubModule


def test_is_invertible_module_with_invalid_inverse():
    fn = IdentityInverse(multiply_inverse=True)
    with torch.no_grad():
        fn.factor.zero_()
    assert not is_invertible_module(fn, test_input_shape=(12, 12))


@pytest.mark.parametrize("random_seed", [1, 42, 900000])
def test_is_invertible_module_random_seeds(random_seed):
    fn = IdentityInverse(multiply_forward=True, multiply_inverse=True)
    assert is_invertible_module(fn, test_input_shape=(1, ), random_seed=random_seed)


def test_is_invertible_module_shared_outputs():
    fnb = MultiSharedOutputs()
    X = torch.rand(1, 2, 5, 5, dtype=torch.float32).requires_grad_()
    with pytest.warns(UserWarning):
        assert is_invertible_module(fnb, test_input_shape=(X.shape,), atol=1e-6)


def test_is_invertible_module_shared_tensors():
    fn = IdentityInverse()
    rm = InvertibleModuleWrapper(fn=fn, keep_input=True, keep_input_inverse=True)
    X = torch.rand(1, 2, 5, 5, dtype=torch.float32).requires_grad_()
    with pytest.warns(UserWarning):
        assert is_invertible_module(fn, test_input_shape=X.shape, atol=1e-6)
    rm.forward(X)
    fn.multiply_forward = True
    rm.forward(X)
    with pytest.warns(UserWarning):
        assert is_invertible_module(fn, test_input_shape=X.shape, atol=1e-6)
    rm.inverse(X)
    fn.multiply_inverse = True
    rm.inverse(X)
    assert is_invertible_module(fn, test_input_shape=X.shape, atol=1e-6)


def test_is_invertible_module():
    X = torch.zeros(1, 10, 10, 10)
    assert not is_invertible_module(torch.nn.Conv2d(10, 10, kernel_size=(1, 1)),
                                    test_input_shape=X.shape)
    fn = AdditiveCoupling(SubModule(), implementation_bwd=-1, implementation_fwd=-1)
    assert is_invertible_module(fn, test_input_shape=X.shape)
    class FakeInverse(torch.nn.Module):
        def forward(self, x):
            return x * 4

        def inverse(self, y):
            return y * 8
    assert not is_invertible_module(FakeInverse(), test_input_shape=X.shape)


def test_is_invertible_module_wrapped():
    X = torch.zeros(1, 10, 10, 10)
    assert not is_invertible_module(InvertibleModuleWrapper(torch.nn.Conv2d(10, 10, kernel_size=(1, 1))),
                                    test_input_shape=X.shape)
    fn = InvertibleModuleWrapper(AdditiveCoupling(SubModule(), implementation_bwd=-1, implementation_fwd=-1))
    assert is_invertible_module(fn, test_input_shape=X.shape)
    class FakeInverse(torch.nn.Module):
        def forward(self, x):
            return x * 4

        def inverse(self, y):
            return y * 8
    assert not is_invertible_module(InvertibleModuleWrapper(FakeInverse()), test_input_shape=X.shape)


@pytest.mark.parametrize("input_shape", (
    "string",
    (2.3, 1.4),
    None,
    True,
    ((1, 3, ), (12.4)),
    ((1, 3, ), False)
))
def test_is_invertible_module_type_check_input_shapes(input_shape):
    with pytest.raises(ValueError):
        is_invertible_module(module_in=IdentityInverse(multiply_forward=True, multiply_inverse=True), test_input_shape=input_shape)
