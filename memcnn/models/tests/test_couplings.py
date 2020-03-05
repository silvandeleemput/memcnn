import torch
import torch.nn
import pytest
import copy
import warnings

from memcnn import create_coupling, InvertibleModuleWrapper
from memcnn.models.tests.test_revop import set_seeds
from memcnn.models.tests.test_models import SubModule
from memcnn.models.affine import AffineAdapterNaive, AffineBlock
from memcnn.models.additive import AdditiveBlock


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
@pytest.mark.parametrize('bwd', [False, True])
@pytest.mark.parametrize('implementation', [-1, 0, 1])
def test_coupling_implementations_against_reference(coupling, bwd, implementation):
    """Test if similar gradients and weights results are obtained after similar training for the couplings"""
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        for seed in range(10):
            set_seeds(seed)

            X = torch.rand(2, 4, 5, 5)

            # define models and their copies
            c1 = torch.nn.Conv2d(2, 2, 3, padding=1)
            c2 = torch.nn.Conv2d(2, 2, 3, padding=1)
            c1_2 = copy.deepcopy(c1)
            c2_2 = copy.deepcopy(c2)

            # are weights between models the same, but do they differ between convolutions?
            assert torch.equal(c1.weight, c1_2.weight)
            assert torch.equal(c2.weight, c2_2.weight)
            assert torch.equal(c1.bias, c1_2.bias)
            assert torch.equal(c2.bias, c2_2.bias)
            assert not torch.equal(c1.weight, c2.weight)

            # define optimizers
            optim1 = torch.optim.SGD([e for e in c1.parameters()] + [e for e in c2.parameters()], 0.1)
            optim2 = torch.optim.SGD([e for e in c1_2.parameters()] + [e for e in c2_2.parameters()], 0.1)
            for e in [c1, c2, c1_2, c2_2]:
                e.train()

            # define an arbitrary reversible function and define graph for model 1
            XX = X.detach().clone().requires_grad_()
            coupling_fn = create_coupling(Fm=c1, Gm=c2, coupling=coupling, implementation_fwd=-1,
                                          implementation_bwd=-1, adapter=AffineAdapterNaive)
            Y = coupling_fn.inverse(XX) if bwd else coupling_fn.forward(XX)
            loss = torch.mean(Y)

            # define the reversible function without custom backprop and define graph for model 2
            XX2 = X.detach().clone().requires_grad_()
            coupling_fn2 = create_coupling(Fm=c1_2, Gm=c2_2, coupling=coupling, implementation_fwd=implementation,
                                           implementation_bwd=implementation, adapter=AffineAdapterNaive)
            Y2 = coupling_fn2.inverse(XX2) if bwd else coupling_fn2.forward(XX2)
            loss2 = torch.mean(Y2)

            # compute gradients manually
            grads = torch.autograd.grad(loss2, (XX2, c1_2.weight, c2_2.weight, c1_2.bias, c2_2.bias), None, retain_graph=True)

            # compute gradients using backward and perform optimization model 2
            loss2.backward()
            optim2.step()

            # gradients computed manually match those of the .backward() pass
            assert torch.equal(c1_2.weight.grad, grads[1])
            assert torch.equal(c2_2.weight.grad, grads[2])
            assert torch.equal(c1_2.bias.grad, grads[3])
            assert torch.equal(c2_2.bias.grad, grads[4])

            # weights differ after training a single model?
            assert not torch.equal(c1.weight, c1_2.weight)
            assert not torch.equal(c2.weight, c2_2.weight)
            assert not torch.equal(c1.bias, c1_2.bias)
            assert not torch.equal(c2.bias, c2_2.bias)

            # compute gradients and perform optimization model 1
            loss.backward()
            optim1.step()

            # weights are approximately the same after training both models?
            assert torch.allclose(c1.weight.detach(), c1_2.weight.detach())
            assert torch.allclose(c2.weight.detach(), c2_2.weight.detach())
            assert torch.allclose(c1.bias.detach(), c1_2.bias.detach())
            assert torch.allclose(c2.bias.detach(), c2_2.bias.detach())

            # gradients are approximately the same after training both models?
            assert torch.allclose(c1.weight.grad.detach(), c1_2.weight.grad.detach())
            assert torch.allclose(c2.weight.grad.detach(), c2_2.weight.grad.detach())
            assert torch.allclose(c1.bias.grad.detach(), c1_2.bias.grad.detach())
            assert torch.allclose(c2.bias.grad.detach(), c2_2.bias.grad.detach())

            fn = InvertibleModuleWrapper(fn=coupling_fn, keep_input=False, keep_input_inverse=False)
            Yout = fn.inverse(XX) if bwd else fn.forward(XX)
            loss = torch.mean(Yout)
            loss.backward()
            assert XX.storage().size() > 0

            fn2 = InvertibleModuleWrapper(fn=coupling_fn2, keep_input=False, keep_input_inverse=False)
            Yout2 = fn2.inverse(XX2) if bwd else fn2.forward(XX2)
            loss = torch.mean(Yout2)
            loss.backward()
            assert XX2.storage().size() > 0


def test_legacy_additive_coupling():
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        AdditiveBlock(Fm=SubModule())


def test_legacy_affine_coupling():
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        AffineBlock(Fm=SubModule())
