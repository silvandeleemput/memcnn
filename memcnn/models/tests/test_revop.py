import warnings
import pytest
import torch
import torch.nn
import numpy as np
import copy
from memcnn.models.affine import AffineAdapterNaive, AffineAdapterSigmoid
from memcnn import ReversibleBlock
from memcnn.models.revop import ReversibleModule, create_coupling


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class SubModule(torch.nn.Module):
    def __init__(self, in_filters, out_filters):
        super(SubModule, self).__init__()
        self.bn = torch.nn.BatchNorm2d(out_filters)
        self.conv = torch.nn.Conv2d(in_filters, out_filters, (3, 3), padding=1)

    def forward(self, x):
        return self.bn(self.conv(x))


class SubModuleStack(torch.nn.Module):
    def __init__(self, Gm, coupling='additive', depth=10, implementation_fwd=-1, implementation_bwd=-1,
                 keep_input=False, adapter=None):
        super(SubModuleStack, self).__init__()
        fn = create_coupling(Fm=Gm, Gm=Gm, coupling=coupling, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd, adapter=adapter)
        self.stack = torch.nn.Sequential(
            *[ReversibleModule(fn=fn, keep_input=keep_input) for _ in range(depth)]
        )

    def forward(self, x):
        return self.stack(x)


def is_memory_cleared(var, isclear, shape):
    if isclear:
        return var.storage().size() == 0
    else:
        return var.storage().size() > 0 and var.shape == shape


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
def test_reversible_block_notimplemented(coupling):
    fm = torch.nn.Conv2d(10, 10, (3, 3), padding=1)
    X = torch.zeros(1, 20, 10, 10)
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            f = ReversibleBlock(fm, coupling=coupling, implementation_bwd=0, implementation_fwd=-2,
                                      adapter=AffineAdapterNaive)
            assert isinstance(f, ReversibleModule)
            f.forward(X)
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            f = ReversibleBlock(fm, coupling=coupling, implementation_bwd=-2, implementation_fwd=0,
                                      adapter=AffineAdapterNaive)
            assert isinstance(f, ReversibleModule)
            f.inverse(X)
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            ReversibleBlock(fm, coupling='unknown', implementation_bwd=-2, implementation_fwd=0,
                                  adapter=AffineAdapterNaive)


@pytest.mark.parametrize('coupling,adapter', [('additive', None),
                                              ('affine', AffineAdapterNaive),
                                              ('affine', AffineAdapterSigmoid)])
@pytest.mark.parametrize('bwd', [False, True])
@pytest.mark.parametrize('keep_input', [False, True])
@pytest.mark.parametrize('keep_input_inverse', [False, True])
def test_reversible_module_fwd_bwd(coupling, adapter, bwd, keep_input, keep_input_inverse):
    """ReversibleBlock test of the memory saving forward and backward passes

    * test inversion Y = RB(X) and X = RB.inverse(Y)
    * test training the block for a single step and compare weights for implementations: 0, 1
    * test automatic discard of input X and its retrieval after the backward pass
    * test usage of BN to identify non-contiguous memory blocks

    """
    for seed in range(10):
        set_seeds(seed)
        dims = (2, 10, 8, 8)
        data = torch.rand(*dims, dtype=torch.float32)
        target_data = torch.rand(*dims, dtype=torch.float32)
        Gm = SubModule(in_filters=5, out_filters=5 if coupling == 'additive' or adapter is AffineAdapterNaive else 10)
        s_grad = [p.data.numpy().copy() for p in Gm.parameters()]
        # test with zero padded convolution
        with torch.set_grad_enabled(True):
            X = data.clone().requires_grad_()

            Ytarget = target_data.clone()

            Xshape = X.shape

            Gm2 = copy.deepcopy(Gm)
            Gm3 = copy.deepcopy(Gm)
            fn = create_coupling(Fm=Gm2, Gm=Gm3, coupling=coupling, implementation_fwd=-1,
                                 implementation_bwd=-1, adapter=adapter)
            rb = ReversibleModule(fn=fn, keep_input=keep_input, keep_input_inverse=keep_input_inverse)

            rb.train()
            rb.zero_grad()

            optim = torch.optim.RMSprop(rb.parameters())
            optim.zero_grad()
            if not bwd:
                Xin = X.clone().requires_grad_()
                Y = rb(Xin)
                #Yrev = torch.tensor(Y, requires_grad=True)
                Yrev = torch.from_numpy(Y.cpu().detach().numpy().copy()).clone().requires_grad_()
                Xinv = rb.inverse(Yrev)
            else:
                Xin = X.clone().requires_grad_()
                Y = rb.inverse(Xin)
                Yrev = torch.from_numpy(Y.cpu().detach().numpy().copy()).clone().requires_grad_()
                #Yrev = torch.tensor(Y, requires_grad=True)
                Xinv = rb(Yrev)
            loss = torch.nn.MSELoss()(Y, Ytarget)

            # has input been retained/discarded after forward (and backward) passes?

            if not bwd:
                assert is_memory_cleared(Yrev, not keep_input_inverse, Xshape)
                assert is_memory_cleared(Xin, not keep_input, Xshape)
            else:
                assert is_memory_cleared(Xin, not keep_input_inverse, Xshape)
                assert is_memory_cleared(Yrev, not keep_input, Xshape)

            optim.zero_grad()

            loss.backward()
            optim.step()

            assert Y.shape == Xshape
            assert X.detach().numpy().shape == data.shape
            assert np.allclose(X.detach().numpy(), data, atol=1e-06)
            assert np.allclose(X.detach().numpy(), Xinv.detach().numpy(), atol=1e-05)  # Model is now trained and will differ
            grads = [p.detach().numpy().copy() for p in Gm2.parameters()]

            assert not np.allclose(grads[0], s_grad[0])


@pytest.mark.parametrize('coupling,adapter', [('additive', None),
                                              ('affine', AffineAdapterNaive),
                                              ('affine', AffineAdapterSigmoid)])
def test_reversible_module_chained(coupling, adapter):
    set_seeds(42)
    dims = (2, 10, 8, 8)
    data = torch.rand(*dims, dtype=torch.float32)
    target_data = torch.rand(*dims, dtype=torch.float32)
    with torch.set_grad_enabled(True):
        X = data.clone().requires_grad_()
        Ytarget = target_data.clone()

        Gm = SubModule(in_filters=5, out_filters=5 if coupling == 'additive' or adapter is AffineAdapterNaive else 10)
        rb = SubModuleStack(Gm, coupling=coupling, depth=2, keep_input=False, adapter=adapter, implementation_bwd=-1, implementation_fwd=-1)
        rb.train()
        optim = torch.optim.RMSprop(rb.parameters())

        rb.zero_grad()

        optim.zero_grad()

        Xin = X.clone()
        Y = rb(Xin)

        loss = torch.nn.MSELoss()(Y, Ytarget)

        optim.zero_grad()
        loss.backward()
        optim.step()

    assert not torch.isnan(loss)


@pytest.mark.parametrize("inverted", [False, True])
def test_reversible_module_disabled_versus_enabled(inverted):
    set_seeds(42)
    Gm = SubModule(in_filters=5, out_filters=5)

    coupling_fn = create_coupling(Fm=Gm, Gm=Gm, coupling='additive', implementation_fwd=-1,
                                  implementation_bwd=-1)
    rb = ReversibleModule(fn=coupling_fn, keep_input=False, keep_input_inverse=False)
    rb2 = ReversibleModule(fn=copy.deepcopy(coupling_fn), keep_input=False, keep_input_inverse=False)
    rb.eval()
    rb2.eval()
    rb2.disable = True
    with torch.no_grad():
        dims = (2, 10, 8, 8)
        data = torch.rand(*dims, dtype=torch.float32)
        X, X2 = data.clone().detach().requires_grad_(), data.clone().detach().requires_grad_()
        if not inverted:
            Y = rb(X)
            Y2 = rb2(X2)
        else:
            Y = rb.inverse(X)
            Y2 = rb2.inverse(X2)

        assert torch.allclose(Y, Y2)

        assert is_memory_cleared(X, True, dims)
        assert is_memory_cleared(X2, False, dims)


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
def test_reversible_module_simple_inverse(coupling):
    """ReversibleBlock inverse test

    * test inversion Y = RB(X) and X = RB.inverse(Y)

    """
    for seed in range(10):
        set_seeds(seed)
        # define some data
        X = torch.rand(2, 4, 5, 5).requires_grad_()

        # define an arbitrary reversible function
        coupling_fn = create_coupling(Fm=torch.nn.Conv2d(2, 2, 3, padding=1), coupling=coupling, implementation_fwd=-1,
                                      implementation_bwd=-1, adapter=AffineAdapterNaive)
        fn = ReversibleModule(fn=coupling_fn, keep_input=False, keep_input_inverse=False)

        # compute output
        Y = fn.forward(X.clone())

        # compute input from output
        X2 = fn.inverse(Y)

        # check that the inverted output and the original input are approximately similar
        assert np.allclose(X2.detach().numpy(), X.detach().numpy(), atol=1e-06)


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
def test_normal_vs_reversible_module(coupling):
    """ReversibleBlock test if similar gradients and weights results are obtained after similar training

    * test training the block for a single step and compare weights and grads for implementations: 0, 1
    * test against normal non Reversible Block function
    * test if recreated input and produced output are contiguous

    """
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
        Xin = X.clone().requires_grad_()
        coupling_fn = create_coupling(Fm=c1_2, Gm=c2_2, coupling=coupling, implementation_fwd=-1,
                                      implementation_bwd=-1, adapter=AffineAdapterNaive)
        fn = ReversibleModule(fn=coupling_fn, keep_input=False, keep_input_inverse=False)

        Y = fn.forward(Xin)
        loss2 = torch.mean(Y)

        # define the reversible function without custom backprop and define graph for model 2
        XX = X.clone().detach().requires_grad_()
        x1, x2 = torch.chunk(XX, 2, dim=1)
        if coupling == 'additive':
            y1 = x1 + c1.forward(x2)
            y2 = x2 + c2.forward(y1)
        elif coupling == 'affine':
            fmr2 = c1.forward(x2)
            fmr1 = torch.exp(fmr2)
            y1 = (x1 * fmr1) + fmr2
            gmr2 = c2.forward(y1)
            gmr1 = torch.exp(gmr2)
            y2 = (x2 * gmr1) + gmr2
        else:
            raise NotImplementedError()
        YY = torch.cat([y1, y2], dim=1)

        loss = torch.mean(YY)

        # compute gradients manually
        grads = torch.autograd.grad(loss, (XX, c1.weight, c2.weight, c1.bias, c2.bias), None, retain_graph=True)

        # compute gradients and perform optimization model 2
        loss.backward()
        optim1.step()

        # gradients computed manually match those of the .backward() pass
        assert torch.equal(c1.weight.grad, grads[1])
        assert torch.equal(c2.weight.grad, grads[2])
        assert torch.equal(c1.bias.grad, grads[3])
        assert torch.equal(c2.bias.grad, grads[4])

        # weights differ after training a single model?
        assert not torch.equal(c1.weight, c1_2.weight)
        assert not torch.equal(c2.weight, c2_2.weight)
        assert not torch.equal(c1.bias, c1_2.bias)
        assert not torch.equal(c2.bias, c2_2.bias)

        # compute gradients and perform optimization model 1
        loss2.backward()
        optim2.step()

        # input is contiguous tests
        assert Xin.is_contiguous()
        assert Y.is_contiguous()

        # weights are approximately the same after training both models?
        assert np.allclose(c1.weight.data.numpy(), c1_2.weight.data.numpy()) #, atol=1e-06)
        assert np.allclose(c2.weight.data.numpy(), c2_2.weight.data.numpy())
        assert np.allclose(c1.bias.data.numpy(), c1_2.bias.data.numpy())
        assert np.allclose(c2.bias.data.numpy(), c2_2.bias.data.numpy())

        # gradients are approximately the same after training both models?
        assert np.allclose(c1.weight.grad.data.numpy(), c1_2.weight.grad.data.numpy()) #, atol=1e-06)
        assert np.allclose(c2.weight.grad.data.numpy(), c2_2.weight.grad.data.numpy())
        assert np.allclose(c1.bias.grad.data.numpy(), c1_2.bias.grad.data.numpy())
        assert np.allclose(c2.bias.grad.data.numpy(), c2_2.bias.grad.data.numpy())
