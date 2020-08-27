import warnings
import pytest
import random
import torch
import torch.nn
import numpy as np
import copy
from memcnn.models.affine import AffineAdapterNaive, AffineAdapterSigmoid, AffineCoupling
from memcnn.models.revop import InvertibleModuleWrapper, ReversibleBlock, create_coupling, \
     is_invertible_module, get_device_states, set_device_states
from memcnn.models.additive import AdditiveCoupling
from memcnn.models.tests.test_models import MultiplicationInverse, SubModule, SubModuleStack


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def is_memory_cleared(var, isclear, shape):
    if isclear:
        return var.storage().size() == 0
    else:
        return var.storage().size() > 0 and var.shape == shape


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('enabled', [True, False])
def test_get_set_device_states(device, enabled):
    shape = (1, 1, 10, 10)
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('This test requires a GPU to be available')
    X = torch.ones(shape, device=device)
    devices, states = get_device_states(X)
    assert len(states) == (1 if device == 'cuda' else 0)
    assert len(devices) == (1 if device == 'cuda' else 0)
    cpu_rng_state = torch.get_rng_state()
    Y = X * torch.rand(shape, device=device)
    with torch.random.fork_rng(devices=devices, enabled=True):
        if enabled:
            if device == 'cpu':
                torch.set_rng_state(cpu_rng_state)
            else:
                set_device_states(devices=devices, states=states)
        Y2 = X * torch.rand(shape, device=device)
    assert torch.equal(Y, Y2) == enabled


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
def test_reversible_block_notimplemented(coupling):
    fm = torch.nn.Conv2d(10, 10, (3, 3), padding=1)
    X = torch.zeros(1, 20, 10, 10)
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            f = ReversibleBlock(fm, coupling=coupling, implementation_bwd=0, implementation_fwd=-2,
                                      adapter=AffineAdapterNaive)
            assert isinstance(f, InvertibleModuleWrapper)
            f.forward(X)
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            f = ReversibleBlock(fm, coupling=coupling, implementation_bwd=-2, implementation_fwd=0,
                                      adapter=AffineAdapterNaive)
            assert isinstance(f, InvertibleModuleWrapper)
            f.inverse(X)
    with pytest.raises(NotImplementedError):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=DeprecationWarning)
            ReversibleBlock(fm, coupling='unknown', implementation_bwd=-2, implementation_fwd=0,
                                  adapter=AffineAdapterNaive)


@pytest.mark.parametrize('fn', [
    AdditiveCoupling(Fm=SubModule(), implementation_fwd=-1, implementation_bwd=-1),
    AffineCoupling(Fm=SubModule(), implementation_fwd=-1, implementation_bwd=-1, adapter=AffineAdapterNaive),
    AffineCoupling(Fm=SubModule(out_filters=10), implementation_fwd=-1, implementation_bwd=-1, adapter=AffineAdapterSigmoid),
    MultiplicationInverse()
])
@pytest.mark.parametrize('bwd', [False, True])
@pytest.mark.parametrize('keep_input', [False, True])
@pytest.mark.parametrize('keep_input_inverse', [False, True])
@pytest.mark.parametrize('preserve_rng_state', [False, True])
def test_invertible_module_wrapper_fwd_bwd(fn, bwd, keep_input, keep_input_inverse, preserve_rng_state):
    """InvertibleModuleWrapper tests for the memory saving forward and backward passes

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

        assert is_invertible_module(fn, test_input_shape=data.shape, atol=1e-4)

        # test with zero padded convolution
        with torch.set_grad_enabled(True):
            X = data.clone().requires_grad_()

            Ytarget = target_data.clone()

            Xshape = X.shape

            rb = InvertibleModuleWrapper(fn=fn, keep_input=keep_input,
                                         keep_input_inverse=keep_input_inverse,
                                         preserve_rng_state=preserve_rng_state)
            s_grad = [p.detach().clone() for p in rb.parameters()]

            rb.train()
            rb.zero_grad()

            optim = torch.optim.RMSprop(rb.parameters())
            optim.zero_grad()
            if not bwd:
                Xin = X.clone().requires_grad_()
                Y = rb(Xin)
                Yrev = Y.detach().clone().requires_grad_()
                Xinv = rb.inverse(Yrev)
            else:
                Xin = X.clone().requires_grad_()
                Y = rb.inverse(Xin)
                Yrev = Y.detach().clone().requires_grad_()
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
            assert X.detach().shape == data.shape
            assert torch.allclose(X.detach(), data, atol=1e-06)
            assert torch.allclose(X.detach(), Xinv.detach(), atol=1e-04)  # Model is now trained and will differ
            grads = [p.detach().clone() for p in rb.parameters()]

            assert not torch.allclose(grads[0], s_grad[0])


@pytest.mark.parametrize('coupling,adapter', [('additive', None),
                                              ('affine', AffineAdapterNaive),
                                              ('affine', AffineAdapterSigmoid)])
def test_chained_invertible_module_wrapper(coupling, adapter):
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

        loss.backward()
        optim.step()

    assert not torch.isnan(loss)


def test_chained_invertible_module_wrapper_shared_fwd_and_bwd_train_passes():
    set_seeds(42)
    Gm = SubModule(in_filters=5, out_filters=5)
    rb_temp = SubModuleStack(Gm=Gm, coupling='additive', depth=5, keep_input=True, adapter=None, implementation_bwd=-1,
                             implementation_fwd=-1)
    optim = torch.optim.SGD(rb_temp.parameters(), lr=0.01)

    initial_params = [p.detach().clone() for p in rb_temp.parameters()]
    initial_state = copy.deepcopy(rb_temp.state_dict())
    initial_optim_state = copy.deepcopy(optim.state_dict())

    dims = (2, 10, 8, 8)
    data = torch.rand(*dims, dtype=torch.float32)
    target_data = torch.rand(*dims, dtype=torch.float32)

    forward_outputs = []
    inverse_outputs = []
    for i in range(10):

        is_forward_pass = i % 2 == 0
        set_seeds(42)
        rb = SubModuleStack(Gm=Gm, coupling='additive', depth=5, keep_input=True,
                            adapter=None, implementation_bwd=-1,
                            implementation_fwd=-1, num_bwd_passes=2)
        rb.train()
        with torch.no_grad():
            for (name, p), p_initial in zip(rb.named_parameters(), initial_params):
                p.set_(p_initial)

        rb.load_state_dict(initial_state)
        optim = torch.optim.SGD(rb_temp.parameters(), lr=0.01)
        optim.load_state_dict(initial_optim_state)

        with torch.set_grad_enabled(True):
            X = data.detach().clone().requires_grad_()
            Ytarget = target_data.detach().clone()

            optim.zero_grad()

            if is_forward_pass:
                Y = rb(X)
                Xinv = rb.inverse(Y)
                Xinv2 = rb.inverse(Y)
                Xinv3 = rb.inverse(Y)
            else:
                Y = rb.inverse(X)
                Xinv = rb(Y)
                Xinv2 = rb(Y)
                Xinv3 = rb(Y)

            for item in [Xinv, Xinv2, Xinv3]:
                assert torch.allclose(X, item, atol=1e-04)

            loss = torch.nn.MSELoss()(Xinv, Ytarget)
            assert not torch.isnan(loss)

            assert Xinv2.grad is None
            assert Xinv3.grad is None

            loss.backward()

            assert Y.grad is not None
            assert Xinv.grad is not None
            assert Xinv2.grad is None
            assert Xinv3.grad is None

            loss2 = torch.nn.MSELoss()(Xinv2, Ytarget)
            assert not torch.isnan(loss2)

            loss2.backward()

            assert Xinv2.grad is not None

            optim.step()

            if is_forward_pass:
                forward_outputs.append(Y.detach().clone())
            else:
                inverse_outputs.append(Y.detach().clone())

    for i in range(4):
        assert torch.allclose(forward_outputs[-1], forward_outputs[i], atol=1e-06)
        assert torch.allclose(inverse_outputs[-1], inverse_outputs[i], atol=1e-06)


@pytest.mark.parametrize("inverted", [False, True])
def test_invertible_module_wrapper_disabled_versus_enabled(inverted):
    set_seeds(42)
    Gm = SubModule(in_filters=5, out_filters=5)

    coupling_fn = create_coupling(Fm=Gm, Gm=Gm, coupling='additive', implementation_fwd=-1,
                                  implementation_bwd=-1)
    rb = InvertibleModuleWrapper(fn=coupling_fn, keep_input=False, keep_input_inverse=False)
    rb2 = InvertibleModuleWrapper(fn=copy.deepcopy(coupling_fn), keep_input=False, keep_input_inverse=False)
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
def test_invertible_module_wrapper_simple_inverse(coupling):
    """InvertibleModuleWrapper inverse test"""
    for seed in range(10):
        set_seeds(seed)
        # define some data
        X = torch.rand(2, 4, 5, 5).requires_grad_()

        # define an arbitrary reversible function
        coupling_fn = create_coupling(Fm=torch.nn.Conv2d(2, 2, 3, padding=1), coupling=coupling, implementation_fwd=-1,
                                      implementation_bwd=-1, adapter=AffineAdapterNaive)
        fn = InvertibleModuleWrapper(fn=coupling_fn, keep_input=False, keep_input_inverse=False)

        # compute output
        Y = fn.forward(X.clone())

        # compute input from output
        X2 = fn.inverse(Y)

        # check that the inverted output and the original input are approximately similar
        assert torch.allclose(X2.detach(), X.detach(), atol=1e-06)


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
def test_normal_vs_invertible_module_wrapper(coupling):
    """InvertibleModuleWrapper test if similar gradients and weights results are obtained after similar training"""
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
        fn = InvertibleModuleWrapper(fn=coupling_fn, keep_input=False, keep_input_inverse=False)

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
        assert torch.allclose(c1.weight.detach(), c1_2.weight.detach())
        assert torch.allclose(c2.weight.detach(), c2_2.weight.detach())
        assert torch.allclose(c1.bias.detach(), c1_2.bias.detach())
        assert torch.allclose(c2.bias.detach(), c2_2.bias.detach())

        # gradients are approximately the same after training both models?
        assert torch.allclose(c1.weight.grad.detach(), c1_2.weight.grad.detach())
        assert torch.allclose(c2.weight.grad.detach(), c2_2.weight.grad.detach())
        assert torch.allclose(c1.bias.grad.detach(), c1_2.bias.grad.detach())
        assert torch.allclose(c2.bias.grad.detach(), c2_2.bias.grad.detach())
