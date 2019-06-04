import pytest
import psutil
import os
import platform
import math
import gc
import multiprocessing as mp
import sys
import numpy as np


def memory_usage_psutil():
    """return the memory usage of the active process in MB"""
    process = psutil.Process(os.getpid())
    func = process.memory_full_info if platform.system() == 'Linux' else process.memory_info
    return func()[0] / float(2 ** 20)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('coupling', ['additive', 'affine'])
@pytest.mark.parametrize('implementation_fwd', [0, 1])
@pytest.mark.parametrize('keep_input', [True, False])
def test_memory_saving(device, coupling, implementation_fwd, keep_input):
    """Tests memory saving features of the reversible blocks

    * This spawns a new process for each test
    * Wrapping in a process is necessary to get reliable CPU RAM estimates.
    * No calls to torch can be made outside of the processes, because it will yield torch cuda init errors

    """
    p = mp.Process(target=memory_saving_sub, args=(device, coupling, implementation_fwd, keep_input))
    p.start()
    p.join()
    if p.exitcode == 42:
        pytest.skip('This test requires a GPU to be available')
    assert p.exitcode == 0


def memory_saving_sub(device, coupling, implementation_fwd, keep_input):
    """Test memory saving of the reversible block

    * tests fitting a large number of images by creating a deep network requiring large
      intermediate feature maps for training

    * keep_input = False should use less memory than keep_input = True on both GPU and CPU RAM

    * input size in bytes: np.prod((2, 10, 500, 250)) * 4 / (1024 ** 2)
                                                  (approx.) = 9.5 MB

    * for a depth=50 this yields 9.5 * 50 (approx.) = 475.0 MB


    Notes
    -----
    * CPU RAM estimates use psutil and are less precise than the GPU RAM estimates using the PyTorch
    * this function is intended to be boxed inside a separate process to tie the memory usage to the process

    """
    import torch
    import torch.nn
    import memcnn.models.revop as revop
    if device == 'cuda' and not torch.cuda.is_available():
        sys.exit(42)

    gc.disable()
    gc.collect()
    dims = [2, 10, 500, 250]
    depth = 50
    memuse = float(np.prod(dims + [depth, 4, ])) / float(1024 ** 2)
    data = torch.rand(*dims, device=device, dtype=torch.float32)
    target_data = torch.rand(*dims, device=device, dtype=torch.float32)

    class SubModule(torch.nn.Module):
        def __init__(self):
            super(SubModule, self).__init__()
            self.bn = torch.nn.BatchNorm2d(10 // 2)
            self.conv = torch.nn.Conv2d(10 // 2, 10 // 2, (3, 3), padding=1)

        def forward(self, x):
            return self.bn(self.conv(x))

    class SubModuleStack(torch.nn.Module):
        def __init__(self, gm, coupling='additive', depth=10, implementation_fwd=1,
                     implementation_bwd=1, keep_input=False):
            super(SubModuleStack, self).__init__()
            self.stack = torch.nn.Sequential(
                *[revop.ReversibleBlock(gm, gm, coupling=coupling, implementation_fwd=implementation_fwd,
                                        implementation_bwd=implementation_bwd,
                                        keep_input=keep_input) for _ in range(depth)]
            )

        def forward(self, x):
            return self.stack(x)

    implementation_bwd = 1
    # same convolution test
    xx = data
    ytarget = target_data
    network = SubModuleStack(SubModule(), depth=depth, keep_input=keep_input, coupling=coupling,
                             implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
    network.to(device)
    network.train()
    network.zero_grad()
    optim = torch.optim.RMSprop(network.parameters())
    optim.zero_grad()
    mem_start = memory_usage_psutil() if not device == 'cuda' else torch.cuda.memory_allocated() / float(1024 ** 2)

    y = network(xx)
    mem_after_forward = memory_usage_psutil() if not device == 'cuda' else torch.cuda.memory_allocated() / float(1024 ** 2)
    loss = torch.nn.MSELoss()(y, ytarget)
    optim.zero_grad()
    loss.backward()
    optim.step()
    gc.collect()
    mem_after_backward = memory_usage_psutil() if not device == 'cuda' else torch.cuda.memory_allocated() / float(1024 ** 2)
    gc.enable()
    if keep_input:
        assert math.floor(mem_after_forward - mem_start) >= memuse
    else:
        assert math.floor(mem_after_forward - mem_start) < (1 if device == 'cuda' else memuse)
    assert math.floor(mem_after_backward - mem_start) >= 9

    sys.exit(0)
