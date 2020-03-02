import pytest
import gc
import numpy as np
import torch
import torch.nn
from memcnn.models.tests.test_revop import SubModuleStack, SubModule
import tracemalloc


def get_allocated_memory(device):
    return (
        torch.cuda.memory_allocated() / float(1024 ** 2)
        if device == "cuda"
        else tracemalloc.get_traced_memory()[0] / float(1024 ** 2)
    )


@pytest.mark.parametrize("coupling", ["additive", "affine"])
@pytest.mark.parametrize("keep_input", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_memory_saving_invertible_model_wrapper(device, coupling, keep_input):
    """Test memory saving of the invertible model wrapper

    * tests fitting a large number of images by creating a deep network requiring large
      intermediate feature maps for training

    * keep_input = False should use less memory than keep_input = True on both GPU and CPU RAM

    * input size in bytes:            np.prod((2, 10, 10, 10)) * 4 / 1024.0 =  7.8125 kB
      for a depth=5 this yields                                  7.8125 * 5 = 39.0625 kB

    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("This test requires a GPU to be available")

    gc.disable()
    gc.collect()

    with torch.set_grad_enabled(True):
        dims = [2, 10, 10, 10]
        depth = 5

        xx = torch.rand(*dims, device=device, dtype=torch.float32).requires_grad_()
        ytarget = torch.rand(*dims, device=device, dtype=torch.float32)

        # same convolution test
        network = SubModuleStack(
            SubModule(in_filters=5, out_filters=5),
            depth=depth,
            keep_input=keep_input,
            coupling=coupling,
            implementation_fwd=-1,
            implementation_bwd=-1,
        )
        network.to(device)
        network.train()
        network.zero_grad()
        optim = torch.optim.RMSprop(network.parameters())
        optim.zero_grad()
        mem_start = get_allocated_memory(device=device)
        tracemalloc.start()

        y = network(xx)
        gc.collect()
        mem_after_forward = get_allocated_memory(device=device)
        loss = torch.nn.MSELoss()(y, ytarget)
        optim.zero_grad()
        loss.backward()
        optim.step()
        gc.collect()
        gc.enable()
        tracemalloc.stop()

        memuse = (
            float(np.prod(dims + [depth, 4])) / float(1024 ** 2)
            if device == "cuda"
            else 0.0056
        )

        measured_memuse = mem_after_forward - mem_start

        assert (measured_memuse >= memuse) == keep_input
