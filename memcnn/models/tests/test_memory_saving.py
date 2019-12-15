import pytest
import gc
import numpy as np
import math
from collections import defaultdict
import torch
import torch.nn
from memcnn.models.tests.test_revop import SubModuleStack, SubModule


def readable_size(num_bytes):
    return '{:.2f}'.format(float(num_bytes) / float(1024 ** 2))


LEN = 79

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2 ** 9
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE = 2 ** 20


class MemReporter(object):
    """A memory reporter that collects tensors and memory usages
    Parameters:
        - model: an extra nn.Module can be passed to infer the name
        of Tensors
    """
    def __init__(self, model=None):
        self.tensor_name = defaultdict(list)
        self.device_mapping = defaultdict(list)
        self.device_tensor_stat = {}
        # to numbering the unknown tensors
        self.name_idx = 0
        if model is not None:
            assert isinstance(model, torch.nn.Module)
            # for model with tying weight, multiple parameters may share
            # the same underlying tensor
            for name, param in model.named_parameters():
                self.tensor_name[param].append(name)
        for param, name in self.tensor_name.items():
            self.tensor_name[param] = '+'.join(name)

    def _get_tensor_name(self, tensor):
        if tensor in self.tensor_name:
            name = self.tensor_name[tensor]
        # use numbering if no name can be inferred
        else:
            name = type(tensor).__name__ + str(self.name_idx)
            self.tensor_name[tensor] = name
            self.name_idx += 1
        return name

    def collect_tensor(self):
        """Collect all tensor objects tracked by python
        NOTICE:
            - the buffers for backward which is implemented in C++ are
            not tracked by python's reference counting.
            - the gradients(.grad) of Parameters is not collected, and
            I don't know why.
        """
        # FIXME: make the grad tensor collected by gc
        objects = gc.get_objects()
        tensors = [obj for obj in objects if isinstance(obj, torch.Tensor)]
        for t in tensors:
            self.device_mapping[str(t.device)].append(t)

    def get_stats(self):
        """Get the memory stat of tensors and then release them
        As a memory profiler, we cannot hold the reference to any tensors, which
        causes possibly inaccurate memory usage stats, so we delete the tensors after
        getting required stats"""
        visited_data = {}
        self.device_tensor_stat.clear()

        def get_tensor_stat(tensor):
            """Get the stat of a single tensor
            Returns:
                - stat: a tuple containing (tensor_name, tensor_size,
            tensor_numel, tensor_memory)
            """
            assert isinstance(tensor, torch.Tensor)

            name = self._get_tensor_name(tensor)

            numel = tensor.numel()
            element_size = tensor.element_size()
            fact_numel = tensor.storage().size()
            fact_memory_size = fact_numel * element_size
            # since pytorch allocate at least 512 Bytes for any tensor, round
            # up to a multiple of 512
            memory_size = math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE

            # tensor.storage should be the actual object related to memory
            # allocation
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                name = '{}(->{})'.format(
                    name,
                    visited_data[data_ptr],
                )
                # don't count the memory for reusing same underlying storage
                memory_size = 0
            else:
                visited_data[data_ptr] = name

            size = tuple(tensor.size())
            # torch scalar has empty size
            if not size:
                size = (1,)

            return (name, size, numel, memory_size)

        for device, tensors in self.device_mapping.items():
            tensor_stats = []
            for tensor in tensors:

                if tensor.numel() == 0:
                    continue
                stat = get_tensor_stat(tensor)  # (name, shape, numel, memory_size)
                tensor_stats.append(stat)
                if isinstance(tensor, torch.nn.Parameter):
                    if tensor.grad is not None:
                        # manually specify the name of gradient tensor
                        self.tensor_name[tensor.grad] = '{}.grad'.format(
                            self._get_tensor_name(tensor)
                        )
                        stat = get_tensor_stat(tensor.grad)
                        tensor_stats.append(stat)

            self.device_tensor_stat[device] = tensor_stats

        self.device_mapping.clear()

    def print_stats(self, verbose=False):
        # header
        show_reuse = verbose
        template_format = '{:<40s}{:>20s}{:>10s}'
        print(template_format.format('Element type', 'Size', 'Used MEM'))
        for device, tensor_stats in self.device_tensor_stat.items():
            print('-' * LEN)
            print('Storage on {}'.format(device))
            total_mem = 0
            total_numel = 0
            for stat in tensor_stats:
                name, size, numel, mem = stat
                if not show_reuse:
                    name = name.split('(')[0]
                print(template_format.format(
                    str(name),
                    str(size),
                    readable_size(mem),
                ))
                total_mem += mem
                total_numel += numel

            print('-'*LEN)
            print('Total Tensors: {} \tUsed Memory: {}'.format(
                total_numel, readable_size(total_mem),
            ))

            if device != torch.device('cpu'):
                # with torch.cuda.device(device): NOTE not supported in
                memory_allocated = torch.cuda.memory_allocated()
                print('The allocated memory on {}: {}'.format(
                    device, readable_size(memory_allocated),
                ))
                if memory_allocated != total_mem:
                    print('Memory differs due to the matrix alignment or'
                          ' invisible gradient buffer tensors')
            print('-'*LEN)

    def collect_stats(self):
        self.collect_tensor()
        self.get_stats()
        for _, tensor_stats in self.device_tensor_stat.items():
            total_mem = 0
            for stat in tensor_stats:
                total_mem += stat[3]

            return total_mem

    def report(self, verbose=False):
        """Interface for end-users to directly print the memory usage
        args:
            - verbose: flag to show tensor.storage reuse information
        """
        self.collect_tensor()
        self.get_stats()
        self.print_stats(verbose)


@pytest.mark.parametrize('coupling', ['additive', 'affine'])
@pytest.mark.parametrize('keep_input', [True, False])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_memory_saving_invertible_model_wrapper(device, coupling, keep_input):
    """Test memory saving of the invertible model wrapper

    * tests fitting a large number of images by creating a deep network requiring large
      intermediate feature maps for training

    * keep_input = False should use less memory than keep_input = True on both GPU and CPU RAM

    * input size in bytes:            np.prod((2, 10, 10, 10)) * 4 / 1024.0 =  7.8125 kB
      for a depth=5 this yields                                  7.8125 * 5 = 39.0625 kB

    Notes
    -----
    * CPU RAM estimates uses the MemReporter from https://github.com/Stonesjtu/pytorch_memlab which tries to estimate
      torch CPU RAM usage by collecting torch.Tensors from the gc. This reliably finds the difference between
      keep_input=(True|False), however it greatly underestimates the total memory consumptions w.r.t. the GPU RAM
      estimates using PyTorch cuda module. It appears to be missing some tensors.
      FIXME CPU Ram estimates - Sil

    """

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('This test requires a GPU to be available')

    mem_reporter = MemReporter()

    gc.disable()
    gc.collect()

    with torch.set_grad_enabled(True):
        dims = [2, 10, 10, 10]
        depth = 5

        xx = torch.rand(*dims, device=device, dtype=torch.float32).requires_grad_()
        ytarget = torch.rand(*dims, device=device, dtype=torch.float32)

        # same convolution test
        network = SubModuleStack(SubModule(in_filters=5, out_filters=5), depth=depth, keep_input=keep_input, coupling=coupling,
                                 implementation_fwd=-1, implementation_bwd=-1)
        network.to(device)
        network.train()
        network.zero_grad()
        optim = torch.optim.RMSprop(network.parameters())
        optim.zero_grad()
        mem_start = 0 if not device == 'cuda' else \
            torch.cuda.memory_allocated() / float(1024 ** 2)

        y = network(xx)
        gc.collect()
        mem_after_forward = mem_reporter.collect_stats() / float(1024 ** 2) if not device == 'cuda' else \
            torch.cuda.memory_allocated() / float(1024 ** 2)
        loss = torch.nn.MSELoss()(y, ytarget)
        optim.zero_grad()
        loss.backward()
        optim.step()
        gc.collect()
        # mem_after_backward = mem_reporter.collect_stats() / float(1024 ** 2) if not device == 'cuda' else \
        #     torch.cuda.memory_allocated() / float(1024 ** 2)
        gc.enable()

        if device == 'cpu':
            memuse = 0.05
        else:
            memuse = float(np.prod(dims + [depth, 4, ])) / float(1024 ** 2)

        measured_memuse = mem_after_forward - mem_start
        if keep_input:
            assert measured_memuse >= memuse
        else:
            assert measured_memuse < (1 if device == 'cuda' else memuse)
        # assert math.floor(mem_after_backward - mem_start) >= 9
