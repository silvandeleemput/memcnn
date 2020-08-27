import torch

# for backwards compatibility
use_context_mans = True

try:
    pytorch_version_one_and_above = int(torch.__version__[0]) > 0
except TypeError:
    pytorch_version_one_and_above = True

def _crop(inp, target_size):
    for t, s in zip(target_size, inp.size()):
        if t >= s:
            inp_slice.append(':')
            continue
        removed = s - t
        left = removed//2
        inp_slice.append('{:d}:{:d}'.format(left, removed-left))
    if all(s == ':' for s in inp_slice):
        return inp
    inp_slice = 'inp[' + ':'.join(inp_slice) + ']'
    inp_slice = eval(inp_slice)
    return inp_slice
    
def _crop_pad(inp, target_size):
    inp_slice = []
    size = inp.size()
    target_size = [s if t == -1 else t for s, t in zip(size, target_size)]

    if all(s == t for s, t in zip(size, target_size)):
        return inp

    if all(s >= t for s, t in zip(size, target_size)):
        inp_crop = inp
    else:
        inp_crop = _crop(inp, target_size)

    if all(s == t for s, t in zip(inp_crop.size(), target_size)):
        return inp_crop

    output = torch.zeros(target_size, device=inp.device, dtype=inp.dtype)
    out_crop = _crop(output, inp_crop.size())
    out_crop = inp_crop

    return output
    
class CropPad(torch.nn.Module):
    def __init__(self, input_size, target_size):
        self.target_size = target_size
        self.input_size = input_size
        
    def forward(self, inp):
        return _crop_pad(inp, self.target_size)
    
    def inverse(out):
        return _crop_pad(inp, self.input_size)
        
