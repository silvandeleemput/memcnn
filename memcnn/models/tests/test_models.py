import torch
import torch.nn

from memcnn import create_coupling, InvertibleModuleWrapper


class MultiplicationInverse(torch.nn.Module):
    def __init__(self, factor=2):
        super(MultiplicationInverse, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1) * factor)

    def forward(self, x):
        return x * self.factor

    def inverse(self, y):
        return y / self.factor


class IdentityInverse(torch.nn.Module):
    def __init__(self, multiply_forward=False, multiply_inverse=False):
        super(IdentityInverse, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1))
        self.multiply_forward = multiply_forward
        self.multiply_inverse = multiply_inverse

    def forward(self, x):
        if self.multiply_forward:
            return x * self.factor
        else:
            return x

    def inverse(self, y):
        if self.multiply_inverse:
            return y * self.factor
        else:
            return y


class MultiSharedOutputs(torch.nn.Module):
    # pylint: disable=R0201
    def forward(self, x):
        y = x * x
        return y, y

    # pylint: disable=R0201
    def inverse(self, y, y2):
        x = torch.max(torch.sqrt(y), torch.sqrt(y2))
        return x


class SubModule(torch.nn.Module):
    def __init__(self, in_filters=5, out_filters=5):
        super(SubModule, self).__init__()
        self.bn = torch.nn.BatchNorm2d(out_filters)
        self.conv = torch.nn.Conv2d(in_filters, out_filters, (3, 3), padding=1)

    def forward(self, x):
        return self.bn(self.conv(x))


class SubModuleStack(torch.nn.Module):
    def __init__(self, Gm, coupling='additive', depth=10, implementation_fwd=-1, implementation_bwd=-1,
                 keep_input=False, adapter=None, num_bwd_passes=1):
        super(SubModuleStack, self).__init__()
        fn = create_coupling(Fm=Gm, Gm=Gm, coupling=coupling, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd, adapter=adapter)
        self.stack = torch.nn.ModuleList(
            [InvertibleModuleWrapper(fn=fn, keep_input=keep_input, keep_input_inverse=keep_input, num_bwd_passes=num_bwd_passes) for _ in range(depth)]
        )

    def forward(self, x):
        for rev_module in self.stack:
            x = rev_module.forward(x)
        return x

    def inverse(self, y):
        for rev_module in reversed(self.stack):
            y = rev_module.inverse(y)
        return y


class SplitChannels(torch.nn.Module):
    def __init__(self, split_location):
        self.split_location = split_location
        super(SplitChannels, self).__init__()

    def forward(self, x):
        return (x[:, :self.split_location, :].clone(),
                x[:, self.split_location:, :].clone())

    # pylint: disable=R0201
    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


class ConcatenateChannels(torch.nn.Module):
    def __init__(self, split_location):
        self.split_location = split_location
        super(ConcatenateChannels, self).__init__()

    # pylint: disable=R0201
    def forward(self, x, y):
        return torch.cat([x, y], dim=1)

    def inverse(self, x):
        return (x[:, :self.split_location, :].clone(),
                x[:, self.split_location:, :].clone())
