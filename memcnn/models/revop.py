import torch
import torch.nn as nn
from torch.autograd import Variable
import copy


class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, implementation=0, keep_input=False):
        super(ReversibleBlock, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.implementation = implementation
        self.keep_input = keep_input

    def forward(self, x):
        # These functions should not store their activations during training (train mode),
        # but the weights need updates on the backward pass
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation == 0:
            out = ReversibleBlockFunction.apply(*args)
        elif self.implementation == 1:
            out = ReversibleBlockFunction2.apply(*args)
        else:
            raise NotImplementedError("Selected implementation ({}) not implemented..."
                                      .format(self.implementation))

        # Clears the input data as it can be reversed on the backward pass
        if not self.keep_input:
            x.data.set_()

        return out

    def inverse(self, y):
        if self.implementation == 0 or self.implementation == 1:
            assert (y.shape[1] % 2 == 0)  # assert if possible

            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute inputs from outputs
            x2 = y2 - self.Gm.forward(y1)
            x1 = y1 - self.Fm.forward(x2)

            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError("Inverse for selected implementation ({}) not implemented..."
                                      .format(self.implementation))
        return x


class ReversibleBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0) # assert if possible

        # store Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        # partition in two equally sized set of channels
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # compute outputs
        x2var = Variable(x2, requires_grad=False, volatile=True)
        fmr = Fm.forward(x2var).data

        y1 = x1 + fmr
        x1.set_()
        del x1
        y1var = Variable(y1, requires_grad=False, volatile=True)
        gmr = Gm.forward(y1var).data
        y2 = x2 + gmr
        x2.set_()
        del x2
        output = torch.cat([y1, y2], dim=1)
        y1.set_()
        y2.set_()
        del y1, y2

        # save the input and output variables
        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        x, output = ctx.saved_variables #[0]
        y1, y2 = Variable.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # check if output gradients can be partitioned on channels
        assert(grad_output.data.shape[1] % 2 == 0)

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        z1_stop = Variable(y1.data, requires_grad=True)

        G_z1 = Gm.forward(z1_stop)
        x2 = y2 - G_z1
        x2_stop = Variable(x2.data, requires_grad=True)

        F_x2 = Fm.forward(x2_stop)
        x1 = y1 - F_x2
        x1_stop = Variable(x1.data, requires_grad=True)

        # Compute outputs building a sub-graph
        z1 = x1_stop + F_x2
        y2_ = x2_stop + G_z1
        y1_ = z1

        # Perform full backward pass on graph...
        y = torch.cat([y1_, y2_], dim=1)
        dd = torch.autograd.grad(y, (x1_stop, x2_stop) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output, retain_graph=False)
        GWeights = [p for p in Gm.parameters()]
        GWgrads = dd[2:2+len(GWeights)]
        FWgrads = dd[2+len(GWeights):]
        x2_grad = dd[1]
        x1_grad = dd[0]

        grad_input = torch.cat([x1_grad, x2_grad], dim=1)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_

        # restore input
        x.data.set_(torch.cat([x1.data, x2.data], dim=1).contiguous())
        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlockFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}

        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction

        """
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0) # assert if possible

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        # partition in two equally sized set of channels
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # compute outputs
        x2var = Variable(x2, requires_grad=False, volatile=True)
        fmr = Fm.forward(x2var).data

        y1 = x1 + fmr
        x1.set_()
        del x1
        y1var = Variable(y1, requires_grad=False, volatile=True)
        gmr = Gm.forward(y1var).data
        y2 = x2 + gmr
        x2.set_()
        del x2
        output = torch.cat([y1, y2], dim=1)
        y1.set_()
        del y1
        y2.set_()
        del y2

        # save the input and output variables
        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        x, output = ctx.saved_variables #[0]
        y1, y2 = Variable.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.data.shape[1] % 2 == 0)
        y1_grad, y2_grad = Variable.chunk(grad_output, 2, dim=1)
        y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)

        z1_stop = Variable(y1.data, requires_grad=True)

        G_z1 = Gm.forward(z1_stop)
        x2 = y2 - G_z1
        x2_stop = Variable(x2.data, requires_grad=True)

        F_x2 = Fm.forward(x2_stop)
        x1 = y1 - F_x2
        x1_stop = Variable(x1.data, requires_grad=True)

        # Compute outputs building a sub-graph
        z1 = x1_stop + F_x2
        y2_ = x2_stop + G_z1
        y1_ = z1

        # Calculate the final gradients for
        dd = torch.autograd.grad(y2_, (z1_stop,) + tuple(Gm.parameters()), y2_grad, retain_graph=False)
        z1_grad = dd[0] + y1_grad
        GWgrads = dd[1:]

        dd = torch.autograd.grad(y1_, (x1_stop, x2_stop) + tuple(Fm.parameters()), z1_grad, retain_graph=False)

        FWgrads = dd[2:]
        x2_grad = dd[1] + y2_grad
        x1_grad = dd[0]
        grad_input = torch.cat([x1_grad, x2_grad], dim=1)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_

        # restore input
        x.data.set_(torch.cat([x1.data, x2.data], dim=1).contiguous())
        return (grad_input, None, None) + FWgrads + GWgrads
