import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from contextlib import contextmanager
import warnings


warnings.filterwarnings(action='ignore', category=UserWarning)


use_context_mans = int(torch.__version__[0]) * 100 + int(torch.__version__[2]) - \
                   (1 if 'a' in torch.__version__ else 0) > 3


@contextmanager
def set_grad_enabled(grad_mode):
    if not use_context_mans:
        yield
    else:
        with torch.set_grad_enabled(grad_mode) as c:
            yield [c]


class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, implementation=1, keep_input=False, implementation_inv=2):
        """The ReversibleBlock

        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function

            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Gm is used as a Module)

            implementation : int
                Switch between different Reversible Operation implementations. Default = 1

            implementation_inv : int
                Switch between different Reversible Operation implementations for inverse pass. Default = 2

            keep_input : bool
                Retain the input information, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        """
        super(ReversibleBlock, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.implementation = implementation
        self.implementation_inv = implementation_inv
        self.keep_input = keep_input

    def forward(self, x):
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]

        if self.implementation == 0:
            out = ReversibleBlockFunction.apply(*args)
        elif self.implementation == 1:
            out = ReversibleBlockFunction2.apply(*args)
        elif self.implementation == 2:
            assert (x.shape[1] % 2 == 0)  # assert if possible

            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute inputs from outputs
            y1 = x1 + self.Fm.forward(x2)
            y2 = x2 + self.Gm.forward(y1)

            out = torch.cat([y1, y2], dim=1)
        else:
            raise NotImplementedError("Selected implementation ({}) not implemented..."
                                      .format(self.implementation))

        # clears the input data as it can be reversed on the backward pass
        if not self.keep_input:
            x.data.set_()

        return out

    def inverse(self, y):
        args = [y, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]

        if self.implementation_inv == 0:
            x = ReversibleBlockInverseFunction.apply(*args)
        elif self.implementation_inv == 1:
            x = ReversibleBlockInverseFunction2.apply(*args)
        elif self.implementation_inv == 2:
            assert (y.shape[1] % 2 == 0)  # assert if possible

            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute inputs from outputs
            x2 = y2 - self.Gm.forward(y1)
            x1 = y1 - self.Fm.forward(x2)

            x = torch.cat([x1, x2], dim=1)
        else:
            raise NotImplementedError("Inverse for selected implementation ({}) not implemented..."
                                      .format(self.implementation_inv))
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
        assert(x.shape[1] % 2 == 0)  # assert if proper split is possible

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        with set_grad_enabled(False):
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            with warnings.catch_warnings():
                x2var = Variable(x2, requires_grad=False, volatile=True)
            fmr = Fm.forward(x2var).data

            y1 = x1 + fmr
            x1.set_()
            del x1
            with warnings.catch_warnings():
                y1var = Variable(y1, requires_grad=False, volatile=True)
            gmr = Gm.forward(y1var).data
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)
            y1.set_()
            y2.set_()
            del y1, y2

        # save the (empty) input and (non-empty) output variables
        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve weight references
        Fm, Gm = ctx.Fm, ctx.Gm

        # retrieve input and output references
        x, output = ctx.saved_variables
        y1, y2 = torch.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.data.shape[1] % 2 == 0)

        with set_grad_enabled(False):
            # recompute x
            z1_stop = Variable(y1.data, requires_grad=True)
            GWeights = [p for p in Gm.parameters()]
            x2 = y2 - Gm.forward(z1_stop)
            x1 = y1 - Fm.forward(x2)


        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1_ = Variable(x1.data, requires_grad=True)
            x2_ = Variable(x2.data, requires_grad=True)
            x1_.requires_grad = True
            x2_.requires_grad = True

            y1_ = x1_ + Fm.forward(x2_)
            y2_ = x2_ + Gm.forward(y1_)
            y = torch.cat([y1_, y2_], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(y, (x1_, x2_ ) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output)

            GWgrads = dd[2:2+len(GWeights)]
            FWgrads = dd[2+len(GWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            y1_.detach_()
            y2_.detach_()
            del y1_, y2_

        # restore input
        x.data.set_(torch.cat([x1, x2], dim=1).data.contiguous())

        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlockInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {y1, y2} = y
        x2 = y2 - Gm(y1)
        x1 = y1 - Fm(x2)
        output = {x1, x2}

        Parameters
        ----------
        cty : torch.autograd.function.RevNetInverseFunctionBackward
            The backward pass context object
        y : TorchTensor
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
        assert(y.shape[1] % 2 == 0)  # assert if proper split is possible

        # store partition size, Fm and Gm functions in context
        cty.Fm = Fm
        cty.Gm = Gm

        with set_grad_enabled(False):
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs
            with warnings.catch_warnings():
                y1var = Variable(y1, requires_grad=False, volatile=True)
            gmr = Gm.forward(y1var).data

            x2 = y2 - gmr
            y2.set_()
            del y2
            with warnings.catch_warnings():
                x2var = Variable(x2, requires_grad=False, volatile=True)
            fmr = Fm.forward(x2var).data
            x1 = y1 - fmr
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1)
            x1.set_()
            x2.set_()
            del x1, x2

        # save the (empty) input and (non-empty) output variables
        cty.save_for_backward(y, output)

        return output

    @staticmethod
    def backward(cty, grad_output):
        # retrieve weight references
        Fm, Gm = cty.Fm, cty.Gm

        # retrieve input and output references
        y, output = cty.saved_variables
        x1, x2 = torch.chunk(output, 2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.data.shape[1] % 2 == 0)

        with set_grad_enabled(False):
            # recompute y
            z1_stop = Variable(x2.data, requires_grad=True)
            FWeights = [p for p in Fm.parameters()]
            y1 = x1 + Fm.forward(z1_stop)
            y2 = x2 + Gm.forward(y1)


        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            y2_ = Variable(y2.data, requires_grad=True)
            y1_ = Variable(y1.data, requires_grad=True)
            y2_.requires_grad = True
            y1_.requires_grad = True

            x2_ = y2_ - Gm.forward(y1_)
            x1_ = y1_ - Fm.forward(x2_)
            x = torch.cat([x1_, x2_], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(x, (y2_, y1_ ) + tuple(Fm.parameters()) + tuple(Gm.parameters()), grad_output)

            FWgrads = dd[2:2+len(FWeights)]
            GWgrads = dd[2+len(FWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            x1_.detach_()
            x2_.detach_()
            del x1_, x2_

        # restore input
        y.data.set_(torch.cat([y1, y2], dim=1).data.contiguous())

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

        with set_grad_enabled(False):
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            with warnings.catch_warnings():
                x2var = Variable(x2, requires_grad=False, volatile=True)
            fmr = Fm.forward(x2var).data

            y1 = x1 + fmr
            x1.set_()
            del x1
            with warnings.catch_warnings():
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
        x, output = ctx.saved_variables

        with set_grad_enabled(False):
            y1, y2 = torch.chunk(output, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.data.shape[1] % 2 == 0)
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)
            y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        with set_grad_enabled(True):
            z1_stop = Variable(y1.data, requires_grad=True)

            G_z1 = Gm.forward(z1_stop)
            x2 = y2 - G_z1
            x2_stop = Variable(x2.data, requires_grad=True)

            F_x2 = Fm.forward(x2_stop)
            x1 = y1 - F_x2
            x1_stop = Variable(x1.data, requires_grad=True)

            # restore input
            x.data.set_(torch.cat([x1.data, x2.data], dim=1).contiguous())

            # compute outputs building a sub-graph
            z1 = x1_stop + F_x2
            y2_ = x2_stop + G_z1
            y1_ = z1

            # calculate the final gradients for the weights and inputs
            dd = torch.autograd.grad(y2_, (z1_stop,) + tuple(Gm.parameters()), y2_grad) #, retain_graph=False)
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

        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlockInverseFunction2(torch.autograd.Function):
    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {y1, y2} = y
    x2 = y2 - Gm(y1)
    x1 = y1 - Fm(x2)
        output = {x1, x2}

        Parameters
        ----------
        cty : torch.autograd.function.RevNetInverseFunctionBackward
            The backward pass context object
        y : TorchTensor
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
        assert(y.shape[1] % 2 == 0) # assert if possible

        # store partition size, Fm and Gm functions in context
        cty.Fm = Fm
        cty.Gm = Gm

        with set_grad_enabled(False):
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs
            with warnings.catch_warnings():
                y1var = Variable(y1, requires_grad=False, volatile=True)
            gmr = Gm.forward(y1var).data

            x2 = y2 - gmr
            y2.set_()
            del y2
            with warnings.catch_warnings():
                x2var = Variable(x2, requires_grad=False, volatile=True)
            fmr = Fm.forward(x2var).data
            x1 = y1 - fmr
            y1.set_()
            del y1
            output = torch.cat([x1, x2], dim=1)
            x1.set_()
            del x1
            x2.set_()
            del x2

        # save the input and output variables
        cty.save_for_backward(y, output)

        return output

    @staticmethod
    def backward(cty, grad_output):

        Fm, Gm = cty.Fm, cty.Gm
        # are all variable objects now
        y, output = cty.saved_variables

        with set_grad_enabled(False):
            x1, x2 = torch.chunk(output, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.data.shape[1] % 2 == 0)
            x1_grad, x2_grad = torch.chunk(grad_output, 2, dim=1)
            x1_grad, x2_grad = x1_grad.contiguous(), x2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, y1_stop, GW, FW
        # Also recompute inputs (y1, y2) from outputs (x1, x2)
        with set_grad_enabled(True):
            z1_stop = Variable(x2.data, requires_grad=True)

            F_z1 = Fm.forward(z1_stop)
            y1 = x1 + F_z1
            y1_stop = Variable(y1.data, requires_grad=True)

            G_y1 = Gm.forward(y1_stop)
            y2 = x2 + G_y1
            y2_stop = Variable(y2.data, requires_grad=True)

            # restore input
            y.data.set_(torch.cat([y1.data, y2.data], dim=1).contiguous())

            # compute outputs building a sub-graph
            z1 = y2_stop - G_y1
            x1_ = y1_stop - F_z1
            x2_ = z1

            # calculate the final gradients for the weights and inputs
            dd = torch.autograd.grad(x1_, (z1_stop,) + tuple(Fm.parameters()), x1_grad) #, retain_graph=False)
            z1_grad = dd[0] + x2_grad # + or - ?
            FWgrads = dd[1:]

            dd = torch.autograd.grad(x2_, (y2_stop, y1_stop) + tuple(Gm.parameters()), z1_grad, retain_graph=False)

            GWgrads = dd[2:]
            y1_grad = dd[1] + x1_grad # + or - ?
            y2_grad = dd[0]

            grad_input = torch.cat([y1_grad, y2_grad], dim=1)

            x1_.detach_()
            x2_.detach_()
            del x1_, x2_

        return (grad_input, None, None) + FWgrads + GWgrads

