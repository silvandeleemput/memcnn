import torch
import torch.nn as nn
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

class NN(nn.Module):
    """ Affine subnetwork:
        Copy function and output s and t
    """
    def __init__(self, module):
        super(NN, self).__init__()
        self.NN_logs = copy.deepcopy(module)
        self.NN_t = module

    def forward(self, x):
        s = torch.exp(self.NN_logs(x))
        t = self.NN_t(x)
        return s, t


class AffineBlock(nn.Module):
    def __init__(self, Fm, Gm=None, keep_input=False, implementation_fwd=1, implementation_bwd=1):
        """The AffineBlock

        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function

            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Gm is used as a Module)

            implementation_fwd : int
                Switch between different Affine Operation implementations for forward pass. Default = 1

            implementation_bwd : int
                Switch between different Affine Operation implementations for inverse pass. Default = 1

            keep_input : bool
                Retain the input information, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            implementation_fwd : int
                Switch between different Affine Operation implementations for forward pass. Default = 1

            implementation_bwd : int
                Switch between different Affine Operation implementations for inverse pass. Default = 1


        """
        super(AffineBlock, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = NN(Gm)
        self.Fm = NN(Fm)
        self.implementation_fwd = implementation_fwd
        self.implementation_bwd = implementation_bwd
        self.keep_input = keep_input

    def forward(self, x):
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]

        if self.implementation_fwd == 0:
            out = AffineBlockFunction.apply(*args)
        elif self.implementation_fwd == 1:
            out = AffineBlockFunction2.apply(*args)
        else:
            raise NotImplementedError("Selected implementation ({}) not implemented..."
                                      .format(self.implementation_fwd))

        # clears the input data as it can be reversed on the backward pass
        if not self.keep_input:
            x.data.set_()

        return out

    def inverse(self, y):
        args = [y, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]

        if self.implementation_bwd == 0:
            x = AffineBlockInverseFunction.apply(*args)
        elif self.implementation_bwd == 1:
            x = AffineBlockInverseFunction2.apply(*args)
        else:
            raise NotImplementedError("Inverse for selected implementation ({}) not implemented..."
                                      .format(self.implementation_bwd))

        # clears the input data as it can be reversed on the backward pass
        if not self.keep_input:
            y.data.set_()

        return x

class AffineBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the affine block computes:
        {x1, x2} = x
        {log_s1, t1} = Fm(x2)
        s1 = exp(log_s1)
        y1 = s1 * x1 + t1
        {log_s2, t2} = Gm(y1)
        s2 = exp(log_s2)
        y2 = s2 * x2 + t2
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

        with torch.no_grad():
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)

            y1 = (x1 * fmr1) + fmr2
            x1.set_()
            del x1
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)
            y2 = (x2 * gmr1) + gmr2
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
        x, output = ctx.saved_tensors
        y1, y2 = torch.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.shape[1] % 2 == 0)

        with set_grad_enabled(False):
            # recompute x
            z1_stop = y1
            z1_stop.requires_grad = True
            GWeights = [p for p in Gm.parameters()]
            gmr1, gmr2 = Gm.forward(z1_stop)
            x2 = (y2 - gmr2) / gmr1
            fmr1, fmr2 = Fm.forward(x2)
            x1 = (y1 - fmr2) / fmr1


        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1.requires_grad = True
            x2.requires_grad = True

            fmr1, fmr2 = Fm.forward(x2)
            y1 = x1 * fmr1 + fmr2
            gmr1, gmr2 = Gm.forward(y1)
            y2 = x2 * gmr1 + gmr2
            y = torch.cat([y1, y2], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(y, (x1, x2 ) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output)

            GWgrads = dd[2:2+len(GWeights)]
            FWgrads = dd[2+len(GWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            y1.detach_()
            y2.detach_()
            del y1, y2

        # restore input
        x.set_(torch.cat([x1, x2], dim=1).contiguous())

        return (grad_input, None, None) + FWgrads + GWgrads


class AffineBlockInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward inverse pass for the affine block computes:
        {y1, y2} = y
        {log_s2, t2} = Gm(y1)
        s2 = exp(log_s2)
        x2 = (y2 - t2) / s2
        {log_s1, t1} = Fm(x2)
        s1 = exp(log_s1)
        x1 = (y1 - t1) / s1
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

        with torch.no_grad():
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs
            y1var = y1
            
            gmr1, gmr2 = Gm.forward(y1var)

            x2 = (y2 - gmr2) / gmr1
            y2.set_()
            del y2
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)

            x1 = (y1 - fmr2 ) / fmr1
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
        y, output = cty.saved_tensors
        x1, x2 = torch.chunk(output, 2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.shape[1] % 2 == 0)

        with set_grad_enabled(False):
            # recompute y
            z1_stop = x2
            z1_stop.requires_grad = True
            FWeights = [p for p in Fm.parameters()]
            fmr1, fmr2 = Fm.forward(z1_stop)
            y1 = (fmr1 * x1) + fmr2
            gmr1, gmr2 = Gm.forward(y1)
            y2 = (gmr1 * x2) + gmr2


        with set_grad_enabled(True):
            # compute outputs building a sub-graph
            y2.requires_grad = True
            y1.requires_grad = True

            gmr1, gmr2 = Gm.forward(y1) #
            x2 = (y2 - gmr2) / gmr1
            fmr1, fmr2 = Fm.forward(x2)
            x1 = (y1 - fmr2) / fmr1
            x = torch.cat([x1, x2], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(x, (y2, y1 ) + tuple(Fm.parameters()) + tuple(Gm.parameters()), grad_output)

            FWgrads = dd[2:2+len(FWeights)]
            GWgrads = dd[2+len(FWeights):]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            x1.detach_()
            x2.detach_()
            del x1, x2

        # restore input
        y.set_(torch.cat([y1, y2], dim=1).contiguous())

        return (grad_input, None, None) + FWgrads + GWgrads


class AffineBlockFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the affine block computes:
        {x1, x2} = x
        {log_s1, t1} = Fm(x2)
        s1 = exp(log_s1)
        y1 = s1 * x1 + t1
        {log_s2, t2} = Gm(y1)
        s2 = exp(log_s2)
        y2 = s2 * x2 + t2
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

        with torch.no_grad():
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)

            y1 = x1 * fmr1 + fmr2
            x1.set_()
            del x1
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)
            y2 = x2 * gmr1 + gmr2
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
        x, output = ctx.saved_tensors

        with set_grad_enabled(False):
            y1, y2 = torch.chunk(output, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.shape[1] % 2 == 0)
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)
            y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        with set_grad_enabled(True):
            z1_stop = y1
            z1_stop.requires_grad=True

            G_z11, G_z12 = Gm.forward(z1_stop)
            x2 = (y2 - G_z12) / G_z11
            x2_stop = x2.detach()
            x2_stop.requires_grad=True

            F_x21, F_x22 = Fm.forward(x2_stop)
            x1 = (y1 - F_x22) / F_x21
            x1_stop = x1.detach()
            x1_stop.requires_grad=True

            # restore input
            x.set_(torch.cat([x1, x2], dim=1).contiguous())

            # compute outputs building a sub-graph
            z1 = x1_stop * F_x21 + F_x22
            y2_ = x2_stop * G_z11 + G_z12
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


class AffineBlockInverseFunction2(torch.autograd.Function):
    @staticmethod
    def forward(cty, y, Fm, Gm, *weights):
        """Forward pass for the affine block computes:

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

        with torch.no_grad():
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs
            y1var = y1
            gmr1, gmr2 = Gm.forward(y1var)

            x2 = (y2 - gmr2) / gmr1
            y2.set_()
            del y2
            x2var = x2
            fmr1, fmr2 = Fm.forward(x2var)
            x1 = (y1 - fmr2) / fmr1
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
        y, output = cty.saved_tensors

        with set_grad_enabled(False):
            x1, x2 = torch.chunk(output, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # partition output gradient also on channels
            assert(grad_output.shape[1] % 2 == 0)
            x1_grad, x2_grad = torch.chunk(grad_output, 2, dim=1)
            x1_grad, x2_grad = x1_grad.contiguous(), x2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, y1_stop, GW, FW
        # Also recompute inputs (y1, y2) from outputs (x1, x2)
        with set_grad_enabled(True):
            z1_stop = x2
            z1_stop.requires_grad=True

            F_z11, F_z12 = Fm.forward(z1_stop)
            y1 = x1 * F_z11 + F_z12
            y1_stop = y1.detach()
            y1_stop.requires_grad=True

            G_y11, G_y12 = Gm.forward(y1_stop)
            y2 = x2 * G_y11 + G_y12
            y2_stop = y2.detach()
            y2_stop.requires_grad=True

            # restore input
            y.set_(torch.cat([y1, y2], dim=1).contiguous())

            # compute outputs building a sub-graph
            z1 = (y2_stop - G_y12) / G_y11
            x1_ = (y1_stop - F_z12) / F_z11
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



