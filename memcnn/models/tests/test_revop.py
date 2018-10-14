import unittest
import torch
import torch.nn
from torch.autograd import Variable
import memcnn.models.revop as revop
import numpy as np
import copy


class ReversibleOperationsTestCase(unittest.TestCase):
    def test_reversible_block(self):
        """ReversibleBlock test

        * test inversion Y = RB(X) and X = RB.inverse(Y)
        * test training the block for a single step and compare weights for implementations 0, 1
        * test automatic discard of input X and its retrieval after the backward pass
        * test usage of BN to identify non-contiguous memory blocks

        """
        dims = (2, 10, 8, 8)
        data = np.random.random(dims).astype(np.float32)
        target_data = np.random.random(dims).astype(np.float32)
        impl_out, impl_grad = [], []

        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(10 // 2)
                self.conv = torch.nn.Conv2d(10 // 2, 10 // 2, (3, 3), padding=1)

            def forward(self, x):
                return self.bn(self.conv(x))

        Gm = SubModule()

        s_grad = [p.data.numpy().copy() for p in Gm.parameters()]
        implementations_fwd = [0, 0, 1, 1]
        implementations_bwd = [0, 0, 1, 1]
        for coupling in ['additive', 'affine']:
            for keep_input in [False, True]:
                for implementation_fwd in implementations_fwd:
                    for implementation_bwd in implementations_bwd:
                        # same convolution test
                        X = Variable(torch.from_numpy(data.copy()))
                        Ytarget = Variable(torch.from_numpy(target_data.copy()))
                        Xshape = X.shape
                        Gm2 = copy.deepcopy(Gm)
                        rb = revop.ReversibleBlock(Gm2, coupling=coupling,
                                                   keep_input=keep_input,
                                                   implementation_fwd=implementation_fwd,
                                                   implementation_bwd=implementation_bwd)
                        rb.train()
                        rb.zero_grad()

                        optim = torch.optim.RMSprop(rb.parameters())
                        optim.zero_grad()

                        Y = rb(X)
                        Xinv = rb.inverse(Y.clone())
                        loss = torch.nn.MSELoss()(Y, Ytarget)


                        # has input been retained/discarded after forward pass?
                        if keep_input:
                            self.assertTrue(X.data.shape == Xshape)
                        else:
                            self.assertTrue(len(X.data.shape) == 0 or (len(X.data.shape) == 1 and X.data.shape[0] == 0))

                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        self.assertTrue(Y.shape == Xshape)
                        self.assertTrue(X.data.numpy().shape == data.shape)
                        self.assertTrue(np.allclose(X.data.numpy(), data, atol=1e-06))
                        self.assertTrue(np.allclose(X.data.numpy(), Xinv.data.numpy()))
                        impl_out.append(Y.data.numpy().copy())
                        impl_grad.append([p.data.numpy().copy() for p in Gm2.parameters()])
                        self.assertFalse(np.allclose(impl_grad[-1][0], s_grad[0]))

                # output and gradients per implementation similar ?
                self.assertTrue(np.allclose(impl_out[0], impl_out[1]))
                for i in range(0, len(implementations_bwd) * len(implementations_fwd) - 1, 1):
                        self.assertTrue(np.allclose(impl_grad[i][0], impl_grad[i + 1][0]))

    def test_reversible_block_inv(self):
        """ReversibleBlock test using inverse training

        * test inversion X = RB.inverse(Y) and Y = RB(X)
        * test training the block for a single step and compare weights for implementations 0, 1
        * test automatic discard of input Y and its retrieval after the backward pass of inverse
        * test usage of BN to identify non-contiguous memory blocks

        """
        dims = (2, 10, 8, 8)
        data = np.random.random(dims).astype(np.float32)
        target_data = np.random.random(dims).astype(np.float32)
        impl_out, impl_grad = [], []

        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(10 // 2)
                self.conv = torch.nn.Conv2d(10 // 2, 10 // 2, (3, 3), padding=1)

            def forward(self, x):
                return self.bn(self.conv(x))

        Gm = SubModule()

        s_grad = [p.data.numpy().copy() for p in Gm.parameters()]
        implementations_fwd = [0, 0, 1, 1]
        implementations_bwd = [0, 0, 1, 1]
        for coupling in ['additive', 'affine']:
            for keep_input in [False, True]:
                for implementation_fwd in implementations_fwd:
                    for implementation_bwd in implementations_bwd:
                        # same convolution test
                        Y = Variable(torch.from_numpy(data.copy()))
                        Xtarget = Variable(torch.from_numpy(target_data.copy()))
                        Yshape = Y.shape
                        Gm2 = copy.deepcopy(Gm)
                        rb = revop.ReversibleBlock(Gm2, coupling=coupling,
                                                   keep_input=keep_input,
                                                   implementation_fwd=implementation_fwd,
                                                   implementation_bwd=implementation_bwd)
                        rb.train()
                        rb.zero_grad()

                        optim = torch.optim.RMSprop(rb.parameters())
                        optim.zero_grad()

                        X = rb.inverse(Y)
                        Yinv = rb(X.clone())
                        loss = torch.nn.MSELoss()(X, Xtarget)

                        # has input been retained/discarded after forward pass?
                        if keep_input:
                            self.assertTrue(Y.data.shape == Yshape)
                        else:
                            self.assertTrue(len(Y.data.shape) == 0 or (len(Y.data.shape) == 1 and Y.data.shape[0] == 0))

                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                        self.assertTrue(X.shape == Yshape)
                        self.assertTrue(Y.data.numpy().shape == data.shape)
                        self.assertTrue(np.allclose(Y.data.numpy(), data, atol=1e-06))
                        self.assertTrue(np.allclose(Y.data.numpy(), Yinv.data.numpy()))
                        impl_out.append(X.data.numpy().copy())
                        impl_grad.append([p.data.numpy().copy() for p in Gm2.parameters()])
                        self.assertFalse(np.allclose(impl_grad[-1][0], s_grad[0]))

                # output and gradients per implementation similar ?
                self.assertTrue(np.allclose(impl_out[0], impl_out[1]))
                for i in range(0, len(implementations_bwd) * len(implementations_fwd) - 1, 1):
                        self.assertTrue(np.allclose(impl_grad[i][0], impl_grad[i + 1][0]))


    def test_normal_vs_revblock(self):
        """ReversibleBlock test if similar gradients and weights results are obtained after similar training

        * test training the block for a single step and compare weights and grads for implementations 0, 1 and 2
        * test against normal non Reversible Block function
        * test if recreated input and produced output are contiguous

        """
        for implementation_fwd in [0, 1]:
            for implementation_bwd in [0, 1]:
                X = Variable(torch.rand(2, 4, 5, 5))

                # define models and their copies
                c1 = torch.nn.Conv2d(2, 2, 3, padding=1)
                c2 = torch.nn.Conv2d(2, 2, 3, padding=1)
                c1_2 = copy.deepcopy(c1)
                c2_2 = copy.deepcopy(c2)

                # are weights between models the same, but do they differ between convolutions?
                self.assertTrue(torch.equal(c1.weight, c1_2.weight))
                self.assertTrue(torch.equal(c2.weight, c2_2.weight))
                self.assertTrue(torch.equal(c1.bias, c1_2.bias))
                self.assertTrue(torch.equal(c2.bias, c2_2.bias))
                self.assertFalse(torch.equal(c1.weight, c2.weight))

                # define optimizers
                optim1 = torch.optim.SGD([e for e in c1.parameters()] + [e for e in c2.parameters()], 0.1)
                optim2 = torch.optim.SGD([e for e in c1_2.parameters()] + [e for e in c2_2.parameters()], 0.1)
                for e in [c1, c2, c1_2, c2_2]:
                    e.train()

                # define an arbitrary reversible function and define graph for model 1
                Xin = X.clone()
                # TODO: add normal test for affine coupling
                fn = revop.ReversibleBlock(c1_2, c2_2, coupling='additive',
                                           keep_input=False,
                                           implementation_fwd=implementation_fwd,
                                           implementation_bwd=implementation_bwd)

                Y = fn.forward(Xin)
                loss2 = torch.mean(Y)

                # define the reversible function without custom backprop and define graph for model 2
                XX = Variable(X.clone().data, requires_grad=True)
                x1, x2 = torch.chunk(XX, 2, dim=1)
                y1 = x1 + c1.forward(x2)
                y2 = x2 + c2.forward(y1)
                YY = torch.cat([y1, y2], dim=1)
                loss = torch.mean(YY)

                # compute gradients manually
                grads = torch.autograd.grad(loss, (XX, c1.weight, c2.weight, c1.bias, c2.bias), None, retain_graph=True)

                # compute gradients and perform optimization model 2
                loss.backward()
                optim1.step()

                # gradients computed manually match those of the .backward() pass
                self.assertTrue(torch.equal(c1.weight.grad, grads[1]))
                self.assertTrue(torch.equal(c2.weight.grad, grads[2]))
                self.assertTrue(torch.equal(c1.bias.grad, grads[3]))
                self.assertTrue(torch.equal(c2.bias.grad, grads[4]))

                # weights differ after training a single model?
                self.assertFalse(torch.equal(c1.weight, c1_2.weight))
                self.assertFalse(torch.equal(c2.weight, c2_2.weight))
                self.assertFalse(torch.equal(c1.bias, c1_2.bias))
                self.assertFalse(torch.equal(c2.bias, c2_2.bias))

                # compute gradients and perform optimization model 1
                loss2.backward()
                optim2.step()

                # input is contiguous tests
                self.assertTrue(Xin.is_contiguous())
                self.assertTrue(Y.is_contiguous())

                # weights are approximately the same after training both models?
                self.assertTrue(np.allclose(c1.weight.data.numpy(), c1_2.weight.data.numpy()))
                self.assertTrue(np.allclose(c2.weight.data.numpy(), c2_2.weight.data.numpy()))
                self.assertTrue(np.allclose(c1.bias.data.numpy(), c1_2.bias.data.numpy()))
                self.assertTrue(np.allclose(c2.bias.data.numpy(), c2_2.bias.data.numpy()))

                # gradients are approximately the same after training both models?
                self.assertTrue(np.allclose(c1.weight.grad.data.numpy(), c1_2.weight.grad.data.numpy()))
                self.assertTrue(np.allclose(c2.weight.grad.data.numpy(), c2_2.weight.grad.data.numpy()))
                self.assertTrue(np.allclose(c1.bias.grad.data.numpy(), c1_2.bias.grad.data.numpy()))
                self.assertTrue(np.allclose(c2.bias.grad.data.numpy(), c2_2.bias.grad.data.numpy()))


    @unittest.skipIf(not torch.cuda.is_available(), reason='This test requires a GPU to be available')
    def test_memory_saving(self):
        """Test memory saving of the reversible block

        * tests fitting a large number of images by creating a very deep network requiring big
          intermediate feature maps for training

        * input size in bytes: np.prod((2, 10, 2000, 2000)) * 4 / (1024 ** 2)
                                                      (approx.) = 305 MB

        * tuned on a Titan X with 12 GB of RAM (depth=25 will just fit, but depth=250 will clearly not fit)
          This will approximately require:
            depth=25:  7629 MB
            depth=250: 76293 MB

        NOTE: This test assumes it is ran on a machine with a GPU with less than +/- 76293 MB
        NOTE: This test can be quite slow to execute

        """
        dims = (2, 10, 2000, 2000)
        data = np.random.random(dims).astype(np.float32)
        target_data = np.random.random(dims).astype(np.float32)

        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(10 // 2)
                self.conv = torch.nn.Conv2d(10 // 2, 10 // 2, (3, 3), padding=1)

            def forward(self, x):
                return self.bn(self.conv(x))

        class SubModuleStack(torch.nn.Module):
            def __init__(self, Gm, coupling, depth=10, implementation_fwd=1, implementation_bwd=1, keep_input=False):
                super(SubModuleStack, self).__init__()
                self.stack = torch.nn.Sequential(
                    *[revop.ReversibleBlock(Gm, Gm, coupling, keep_input=keep_input, implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd) for _ in range(depth)]
                )

            def forward(self, x):
                return self.stack(x)

        for coupling in ['additive', 'affine']:
            for keep_input in [False, True]:
                for implementation_fwd in [1]:
                    for implementation_bwd in [1]:
                        # same convolution test
                        X = Variable(torch.from_numpy(data.copy())).cuda()
                        Ytarget = Variable(torch.from_numpy(target_data.copy())).cuda()
                        network = SubModuleStack(SubModule(), coupling, depth=250, keep_input=keep_input,
                                                 implementation_fwd=implementation_fwd,
                                                 implementation_bwd=implementation_bwd)
                        network.cuda()
                        network.train()
                        network.zero_grad()
                        optim = torch.optim.RMSprop(network.parameters())
                        optim.zero_grad()
                        try:
                            Y = network(X)
                            loss = torch.nn.MSELoss()(Y, Ytarget)
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                            # Should not be reached when input is kept
                            self.assertFalse(keep_input)
                        except RuntimeError:
                            # Running out of memory should only happen when input is kept
                            self.assertTrue(keep_input)
                        finally:
                            del network
                            del optim
                            del X
                            del Ytarget


if __name__ == '__main__':
    unittest.main()
