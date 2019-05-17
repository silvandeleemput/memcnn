import unittest
import torch
import torch.nn
from torch.autograd import Variable
import memcnn.models.revop as revop
import numpy as np
import copy


class ReversibleOperationsTestCase(unittest.TestCase):
    def test_reversible_block_fwd_bwd(self):
        """ReversibleBlock test of the memory saving forward and backward passes

        * test inversion Y = RB(X) and X = RB.inverse(Y)
        * test training the block for a single step and compare weights for implementations: 0, 1
        * test automatic discard of input X and its retrieval after the backward pass
        * test usage of BN to identify non-contiguous memory blocks

        """
        dims = (2, 10, 8, 8)
        data = np.random.random(dims).astype(np.float32)
        target_data = np.random.random(dims).astype(np.float32)

        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(10 // 2)
                self.conv = torch.nn.Conv2d(10 // 2, 10 // 2, (3, 3), padding=1)

            def forward(self, x):
                return self.bn(self.conv(x))

        Gm = SubModule()

        s_grad = [p.data.numpy().copy() for p in Gm.parameters()]
        for _ in range(10):
            for bwd in [False, True]:
                for coupling in ['additive']:  # , 'affine']:
                    impl_out, impl_grad = [], []
                    for keep_input in [False, True]:
                        for implementation_fwd in [-1, -1, 0, 0, 1, 1]:
                            for implementation_bwd in [-1, 0, 1]:
                                keep_input = keep_input or (implementation_bwd == -1)
                                # print(bwd, coupling, keep_input, implementation_fwd, implementation_bwd)
                                # test with zero padded convolution
                                X = Variable(torch.from_numpy(data.copy())).clone()
                                Ytarget = Variable(torch.from_numpy(target_data.copy())).clone()
                                Xshape = X.shape
                                Gm2 = copy.deepcopy(Gm)
                                rb = revop.ReversibleBlock(Gm2, coupling=coupling, implementation_fwd=implementation_fwd,
                                                           implementation_bwd=implementation_bwd, keep_input=keep_input)
                                rb.train()
                                rb.zero_grad()

                                optim = torch.optim.RMSprop(rb.parameters())
                                optim.zero_grad()

                                if not bwd:
                                    Y = rb(X)
                                    Yrev = Y.clone()
                                    Xinv = rb.inverse(Yrev)
                                else:
                                    Y = rb.inverse(X)
                                    Yrev = Y.clone()
                                    Xinv = rb(Yrev)
                                loss = torch.nn.MSELoss()(Y, Ytarget)

                                # has input been retained/discarded after forward (and backward) passes?
                                if keep_input:
                                    self.assertTrue(X.data.shape == Xshape)
                                    self.assertTrue(Y.data.shape == Yrev.shape)
                                else:
                                    self.assertTrue(len(X.data.shape) == 0 or (len(X.data.shape) == 1 and X.data.shape[0] == 0))
                                    self.assertTrue(len(Yrev.data.shape) == 0 or (len(Yrev.data.shape) == 1
                                                                                  and Yrev.data.shape[0] == 0))

                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                                self.assertTrue(Y.shape == Xshape)
                                self.assertTrue(X.data.numpy().shape == data.shape)
                                self.assertTrue(np.allclose(X.data.numpy(), data, atol=1e-06))
                                self.assertTrue(np.allclose(X.data.numpy(), Xinv.data.numpy(), atol=1e-06))
                                impl_out.append(Y.data.numpy().copy())
                                impl_grad.append([p.data.numpy().copy() for p in Gm2.parameters()])
                                self.assertFalse(np.allclose(impl_grad[-1][0], s_grad[0]))

                        # output and gradients similar over all implementations?
                        for i in range(0, len(impl_grad) - 1, 1):
                            self.assertTrue(np.allclose(impl_grad[i][0], impl_grad[i + 1][0]))
                            self.assertTrue(np.allclose(impl_out[i], impl_out[i + 1]))

    def test_revblock_simple_inverse(self):
        """ReversibleBlock inverse test

        * test inversion Y = RB(X) and X = RB.inverse(Y)

        """
        for _ in range(10):
            for coupling in ['additive']:  # , 'affine']:
                for implementation_fwd in [-1, 0, 1]:
                    for implementation_bwd in [-1, 0, 1]:
                        # define some data
                        X = Variable(torch.rand(2, 4, 5, 5))

                        # define an arbitrary reversible function
                        fn = revop.ReversibleBlock(torch.nn.Conv2d(2, 2, 3, padding=1), keep_input=False, coupling=coupling,
                                                   implementation_fwd=implementation_fwd,
                                                   implementation_bwd=implementation_bwd)

                        # compute output
                        Y = fn.forward(X.clone())

                        # compute input from output
                        X2 = fn.inverse(Y)

                        # check that the inverted output and the original input are approximately similar
                        self.assertTrue(np.allclose(X2.data.numpy(), X.data.numpy(), atol=1e-06))

    def test_normal_vs_revblock(self):
        """ReversibleBlock test if similar gradients and weights results are obtained after similar training

        * test training the block for a single step and compare weights and grads for implementations: 0, 1
        * test against normal non Reversible Block function
        * test if recreated input and produced output are contiguous

        """
        for _ in range(10):
            for coupling in ['additive']: #, 'affine']:
                for implementation_fwd in [-1, 0, 1]:
                    for implementation_bwd in [-1, 0, 1]:
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
                        fn = revop.ReversibleBlock(c1_2, c2_2, keep_input=False, coupling=coupling,
                                                   implementation_fwd=implementation_fwd,
                                                   implementation_bwd=implementation_bwd)
                        Y = fn.forward(Xin)
                        loss2 = torch.mean(Y)

                        # define the reversible function without custom backprop and define graph for model 2
                        XX = Variable(X.clone().data, requires_grad=True)
                        x1, x2 = torch.chunk(XX, 2, dim=1)
                        if coupling == 'additive':
                            y1 = x1 + c1.forward(x2)
                            y2 = x2 + c2.forward(y1)
                        elif coupling == 'affine':
                            fmr1, fmr2 = c1.forward(x2)
                            y1 = (x1 * fmr1) + fmr2
                            gmr1, gmr2 = c2.forward(y1)
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
                        self.assertTrue(np.allclose(c1.weight.data.numpy(), c1_2.weight.data.numpy(), atol=1e-06))
                        self.assertTrue(np.allclose(c2.weight.data.numpy(), c2_2.weight.data.numpy()))
                        self.assertTrue(np.allclose(c1.bias.data.numpy(), c1_2.bias.data.numpy()))
                        self.assertTrue(np.allclose(c2.bias.data.numpy(), c2_2.bias.data.numpy()))

                        # gradients are approximately the same after training both models?
                        self.assertTrue(np.allclose(c1.weight.grad.data.numpy(), c1_2.weight.grad.data.numpy(), atol=1e-06))
                        self.assertTrue(np.allclose(c2.weight.grad.data.numpy(), c2_2.weight.grad.data.numpy()))
                        self.assertTrue(np.allclose(c1.bias.grad.data.numpy(), c1_2.bias.grad.data.numpy()))
                        self.assertTrue(np.allclose(c2.bias.grad.data.numpy(), c2_2.bias.grad.data.numpy()))


if __name__ == '__main__':
    unittest.main()
