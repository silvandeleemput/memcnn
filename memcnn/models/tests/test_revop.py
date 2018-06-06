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
        * test training the block for a single step and compare weights for implementations 1 & 2
        * test automatic discard of input X and its retrieval after the backward pass

        """
        dims = (2, 10, 8, 8)
        data = np.random.random(dims).astype(np.float32)
        target_data = np.random.random(dims).astype(np.float32)
        impl_out, impl_grad = [], []
        Gm = torch.nn.Conv2d(10 // 2, 10 // 2, (3, 3), padding=1)
        s_grad = [p.data.numpy().copy() for p in Gm.parameters()]
        implementations = [0, 0, 1, 1]
        for implementation in implementations:
            # same convolution test
            X = Variable(torch.from_numpy(data.copy()))
            Ytarget = Variable(torch.from_numpy(target_data.copy()))
            Xshape = X.shape
            Gm2 = copy.deepcopy(Gm)
            rb = revop.ReversibleBlock(Gm2, implementation=implementation, keep_input=False)
            rb.train()
            rb.zero_grad()

            optim = torch.optim.RMSprop(rb.parameters())
            optim.zero_grad()

            Y = rb(X)
            Xinv = rb.inverse(Y)
            loss = torch.nn.MSELoss()(Y, Ytarget)

            self.assertTrue(len(X.data.shape) == 0)

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
        for i in range(0, len(implementations) - 1, 1):
            self.assertTrue(np.allclose(impl_grad[i][0], impl_grad[i + 1][0]))


    def test_revblock_inverse(self):
        """ReversibleBlock inverse test

        * test inversion Y = RB(X) and X = RB.inverse(Y)

        """
        for implementation in range(2):
            # define some data
            X = Variable(torch.rand(2, 4, 5, 5))

            # define an arbitrary reversible function
            fn = revop.ReversibleBlock(torch.nn.Conv2d(2, 2, 3, padding=1), keep_input=False,
                                       implementation=implementation)

            # compute output
            Y = fn.forward(X.clone())

            # compute input from output
            X2 = fn.inverse(Y)

            # check that the inverted output and the original input are approximately similar
            self.assertTrue(np.allclose(X2.data.numpy(), X.data.numpy(), atol=1e-7))


    def test_normal_vs_revblock(self):
        """ReversibleBlock test if similar gradients and weights results are obtained for

        * test training the block for a single step and compare weights and grads for implementations 1 & 2
        * test against normal non Reversible Block function

        """
        for implementation in range(2):
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
            fn = revop.ReversibleBlock(c1_2, c2_2, keep_input=False, implementation=implementation)
            Y = fn.forward(X.clone())
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


if __name__ == '__main__':
    unittest.main()
