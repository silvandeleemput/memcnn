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
        * test training the block for a single step and compare weights within implementations 1 & 2
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
        for i in range(0, len(implementations) - 1, 2):
            self.assertTrue(np.allclose(impl_grad[i][0], impl_grad[i + 1][0]))

    def test_revblock_inverse(self):
        # define some data
        X = Variable(torch.rand(2, 4, 5, 5))

        # define an arbitrary reversible function
        fn = revop.ReversibleBlock(torch.nn.Conv2d(2, 2, 3, padding=1), keep_input=False)

        # compute output
        Y = fn.forward(X.clone())

        # compute input from output
        X2 = fn.inverse(Y)

        # check that the inverted output and the original input are approximately similar
        self.assertTrue(np.allclose(X2.data.numpy(), X.data.numpy()))


if __name__ == '__main__':
    unittest.main()
