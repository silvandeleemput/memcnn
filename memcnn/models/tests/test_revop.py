import unittest
import torch
import torch.nn
from torch.autograd import Variable
import memcnn.models.revop as revop
import numpy as np

class ReversibleOperationsTestCase(unittest.TestCase):

    def test_reversible_block(self):
        """test ReversibleBlock inversions and reproducibility of input after backward pass"""
        for implementation in [0, 1]:
            # same convolution test
            Gm = torch.nn.Conv2d(10 // 2, 10 // 2, (3,3), padding=1)
            dims = (2,10,8,8)

            Xdata = np.random.random(dims).astype(np.float32)

            X = Variable(torch.from_numpy(Xdata))
            Xshape = X.shape
            rb = revop.ReversibleBlock(Gm, implementation=implementation)
            Y = rb(X)
            X.data.set_()
            self.assertTrue(len(X.data.shape) == 0)
            Y.backward(torch.ones_like(Y))

            self.assertTrue(Y.shape == Xshape)
            self.assertTrue(X.data.numpy().shape == Xdata.shape)
            self.assertTrue(np.isclose(X.data.numpy(), Xdata, atol=1e-06).all())

if __name__ == '__main__':
    unittest.main()