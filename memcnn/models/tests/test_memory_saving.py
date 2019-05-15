import unittest
import torch
import torch.nn
import memcnn.models.revop as revop
import numpy as np


class ReversibleOperationsTestCase(unittest.TestCase):

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
            def __init__(self, Gm, coupling='additive', depth=10, implementation_fwd=1, implementation_bwd=1, keep_input=False):
                super(SubModuleStack, self).__init__()
                self.stack = torch.nn.Sequential(
                    *[revop.ReversibleBlock(Gm, Gm, coupling=coupling, implementation_fwd=implementation_fwd,
                                            implementation_bwd=implementation_bwd,
                                            keep_input=keep_input) for _ in range(depth)]
                )

            def forward(self, x):
                return self.stack(x)

        for coupling in ['additive']:
            for keep_input in [False, True]:
                for implementation_fwd in [1]:
                    for implementation_bwd in [1]:
                        # same convolution test
                        X = torch.from_numpy(data.copy()).cuda()
                        Ytarget = torch.from_numpy(target_data.copy()).cuda()
                        network = SubModuleStack(SubModule(), depth=250, keep_input=keep_input, coupling=coupling,
                                                 implementation_fwd=implementation_fwd, implementation_bwd=implementation_bwd)
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
                            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
