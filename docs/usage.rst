=====
Usage
=====

To use MemCNN in a project::

    import memcnn


Example usage: ReversibleBlock
------------------------------

.. code:: python

    # some required imports
    import torch
    import torch.nn as nn
    import numpy as np
    import memcnn


    # define a new class of operation(s) PyTorch style
    class ExampleOperation(nn.Module):
        def __init__(self, channels):
            super(ExampleOperation, self).__init__()
            self.seq = nn.Sequential(
                                        nn.Conv2d(in_channels=channels, out_channels=channels,
                                                  kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(num_features=channels),
                                        nn.ReLU(inplace=True)
                                    )

        def forward(self, x):
            return self.seq(x)


    # generate some random input data (b, c, y, x)
    data = np.random.random((2, 10, 8, 8)).astype(np.float32)
    X = torch.from_numpy(data)

    # application of the operation(s) the normal way
    Y = ExampleOperation(channels=10)(X)

    # application of the operation(s) using the reversible block
    F, G = ExampleOperation(channels=10 // 2), ExampleOperation(channels=10 // 2)
    Y = memcnn.ReversibleBlock(F, G, coupling='additive')(X)


Run PyTorch Experiments
-----------------------

.. include:: ./usage_experiments.rst
