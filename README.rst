======
MemCNN
======

.. image:: https://img.shields.io/circleci/build/github/silvandeleemput/memcnn/master.svg        
        :alt: CircleCI
        :target: https://circleci.com/gh/silvandeleemput/memcnn/tree/master

.. image:: https://readthedocs.org/projects/memcnn/badge/?version=latest        
        :alt: Documentation Status
        :target: https://memcnn.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/codecov/c/gh/silvandeleemput/memcnn/master.svg   
        :alt: Codecov branch
        :target: https://codecov.io/gh/silvandeleemput/memcnn

.. image:: https://img.shields.io/pypi/v/memcnn.svg
        :target: https://pypi.python.org/pypi/memcnn

.. image:: https://img.shields.io/pypi/implementation/memcnn.svg        
        :alt: PyPI - Implementation
        :target: https://pypi.python.org/pypi/memcnn

.. image:: https://img.shields.io/pypi/pyversions/memcnn.svg        
        :alt: PyPI - Python Version
        :target: https://pypi.python.org/pypi/memcnn

.. image:: https://img.shields.io/github/license/silvandeleemput/memcnn.svg        
        :alt: GitHub
        :target: https://memcnn.readthedocs.io/en/latest/?badge=latest

A `PyTorch <http://pytorch.org/>`__ framework for developing memory
efficient deep invertible networks

* Free software: MIT license
* Documentation: https://memcnn.readthedocs.io.
* Installation: https://memcnn.readthedocs.io/en/latest/installation.html


Reference: Sil C. van de Leemput, Jonas Teuwen, Rashindra Manniesing.
`MemCNN: a Framework for Developing Memory Efficient Deep Invertible
Networks <https://openreview.net/forum?id=r1KzqK1wz>`__. *International
Conference on Learning Representations (ICLR) 2018 Workshop Track.
(https://iclr.cc/)*

Licencing
---------

This repository comes with the MIT license, which implies everyone has
the right to use, copy, distribute and/or modify this work. If you do,
please cite our work.

Features
--------

* Simple `ReversibleBlock` wrapper class to wrap and convert arbitrary PyTorch Modules to invertible versions.
* Simple switching between `additive` and `affine` invertible coupling schemes and different implementations.
* Simple toggling of memory saving by setting the `keep_input` property of the `ReversibleBlock`.
* Train and evaluation code for reproducing RevNet experiments using MemCNN.
* CI tests for Python v2.7 and v3.6 and torch v0.4, v1.0, and v1.1.

Example usage: ReversibleBlock
------------------------------

.. code:: python

    # some required imports
    import torch
    import torch.nn as nn
    import numpy as np
    import memcnn.models.revop


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
    Y = memcnn.models.revop.ReversibleBlock(F, G, coupling='additive')(X)

Run PyTorch Experiments
-----------------------

.. code:: bash

    ./train.py [MODEL] [DATASET] --fresh

Available values for ``DATASET`` are ``cifar10`` and ``cifar100``.

Available values for ``MODEL`` are ``resnet32``, ``resnet110``,
``resnet164``, ``revnet38``, ``revnet110``, ``revnet164``

If not available datasets are automatically downloaded.

Results
-------

TensorFlow results were obtained from `the reversible residual
network <https://arxiv.org/abs/1707.04585>`__ running the code from
their `GitHub <https://github.com/renmengye/revnet-public>`__.

.. raw:: html

        <table>
        <tr><th>            </th><th colspan="4"> TensorFlow        </th><th colspan="4"> PyTorch     </th></tr>
        <tr><th>            </th><th colspan="2"> Cifar-10        </th><th th colspan="2"> Cifar-100        </th><th th colspan="2"> Cifar-10       </th><th th colspan="2"> Cifar-100          </th></tr>
        <tr><th> Model      </th><th> acc.      </th><th> time  </th><th> acc.      </th><th> time   </th><th> acc.      </th><th> time    </th><th> acc.      </th><th> time    </th></tr>
        <tr><td> resnet-32  </td><td> 92.74     </td><td> 2:04  </td><td> 69.10     </td><td> 1:58   </td><td> 92.86     </td><td> 1:51    </td><td> 69.81     </td><td> 1:51    </td></tr>
        <tr><td> resnet-110 </td><td> 93.99     </td><td> 4:11  </td><td> 73.30     </td><td> 6:44   </td><td> 93.55     </td><td> 2:51    </td><td> 72.40     </td><td> 2:39    </td></tr>
        <tr><td> resnet-164 </td><td> 94.57     </td><td> 11:05 </td><td> 76.79     </td><td> 10:59  </td><td> 94.80     </td><td> 4:59    </td><td> 76.47     </td><td> 3:45    </td></tr>
        <tr><td> revnet-38  </td><td> 93.14     </td><td> 2:17  </td><td> 71.17     </td><td> 2:20   </td><td> 92.8     </td><td> 2:09    </td><td> 69.9     </td><td> 2:16    </td></tr>
        <tr><td> revnet-110 </td><td> 94.02     </td><td> 6:59  </td><td> 74.00     </td><td> 7:03   </td><td> 94.1     </td><td> 3:42    </td><td> 73.3     </td><td> 3:50    </td></tr>
        <tr><td> revnet-164 </td><td> 94.56     </td><td> 13:09 </td><td> 76.39     </td><td> 13:12  </td><td> 94.9     </td><td> 7:21    </td><td> 76.9     </td><td> 7:17    </td></tr>
        </table>

The PyTorch results listed were recomputed on June 11th 2018, and differ
from the results in the paper. The Tensorflow results are still the
same.

Memory consumption of model training in PyTorch
-----------------------------------------------

.. raw:: html

        <table>
        <tr><th> Model      </th><th> GPU VRAM (MB) </th></tr>
        <tr><td> resnet-32  </td><td> 766     </td></tr>
        <tr><td> resnet-110 </td><td> 1357     </td></tr>
        <tr><td> resnet-164 </td><td> 3083     </td></tr>
        <tr><td> revnet-38  </td><td> 677     </td></tr>
        <tr><td> revnet-110 </td><td> 706     </td></tr>
        <tr><td> revnet-164 </td><td> 1226     </td></tr>
        </table>

Works using MemCNN
------------------

* `MemCNN: a Framework for Developing Memory Efficient Deep Invertible Networks <https://openreview.net/forum?id=r1KzqK1wz>`__ by Sil C. van de Leemput
* `Reversible GANs for Memory-efficient Image-to-Image Translation <https://arxiv.org/abs/1902.02729>`__ by Tycho van der Ouderaa

Citation
--------

If you use our code, please cite:

.. code:: bibtex

    @inproceedings{
      leemput2018memcnn,
      title={MemCNN: a Framework for Developing Memory Efficient Deep Invertible Networks},
      author={Sil C. van de Leemput, Jonas Teuwen, Rashindra Manniesing},
      booktitle={ICLR 2018 Workshop Track},
      year={2018},
      url={https://openreview.net/forum?id=r1KzqK1wz},
    }
    
