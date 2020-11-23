======
MemCNN
======

.. image:: https://img.shields.io/circleci/build/github/silvandeleemput/memcnn/master.svg        
        :alt: CircleCI - Status master branch
        :target: https://circleci.com/gh/silvandeleemput/memcnn/tree/master

.. image:: https://img.shields.io/docker/cloud/build/silvandeleemput/memcnn.svg
        :alt: Docker - Status
        :target: https://hub.docker.com/r/silvandeleemput/memcnn

.. image:: https://readthedocs.org/projects/memcnn/badge/?version=latest        
        :alt: Documentation - Status master branch
        :target: https://memcnn.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/codacy/grade/95de32e0d7c54d038611da47e9f0948b/master.svg
        :alt: Codacy - Branch grade
        :target: https://app.codacy.com/project/silvandeleemput/memcnn/dashboardgit

.. image:: https://img.shields.io/codecov/c/gh/silvandeleemput/memcnn/master.svg   
        :alt: Codecov - Status master branch
        :target: https://codecov.io/gh/silvandeleemput/memcnn

.. image:: https://img.shields.io/pypi/v/memcnn.svg
        :alt: PyPI - Latest release
        :target: https://pypi.python.org/pypi/memcnn

.. image:: https://img.shields.io/conda/vn/silvandeleemput/memcnn?label=anaconda
        :alt: Conda - Latest release
        :target: https://anaconda.org/silvandeleemput/memcnn

.. image:: https://img.shields.io/pypi/implementation/memcnn.svg        
        :alt: PyPI - Implementation
        :target: https://pypi.python.org/pypi/memcnn

.. image:: https://img.shields.io/pypi/pyversions/memcnn.svg        
        :alt: PyPI - Python version
        :target: https://pypi.python.org/pypi/memcnn

.. image:: https://img.shields.io/github/license/silvandeleemput/memcnn.svg        
        :alt: GitHub - Repository license
        :target: https://github.com/silvandeleemput/memcnn/blob/master/LICENSE.txt

.. image:: http://joss.theoj.org/papers/10.21105/joss.01576/status.svg
        :alt: JOSS - DOI
        :target: https://doi.org/10.21105/joss.01576

A `PyTorch <http://pytorch.org/>`__ framework for developing memory-efficient invertible neural networks.

* Free software: `MIT license <https://github.com/silvandeleemput/memcnn/blob/master/LICENSE.txt>`__ (please cite our work if you use it)
* Documentation: https://memcnn.readthedocs.io.
* Installation: https://memcnn.readthedocs.io/en/latest/installation.html

Features
--------

* Enable memory savings during training by wrapping arbitrary invertible PyTorch functions with the `InvertibleModuleWrapper` class.
* Simple toggling of memory saving by setting the `keep_input` property of the `InvertibleModuleWrapper`.
* Turn arbitrary non-linear PyTorch functions into invertible versions using the `AdditiveCoupling` or the `AffineCoupling` classes.
* Training and evaluation code for reproducing RevNet experiments using MemCNN.
* CI tests for Python v3.7 and torch v1.0, v1.1, v1.4 and v1.7 with good code coverage.

Examples
--------

Creating an AdditiveCoupling with memory savings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import torch
    import torch.nn as nn
    import memcnn


    # define a new torch Module with a sequence of operations: Relu o BatchNorm2d o Conv2d
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


    # generate some random input data (batch_size, num_channels, y_elements, x_elements)
    X = torch.rand(2, 10, 8, 8)

    # application of the operation(s) the normal way
    model_normal = ExampleOperation(channels=10)
    model_normal.eval()

    Y = model_normal(X)

    # turn the ExampleOperation invertible using an additive coupling
    invertible_module = memcnn.AdditiveCoupling(
        Fm=ExampleOperation(channels=10 // 2),
        Gm=ExampleOperation(channels=10 // 2)
    )

    # test that it is actually a valid invertible module (has a valid inverse method)
    assert memcnn.is_invertible_module(invertible_module, test_input_shape=X.shape)

    # wrap our invertible_module using the InvertibleModuleWrapper and benefit from memory savings during training
    invertible_module_wrapper = memcnn.InvertibleModuleWrapper(fn=invertible_module, keep_input=True, keep_input_inverse=True)

    # by default the module is set to training, the following sets this to evaluation
    # note that this is required to pass input tensors to the model with requires_grad=False (inference only)
    invertible_module_wrapper.eval()

    # test that the wrapped module is also a valid invertible module
    assert memcnn.is_invertible_module(invertible_module_wrapper, test_input_shape=X.shape)

    # compute the forward pass using the wrapper
    Y2 = invertible_module_wrapper.forward(X)

    # the input (X) can be approximated (X2) by applying the inverse method of the wrapper on Y2
    X2 = invertible_module_wrapper.inverse(Y2)

    # test that the input and approximation are similar
    assert torch.allclose(X, X2, atol=1e-06)

Run PyTorch Experiments
-----------------------

After installing MemCNN run:

.. code:: bash

    python -m memcnn.train [MODEL] [DATASET] [--fresh] [--no-cuda]

* Available values for ``DATASET`` are ``cifar10`` and ``cifar100``.
* Available values for ``MODEL`` are ``resnet32``, ``resnet110``, ``resnet164``, ``revnet38``, ``revnet110``, ``revnet164``
* Use the ``--fresh`` flag to remove earlier experiment results.
* Use the ``--no-cuda`` flag to train on the CPU rather than the GPU through CUDA.

Datasets are automatically downloaded if they are not available.

When using Python 3.* replace the ``python`` directive with the appropriate Python 3 directive. For example when using the MemCNN docker image use ``python3.6``.

When MemCNN was installed using `pip` or from sources you might need to setup a configuration file before running this command.
Read the corresponding section about how to do this here: https://memcnn.readthedocs.io/en/latest/installation.html

Results
-------

TensorFlow results were obtained from `the reversible residual
network <https://arxiv.org/abs/1707.04585>`__ running the code from
their `GitHub <https://github.com/renmengye/revnet-public>`__.

The PyTorch results listed were recomputed on June 11th 2018, and differ
from the results in the ICLR paper. The Tensorflow results are still the
same.

Prediction accuracy
^^^^^^^^^^^^^^^^^^^

+------------+------------------------+--------------------------+----------------------+----------------------+
|            |               Cifar-10                            |               Cifar-100                     |
+------------+------------------------+--------------------------+----------------------+----------------------+
| Model      |    Tensorflow          |      PyTorch             |      Tensorflow      |     PyTorch          |
+============+========================+==========================+======================+======================+
| resnet-32  |  92.74                 |    92.86                 |   69.10              |  69.81               |
+------------+------------------------+--------------------------+----------------------+----------------------+
| resnet-110 |  93.99                 |    93.55                 |   73.30              |  72.40               |
+------------+------------------------+--------------------------+----------------------+----------------------+
| resnet-164 |  94.57                 |    94.80                 |   76.79              |  76.47               |
+------------+------------------------+--------------------------+----------------------+----------------------+
| revnet-38  |  93.14                 |    92.80                 |   71.17              |  69.90               |
+------------+------------------------+--------------------------+----------------------+----------------------+
| revnet-110 |  94.02                 |    94.10                 |   74.00              |  73.30               |
+------------+------------------------+--------------------------+----------------------+----------------------+
| revnet-164 |  94.56                 |    94.90                 |   76.39              |  76.90               |
+------------+------------------------+--------------------------+----------------------+----------------------+

Training time (hours : minutes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------+------------------------+--------------------------+----------------------+----------------------+
|            |               Cifar-10                            |               Cifar-100                     |
+------------+------------------------+--------------------------+----------------------+----------------------+
| Model      |    Tensorflow          |      PyTorch             |      Tensorflow      |     PyTorch          |
+============+========================+==========================+======================+======================+
| resnet-32  |             2:04       |    1:51                  |       1:58           |              1:51    |
+------------+------------------------+--------------------------+----------------------+----------------------+
| resnet-110 |             4:11       |    2:51                  |       6:44           |              2:39    |
+------------+------------------------+--------------------------+----------------------+----------------------+
| resnet-164 |            11:05       |    4:59                  |   10:59              |              3:45    |
+------------+------------------------+--------------------------+----------------------+----------------------+
| revnet-38  |             2:17       |    2:09                  |       2:20           |              2:16    |
+------------+------------------------+--------------------------+----------------------+----------------------+
| revnet-110 |             6:59       |    3:42                  |       7:03           |              3:50    |
+------------+------------------------+--------------------------+----------------------+----------------------+
| revnet-164 |            13:09       |    7:21                  |   13:12              |              7:17    |
+------------+------------------------+--------------------------+----------------------+----------------------+

Memory consumption of model training in PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------+--------------------------+----------------------+----------------------+------------------------+--------------------------+----------------------+----------------------+
|               Layers                              |               Parameters                    |               Parameters (MB)                     |               Activations (MB)              |
+------------------------+--------------------------+----------------------+----------------------+------------------------+--------------------------+----------------------+----------------------+
|    ResNet              |      RevNet              |    ResNet            |      RevNet          |    ResNet              |      RevNet              |    ResNet            |      RevNet          |
+========================+==========================+======================+======================+========================+==========================+======================+======================+
|               32       |    38                    |       466906         |          573994      |             1.9        |    2.3                   |       238.6          |              85.6    |
+------------------------+--------------------------+----------------------+----------------------+------------------------+--------------------------+----------------------+----------------------+
|              110       |    110                   |       1730714        |           1854890    |             6.8        |    7.3                   |       810.7          |              85.7    |
+------------------------+--------------------------+----------------------+----------------------+------------------------+--------------------------+----------------------+----------------------+
|              164       |    164                   |   1704154            |         1983786      |            6.8         |    7.9                   |   2452.8             |             432.7    |
+------------------------+--------------------------+----------------------+----------------------+------------------------+--------------------------+----------------------+----------------------+

The `ResNet` model is the conventional Residual Network implementation in PyTorch, while
the RevNet model uses the `memcnn.InvertibleModuleWrapper` to achieve memory savings.

Works using MemCNN
------------------

* `MemCNN: a Framework for Developing Memory Efficient Deep Invertible Networks <https://openreview.net/forum?id=r1KzqK1wz>`__ by Sil C. van de Leemput et al.
* `Reversible GANs for Memory-efficient Image-to-Image Translation <https://arxiv.org/abs/1902.02729>`__ by Tycho van der Ouderaa et al.
* `Chest CT Super-resolution and Domain-adaptation using Memory-efficient 3D Reversible GANs <https://openreview.net/forum?id=SkxueFsiFV>`__ by Tycho van der Ouderaa et al.
* `iUNets: Fully invertible U-Nets with Learnable Up- and Downsampling <https://arxiv.org/abs/2005.05220>`__ by Christian Etmann et al.

Citation
--------

Sil C. van de Leemput, Jonas Teuwen, Bram van Ginneken, and Rashindra Manniesing.
MemCNN: A Python/PyTorch package for creating memory-efficient invertible neural networks.
Journal of Open Source Software, 4, 1576, http://dx.doi.org/10.21105/joss.01576, 2019.

If you use our code, please cite:

.. code:: bibtex

    @article{vandeLeemput2019MemCNN,
      journal = {Journal of Open Source Software},
      doi = {10.21105/joss.01576},
      issn = {2475-9066},
      number = {39},
      publisher = {The Open Journal},
      title = {MemCNN: A Python/PyTorch package for creating memory-efficient invertible neural networks},
      url = {http://dx.doi.org/10.21105/joss.01576},
      volume = {4},
      author = {Sil C. {van de} Leemput and Jonas Teuwen and Bram {van} Ginneken and Rashindra Manniesing},
      pages = {1576},
      date = {2019-07-30},
      year = {2019},
      month = {7},
      day = {30},
    }
