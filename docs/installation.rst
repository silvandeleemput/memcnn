.. highlight:: shell

============
Installation
============

Requirements
------------

-  `Python <https://python.org/>`__ 2.7 or 3.5+
-  `PyTorch <http://pytorch.org/>`__ 1.1 (0.4 downwards compatible, CUDA
   support recommended)


Stable release
--------------

These are the preferred methods to install MemCNN, as they will always install the most recent stable release.

PyPi
^^^^

To install MemCNN using the Python package manager, run this command in your terminal:

.. code-block:: console

    $ pip install memcnn

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Anaconda
^^^^^^^^

To install MemCNN using Anaconda, run this command in your terminal:

.. code-block:: console

    $ conda install -c silvandeleemput -c pytorch -c simpleitk -c conda-forge memcnn

If you don't have `conda`_ installed, this `Anaconda installation guide`_ can guide
you through the process.

.. _conda: https://www.anaconda.com/
.. _Anaconda installation guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

From sources
------------

The sources for MemCNN can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/silvandeleemput/memcnn

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/silvandeleemput/memcnn/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/silvandeleemput/memcnn
.. _tarball: https://github.com/silvandeleemput/memcnn/tarball/master


Using docker
------------

MemCNN has several pre-build docker images that are hosted on dockerhub.
You can directly pull these and to have a working environment for running the experiments.

Run image from repository
^^^^^^^^^^^^^^^^^^^^^^^^^

Run the latest docker build of MemCNN from the repository (automatically pulls the image):

.. code-block:: console

    $ docker run --shm-size=4g --runtime=nvidia -it silvandeleemput/memcnn:latest

For ``--runtime=nvidia`` to work `nvidia-docker <https://github.com/nvidia/nvidia-docker>`__ must be installed on your system.
It can be omitted but this will drop GPU training support.

This will open a preconfigured bash shell, which is correctly configured
to run the experiments. The latest version has Ubuntu 18.04 and Python 3.6 installed.

By default, the datasets and experimental results will be put inside the created
docker container under: ``\home\user\data`` and
``\home\user\experiments`` respectively.

Build image from source
^^^^^^^^^^^^^^^^^^^^^^^

Requirements:

-  NVIDIA graphics card and the proper NVIDIA-drivers on your system


The following bash commands will clone this repository and do a one-time
build of the docker image with the right environment installed:

.. code-block:: console

    $ git clone https://github.com/silvandeleemput/memcnn.git
    $ docker build ./memcnn/docker --tag=silvandeleemput/memcnn:latest

After the one-time install on your machine, the docker image can be invoked
using the same commands as listed above.

Experiment configuration file
-----------------------------

To run the experiments, MemCNN requires setting up a configuration file containing locations to put the data files.
This step is not necessary for the docker builds.


The configuration file ``config.json`` goes in the ``/memcnn/config/`` directory of the library and should be formatted as follows:

.. code:: json

    {
        "data_dir": "/home/user/data",
        "results_dir": "/home/user/experiments"
    }

* data_dir    : location for storing the input training datasets
* results_dir : location for storing the experiment files during training

Change the data paths to your liking.

If you are unsure where MemCNN and/or the configuration file is located on your machine run:

.. code-block:: console

    $ python -m memcnn.train

If the configuration file is not setup correctly, this command should give the user the correct path to the configuration file.
Next, create/edit the file at the given location.
