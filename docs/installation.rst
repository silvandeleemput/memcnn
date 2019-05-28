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

To install MemCNN, run this command in your terminal:

.. code-block:: console

    $ pip install memcnn

This is the preferred method to install MemCNN, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


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


Using NVIDIA docker
-------------------

Requirements
^^^^^^^^^^^^

-  NVIDIA graphics card and the proper NVIDIA-drivers on your system
-  `nvidia-docker <https://github.com/nvidia/nvidia-docker>`__ installed
   on your system

The following bash commands will clone this repository and do a one-time
build of the docker image with the right environment installed:

.. code:: bash

    git clone https://github.com/silvandeleemput/memcnn.git
    docker build ./memcnn/docker --tag=memcnn-docker

After the one-time install on your machine, the docker can be invoked
by:

.. code:: bash

    docker run --shm-size=4g --runtime=nvidia -it memcnn-docker

This will open a preconfigured bash shell, which is correctly configured
to run the experiments from the next section.

The datasets and experimental results will be put inside the created
docker container under: ``\home\user\data`` and
``\home\user\experiments`` respectively
