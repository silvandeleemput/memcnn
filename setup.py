from distutils.core import setup

setup(
    name='MemCNN',
    version='0.4.0',
    author='S.C. van de Leemput',
    author_email='silvandeleemput@gmail.com',
    packages=['memcnn'],
    scripts=[],
    url='http://pypi.python.org/pypi/memcnn/',
    license='LICENSE.txt',
    description='A PyTorch framework for developing memory efficient deep invertible networks.',
    long_description='A PyTorch framework for developing memory efficient deep invertible networks which'
                     'provides a simple drop-in implementation for ReversibleBlocks with invertible couplings.',
    install_requires=[
        "tensorflow >= 1.11.0"
        "torch >= 0.4.0",
        "torchvision >= 0.2.1",
        "tensorboardX >= 1.4",
        "SimpleITK >= 1.0.1",
        "tqdm >= 4.19.5",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
        ],
    keywords='MemCNN invertible PyTorch',
)
