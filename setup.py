import os
import sys
from distutils.core import setup
from setuptools.command.install import install

# circleci.py version
VERSION = "0.1.0"

with open("README.rst", 'r') as fh:
    long_description = fh.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name='MemCNN',
    version=VERSION,
    author='S.C. van de Leemput',
    author_email='silvandeleemput@gmail.com',
    packages=['memcnn'],
    scripts=[],
    url='http://pypi.python.org/pypi/memcnn/',
    license='LICENSE.txt',
    description='A PyTorch framework for developing memory efficient deep invertible networks.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    install_requires=[
        "tensorflow >= 1.11.0"
        "torch >= 0.4.0",
        "torchvision >= 0.2.1",
        "tensorboardX >= 1.4",
        "SimpleITK >= 1.0.1",
        "tqdm >= 4.19.5",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent"
        ],
    keywords='MemCNN invertible PyTorch',
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
