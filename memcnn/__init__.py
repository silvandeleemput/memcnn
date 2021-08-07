# -*- coding: utf-8 -*-

"""Top-level package for MemCNN."""

__author__ = """Sil van de Leemput"""
__email__ = 'silvandeleemput@gmail.com'
__version__ = '1.5.1'


from memcnn.models.revop import ReversibleBlock, InvertibleModuleWrapper, create_coupling, is_invertible_module
from memcnn.models.additive import AdditiveCoupling
from memcnn.models.affine import AffineCoupling, AffineAdapterNaive, AffineAdapterSigmoid

__all__ = [
    'AdditiveCoupling',
    'AffineCoupling',
    'AffineAdapterNaive',
    'AffineAdapterSigmoid',
    'InvertibleModuleWrapper',
    'ReversibleBlock',
    'create_coupling',
    'is_invertible_module'
]
