"""
    spider.distance.metricl sub-package
    __init__.py

    @author: david johnson

    Supervised and semi-supervised primitives that learn distance functions.
   
    defines the module index
"""

__all__ = ["rfd"]

from .rfd import RFD
from .utils import get_random_constraints, normalize_labels



