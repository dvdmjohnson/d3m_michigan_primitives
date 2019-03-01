"""
    spider.unsupervised_learning sub-package
    __init__.py

    @author: kgilman

    Primitive that performs GRASTA to learn low-rank subspace and sparse outliers from streaming data
   
    defines the module index
"""

# to allow from spider.preprocessing import *
__all__ = ["grasta","grasta_masked",]

## sub-packages
from .grasta import grasta
from .grasta_masked import grasta_masked
from .grouse import grouse


