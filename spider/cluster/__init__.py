"""
    spider.cluster sub-package
    __init__.py

    @author: david johnson

    Primitive that clusters raw data
   
    defines the module index
"""

# to allow from spider.cluster import *
__all__ = ["ssc_cvx",
           "ssc_admm",
           "ssc_omp",
           "kss",
           "ekss"]

## modules in spider.cluster
# to allow spider.cluster.XXX

## sub-packages
from .ssc_cvx import SSC_CVX
from .ssc_admm import SSC_ADMM
from .kss import KSS
from .ekss import EKSS
from .ssc_omp import SSC_OMP
