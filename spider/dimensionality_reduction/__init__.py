"""
    spider.dimensionality_reduction sub-package
    __init__.py

    @author: agitlin

    Primitive that reduces dimensionality of collection of vectors
   
    defines the module index
"""

# to allow from spider.dimensionality_reduction import *
__all__ = ["pcp_ialm",
           "go_dec",
           "rpca_lbd"]

## modules in spider.dimensionality_reduction
# to allow spider.dimensionality_reduction.XXX
# import base

## sub-packages
from .pcp_ialm import PCP_IALM, PCP_IALMHyperparams
from .go_dec import GO_DEC, GO_DECHyperparams
from .rpca_lbd import RPCA_LBD

