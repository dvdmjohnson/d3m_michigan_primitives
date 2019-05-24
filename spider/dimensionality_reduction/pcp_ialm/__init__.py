"""
    spider.dimensionality_reduction.pcp_ialm sub-package
    __init__.py

    @author: agitlin

    Primitive that uses Robust Principal Component Analysis (RPCA) via
    Principal Component Pursuit (PCP) with the Inexact Augmented Lagrange Multipliers (IALM) method
    to perform dimensionality reduction on a data matrix
    
    defines the module index
"""

from .pcp_ialm import PCP_IALM, PCP_IALMHyperparams
