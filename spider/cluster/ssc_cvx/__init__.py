"""
    spider.cluster.ssc_admm sub-package
    __init__.py

    @author: agitlin

    Primitive that uses the sparse subspace clustering (SSC) algorithm
    with convex optimization to cluster data into subspaces

    defines the module index
"""

from .ssc_cvx import SSC_CVX, SSC_CVXHyperparams
