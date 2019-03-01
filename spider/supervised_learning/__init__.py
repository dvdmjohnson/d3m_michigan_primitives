"""
    spider.supervised_learning sub-package
    __init__.py

    @author: zeyu sun 

    Supervised Learning primitives

    difines the model index
"""

__all__ = ["OWLRegression", "GoTurn"]

from .owl import OWLRegression
from .goturn import GoTurn
