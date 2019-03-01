"""
    spider.preprocessing sub-package
    __init__.py

    @author: erichof

    Primitive that processes video data prior to feeding it into a neural network
   
    defines the module index
"""

# to allow from spider.preprocessing import *
__all__ = ["trecs"]

## sub-packages
from .trecs import trecs

