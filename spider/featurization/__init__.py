"""
    spider.featurization sub-package
    __init__.py

    @author: jason corso

    Primitive that featurization raw data into usable vector representations
   
    defines the module index
"""

# to allow from spider.featurization import *
__all__ = [
    "audio_slicer",
    "logmelspectrogram",
    "vgg16"
]
