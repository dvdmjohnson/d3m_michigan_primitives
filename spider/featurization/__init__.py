"""
    spider.featurization sub-package
    __init__.py

    @author: jason corso

    Primitive that featurization raw data into usable vector representations
   
    defines the module index
"""

# to allow from spider.featurization import *
__all__ = ["audio_featurization",
           "audio_slicer",
           "logmelspectrogram",
           "vgg16"]

## sub-packages
from .audio_featurization import audio_featurization
from .audio_slicer import audio_slicer
from .logmelspectrogram import logmelspectrogram
from .vgg16 import VGG16 as vgg16
