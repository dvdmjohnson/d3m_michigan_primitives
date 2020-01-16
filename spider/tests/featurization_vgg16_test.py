"""
Unit tests for the spider.featurization.vgg16 module

@author Ryan Szeto (szetor@umich.edu)
"""

import unittest

import numpy as np

from spider.featurization.vgg16 import VGG16, VGG16Hyperparams


class TestVGG16(unittest.TestCase):

    def test_default(self):
        """Runs with the default hyperparameters and checks the output size.

        TODO: Check for the correct output format
        """

        hp = VGG16Hyperparams(
            num_cores=1,
            num_gpus=1,
            output_layer='fc2'
        )
        volumes = {
            'vgg16_weights.h5': '/volumes/8b81f25be4126c5ec088f19901b2b34e9a40e3d46246c9d208a3d727462b4f5c'
        }

        input = np.zeros((1, 2, 20, 20, 3))

        model = VGG16(hyperparams=hp, volumes=volumes, random_seed=123)
        output = model.produce(inputs=input).value

