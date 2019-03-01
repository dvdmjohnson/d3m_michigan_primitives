'''
    spider.tests: featurization_vgg16_test.py

    @tsnowak

    Unit tests for the spider.featurization.vgg16 module
'''

import unittest
import tempfile
import os.path
from random import randint
import shutil
import numpy as np
from keras.preprocessing import image

from d3m import container

from spider.featurization.vgg16 import VGG16, VGG16Hyperparams

# disable TF warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class testVGG16(unittest.TestCase):

    def test_nparray_input(self):
        """ Test input_handler given nparray input and using GPU. 
        """
        hp = VGG16Hyperparams(num_cores = 1, num_gpus = 1, output_layer = 'fc1')
        vgg16 = VGG16(hyperparams=hp, random_seed=randint(0, 2**32-1))
        data = np.ones((224,224,3))
        output = vgg16._input_handler(data)

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 224)
        self.assertEqual(output.shape[3], 3)

        data = np.random.rand(120,75,3)
        output = vgg16._input_handler(data)

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 224)
        self.assertEqual(output.shape[3], 3)

    def test_model(self):
        """ Verify shapes of features at varying layers in the model
        """
        hp = VGG16Hyperparams(num_cores = 1, num_gpus = 0, output_layer = 'fc1')
        vgg16 = VGG16(hyperparams=hp, random_seed=randint(0, 2**32-1))
        data_dir = os.path.dirname(os.path.abspath(__file__))
        imagepath = os.path.join(data_dir, 'data/elephant.jpg')

        img = image.load_img(imagepath, target_size=(224,224))
        x = image.img_to_array(img)
        x2 = np.asarray([x,x])
        data = np.rollaxis(container.ndarray(x2), 3, 1)

        features = vgg16.produce(inputs=data).value
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 4096)

if __name__ == '__main__':
    unittest.main()
