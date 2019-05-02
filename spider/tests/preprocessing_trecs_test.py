'''
    spider.tests: preprocessing_trecs_test.py

    @erichof

    Unit tests for the spider.preprocessing.trecs module
'''

import unittest
import os.path
from random import randint
import numpy as np

from d3m import container, exceptions

from spider.preprocessing.trecs import TRECS, TRECSHyperparams

class testTRECS(unittest.TestCase):

    def test_video_input(self):
        """ Test input_handler given video input.         
        """
        hp = TRECSHyperparams(trecs_method='cvr', default_alpha=2.0, output_frames=50)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))
        data_dir = os.path.dirname(os.path.abspath(__file__))
        data = os.path.join(data_dir, 'data/v_Biking_g14_c03.avi')

        output = trecs._input_handler(data)        

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[0], 141)
        self.assertEqual(output.shape[1], 240)
        self.assertEqual(output.shape[2], 320)
        self.assertEqual(output.shape[3], 3)


    def test_nparray_input(self):
        """ Test input_handler given nparray input.         
        """
        hp = TRECSHyperparams(trecs_method='cvr', default_alpha=2.0, output_frames=50)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

        data = np.ones((141, 240, 320, 3))
        output = trecs._input_handler(data)

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[0], 141)
        self.assertEqual(output.shape[1], 240)
        self.assertEqual(output.shape[2], 320)
        self.assertEqual(output.shape[3], 3)


    def test_cvr(self):
        """ Test constant value resampling given nparray input.         
        """
        hp = TRECSHyperparams(trecs_method='cvr', default_alpha=2.0, output_frames=50)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

        data = container.List([container.ndarray(np.ones((141, 240, 320, 3)))])
        output_list = trecs.produce(inputs=data).value
        output = output_list[0]

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[0], 50)
        self.assertEqual(output.shape[1], 240)
        self.assertEqual(output.shape[2], 320)
        self.assertEqual(output.shape[3], 3)


    def test_rr(self):
        """ Test random resampling given nparray input.         
        """
        hp = TRECSHyperparams(trecs_method='rr', default_alpha=1.0, output_frames=50)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

        data = container.List([container.ndarray(np.ones((141, 240, 320, 3)))])
        output_list = trecs.produce(inputs=data).value
        output = output_list[0]

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[0], 50)
        self.assertEqual(output.shape[1], 240)
        self.assertEqual(output.shape[2], 320)
        self.assertEqual(output.shape[3], 3)


    def test_sr(self):
        """ Test sinusoidal resampling given nparray input.         
        """
        hp = TRECSHyperparams(trecs_method='sr', default_alpha=1.0, output_frames=50)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

        data = container.List([container.ndarray(np.ones((141, 240, 320, 3)))])
        output_list = trecs.produce(inputs=data).value
        output = output_list[0]

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[0], 50)
        self.assertEqual(output.shape[1], 240)
        self.assertEqual(output.shape[2], 320)
        self.assertEqual(output.shape[3], 3)


    def test_output_frames(self):
        """ Test the repetition of frames if output_frames > input_frames     
        """
        hp = TRECSHyperparams(trecs_method='cvr', default_alpha=2.0, output_frames=100)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2**32-1))

        data = container.List([container.ndarray(np.ones((90, 240, 320, 3)))])
        output_list = trecs.produce(inputs=data).value
        output = output_list[0]

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[0], 100)
        self.assertEqual(output.shape[1], 240)
        self.assertEqual(output.shape[2], 320)
        self.assertEqual(output.shape[3], 3)


    def test_incorrect_trecs_method(self):
        """ Test that incorrect trecs_methods are not permitted 
        """
        with self.assertRaises(exceptions.InvalidArgumentValueError):
            TRECSHyperparams(trecs_method='other', default_alpha=2.0, output_frames=100)
        
       
 
if __name__ == '__main__':
    unittest.main()
