'''
    spider.tests: preprocessing_trecs_test.py

    @erichof

    Unit tests for the spider.preprocessing.trecs module
'''

import unittest
import os
from random import randint
import numpy as np
from common_primitives.video_reader import VideoReaderPrimitive as VideoReader
from d3m import container, exceptions
from common_primitives import dataset_to_dataframe
from d3m.container.dataset import D3MDatasetLoader
from spider.preprocessing.trecs import TRECS, TRECSHyperparams


class testTRECS(unittest.TestCase):

    def test_cvr(self):
        """ Test constant value resampling given nparray input.
        """
        hp = TRECSHyperparams(trecs_method='cvr', default_alpha=2.0, output_frames=50)
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2 ** 32 - 1))

        dataset_doc_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        'data/video_dataset_1/datasetDoc.json')
        dataset = D3MDatasetLoader().load('file://' + dataset_doc_path)

        df0 = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value

        vr = VideoReader(hyperparams=Hyperparams(resize_to=(240, 320)))
        data = vr.produce(inputs=df0).value

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
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2 ** 32 - 1))

        dataset_doc_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        'data/video_dataset_1/datasetDoc.json')
        dataset = D3MDatasetLoader().load('file://' + dataset_doc_path)

        df0 = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value

        vr = VideoReader(hyperparams=default)
        data = vr.produce(inputs=df0).value

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
        hp = TRECSHyperparams(trecs_method='sr', default_alpha=1.0, output_frames=50)§
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2 ** 32 - 1))

        dataset_doc_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        'data/video_dataset_1/datasetDoc.json')
        dataset = D3MDatasetLoader().load('file://' + dataset_doc_path)

        df0 = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value

        vr = VideoReader(hyperparams=Hyperparams(resize_to=(240, 320)))
        data = vr.produce(inputs=df0).value

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
        trecs = TRECS(hyperparams=hp, random_seed=randint(0, 2 ** 32 - 1))

        dataset_doc_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        'data/video_dataset_1/datasetDoc.json')
        dataset = D3MDatasetLoader().load('file://' + dataset_doc_path)

        df0 = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value

        vr = VideoReader(hyperparams=Hyperparams(resize_to=(240, 320)))
        data = vr.produce(inputs=df0).value

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
