'''
    spider.tests: supervised_learning_goturn_test.py

    @natlouis

    Unit tests for the spider.supervised_learning.goturn module
'''

import unittest
import tempfile
import os.path
from random import randint
import shutil
import numpy as np
import tensorflow as tf

from d3m import container
from d3m.container.dataset import D3MDatasetLoader
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
import common_primitives
from common_primitives import dataset_to_dataframe
from common_primitives import video_reader

from spider.supervised_learning.goturn import GoTurn, GoTurnHyperparams

class testGoTurn(unittest.TestCase):
    
    def test_model(self):
        """ Verify network accepts appropriate inputs and produces of the proper shape
        """
        hp = GoTurnHyperparams(num_gpus=1, num_epochs=1, learning_rate=1e-5, momentum=0.9, weight_decay=0.0005)
        goturn = GoTurn(hyperparams=hp, random_seed=randint(0,2**32-1))

        parent_directory = os.path.abspath(os.path.join(os.path.dirname(common_primitives.__file__),os.pardir))
        dataset_doc_path = os.path.join(parent_directory,'tests','data','datasets','video_dataset_1','datasetDoc.json')
        dataset = D3MDatasetLoader().load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        df0 =  dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value

        vr_hyperparams_class = video_reader.VideoReaderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        vr = video_reader.VideoReaderPrimitive(hyperparams=vr_hyperparams_class.defaults())
        input_df = vr.produce(inputs=df0).value

        #Train
        train_targets = container.ndarray(np.random.rand(len(input_df.iloc[:,1][0]),4)) #Randomize targets for testing
        goturn.set_training_data(inputs=input_df, targets=train_targets)
        goturn.fit()

        #Test
        predictions = goturn.produce(inputs=input_df, targets=train_targets).value
        self.assertEqual(predictions.shape[1],4)

if __name__ == '__main__':
    unittest.main()

