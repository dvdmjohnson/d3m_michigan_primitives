'''
    spider.tests: featurization_i3d_test.py

    @natlouis

    Unit tests for the spider.featurization.i3d module
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

from spider.featurization.i3d import I3D, I3DHyperparams

# disable TF warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class testI3D(unittest.TestCase):

    def test_model(self):
        """ Verify shapes of features at varying layers in the model
        """
        tf.reset_default_graph()
        hp = I3DHyperparams(num_gpus=1, output_layer='avg_pool_mixed_5c')
        i3d = I3D(hyperparams=hp, random_seed=randint(0,2**32-1))

        parent_directory = os.path.abspath(os.path.join(os.path.dirname(common_primitives.__file__),os.pardir))
        dataset_doc_path = os.path.join(parent_directory,'tests','data','datasets','video_dataset_1','datasetDoc.json')
        dataset = D3MDatasetLoader().load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        df0 =  dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_to_dataframe.Hyperparams(dataframe_resource='0')).produce(inputs=dataset).value  
        
        vr_hyperparams_class = video_reader.VideoReaderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        vr = video_reader.VideoReaderPrimitive(hyperparams=vr_hyperparams_class.defaults())
        input_df = vr.produce(inputs=df0).value 

        features = i3d.produce(inputs=input_df).value
        self.assertEqual(features.shape[1],1024)

if __name__ == '__main__':
    unittest.main()

