import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, featurization
from d3m import container, utils as d3m_utils
from common_primitives import dataset_to_dataframe
import collections
import os
import warnings
import pickle
import stopit
import copy
import numpy as np

import time
import sys
import cv2
import tensorflow as tf

from spider.featurization.i3d.utils import *
__all__ = ('I3D')

Inputs = container.DataFrame
Outputs = container.ndarray

#I3D architecture heavily adapted from M-PACT Github repo: https://github.com/MichiganCOG/M-PACT

class I3DHyperparams(hyperparams.Hyperparams):
    num_gpus = hyperparams.Bounded[int](lower=0,
        upper=None,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description='Number of gpus (not gpu cores) this process is allowed to use.')
    output_layer = hyperparams.Enumeration[str](values=[
        'mixed_3b', 'mixed_3c', 'mixed_4b', 'mixed_4c', 'mixed_4d', 'mixed_4e', 'mixed_4f', 'mixed_5b', 'mixed_5c', 'avg_pool_mixed_5c', 'final'],
        default='avg_pool_mixed_5c',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Select from last feature layer or from classification layer.')

class I3D(featurization.FeaturizationTransformerPrimitiveBase[Inputs, Outputs, I3DHyperparams]):
    
    metadata= metadata_module.PrimitiveMetadata({
        'id': 'a7df6ced-79b3-43b5-8360-32f8ff263748',
        'version': '0.0.5',
        'name': 'I3D',
        'description': """Uses a pre-trained 3D convolutional neural network to featurize RGB videos.""",
        'keywords' : ['feature extraction', 'video', 'i3d', 'kinetics', 'deep learning', 'pre-trained'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                'https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/featurize/i3d/i3d.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
            'citation': """@inproceeding{@inproceedings{carreira2017quo,
                title={Quo vadis, action recognition? a new model and the kinetics dataset},
                author={Carreira, Joao and Zisserman, Andrew},
                booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                pages={4724--4733},
                year={2017},
                organization={IEEE}}"""
        },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package': 'librosa',
             'version': '0.5.1'
            },
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package': 'cvxpy',
             'version': '0.4.11'
            },
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://gitlab.datadrivendiscovery.org/michigan/spider.git@{git_commit}#egg=spider'.format(
             git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                'package': 'ffmpeg',
                'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.feature_extraction.i3d.umich',
        'hyperparams_to_tune': ['output_layer'],
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK],
        'primitive_family': metadata_module.PrimitiveFamily.FEATURE_EXTRACTION
    })

    def __init__(self, *, hyperparams: I3DHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        """
        I3D Video Featurization
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        #Mapping to layer number in I3D model
        i3d_layer_dictionary = {'mixed_3b':'29','mixed_3c':'49','mixed_4b':'70','mixed_4c':'90','mixed_4d':'110','mixed_4e':'130','mixed_4f':'150','mixed_5b':'171','mixed_5c':'191','avg_pool_mixed_5c':'192','final':'logits'}

        self._num_gpus = hyperparams['num_gpus']
        params = hyperparams['output_layer']
        self._output_feature_layer = i3d_layer_dictionary[params]

        self._model_name = 'i3d'
        self._weights_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights')
        self._weights_filename = 'i3d_rgb_kinetics.npy'
        self._input_dims = 250 #input video's frame length (to i3d model)
        self._output_frame_shape = (224,224,3)
        self._output_video_shape = (self._input_dims,) + self._output_frame_shape

        #Create session
        config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU':self._num_gpus})
        self._sess = tf.Session(config=config)

    def _unit_3d(self, layer_numbers, input_layer, kernel_size=(1,1,1,1), stride=(1,1,1), activation_fn=tf.nn.relu, use_batch_norm=True, use_bias=False, is_training=True, name='unit_3d', freeze=False):
            """
            Args:
                :layer_numbers:   List detailing the connecting layer indices
                :input_layer:     Input layer to the conv_block
                :kernel_size:     List detailing the height, width  and temporal dimension of the kernel
                :strides:         Integer value for stride between filters (Height, Width and Temporal width)
                :activation_fn:   Activation function to be applied at the end of 3d convolution
                :use_batch_norm:  Boolean indicating the use of batch normalization
                :use_bias:        Boolean indication the use of bias
                :name:            Name of 3d convolution unit

            Return:
                :layers:        Stack of layers
            """

            # BIAS IS NOT USED BUT OUR LAYER UTILS DOES NOT OFFER THE OPTION TO AVOID BIAS!!

            layers = {}

            layers[layer_numbers[0]] = conv3d_layer(input_tensor = input_layer, filter_dims = kernel_size, name = 'RGB/inception_i3d/' + name + '/conv_3d', stride_dims = stride, non_linear_fn = None, use_bias=use_bias, trainable=freeze)

            if use_batch_norm:
                layers[layer_numbers[1]] = batch_normalization(layers[layer_numbers[0]], training = is_training, name = 'RGB/inception_i3d/' + name + '/batch_norm', trainable=freeze)

                if activation_fn is not None:
                    layers[layer_numbers[2]] = activation_fn(layers[layer_numbers[1]])

                # END IF

            else:
                if activation_fn is not None:
                    layers[layer_numbers[1]] = activation_fn(layers[layer_numbers[0]])

                # END IF

            # END IF

            return layers

    def _inference(self, inputs, is_training, input_dims, output_dims, seq_length, scope, dropout_rate = 0.7, return_layer=['logits'], weight_decay=0.0):
        """
        Args:
            :inputs:       Input to model of shape [Frames x Height x Width x Channels]
            :is_training:  Boolean variable indicating phase (TRAIN OR TEST)
            :input_dims:   Length of input sequence
            :output_dims:  Integer indicating total number of classes in final prediction
            :seq_length:   Length of output sequence from LSTM
            :scope:        Scope name for current model instance
            :dropout_rate: Value indicating proability of keep inputs
            :return_layer: List of strings matching name of a layer in current model
            :weight_decay: Double value of weight decay

        Return:
            :layers[return_layer]: The requested layer's output tensor
        """

        ############################################################################
        #                Creating ResNet50 + LSTM Network Layers                   #
        ############################################################################

        with tf.name_scope(scope, 'i3d', [inputs]):

            layers = {}

            layers.update(self._unit_3d(layer_numbers=['1','2','3'], input_layer=inputs, kernel_size=[7,7,7,64], stride=[2,2,2], name='Conv3d_1a_7x7', is_training=False))

            layers['4'] = max_pool3d_layer(layers['3'], filter_dims=[1,1,3,3,1], stride_dims=[1,1,2,2,1], padding='SAME', name='RGB/inception_i3d/MaxPool3d_2a_3x3')

            layers.update(self._unit_3d(layer_numbers=['5','6','7'], input_layer=layers['4'], kernel_size=[1,1,1,64], name='Conv3d_2b_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['8','9','10'], input_layer=layers['7'], kernel_size=[3,3,3,192], name='Conv3d_2c_3x3', is_training=False))

            layers['11_inp'] = max_pool3d_layer(layers['10'], filter_dims=[1,1,3,3,1], stride_dims=[1,1,2,2,1], padding='SAME', name='RGB/inception_i3d/MaxPool3d_3a_3x3')

            #### Mixed_3b ####

            layers.update(self._unit_3d(layer_numbers=['11','12','13'], input_layer=layers['11_inp'], kernel_size=[1,1,1,64], name='Mixed_3b/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['14','15','16'], input_layer=layers['11_inp'], kernel_size=[1,1,1,96], name='Mixed_3b/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['17','18','19'], input_layer=layers['16'], kernel_size=[3,3,3,128], name='Mixed_3b/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['20','21','22'], input_layer=layers['11_inp'], kernel_size=[1,1,1,16], name='Mixed_3b/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['23','24','24'], input_layer=layers['22'], kernel_size=[3,3,3,32], name='Mixed_3b/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['25'] = max_pool3d_layer(layers['11_inp'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_3b/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['26','27','28'], input_layer=layers['25'], kernel_size=[1,1,1,32], name='Mixed_3b/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['29'] = tf.concat([layers['13'], layers['19'], layers['24'], layers['28']], 4)

            #### END OF MIXED_3b ####

            #### Mixed_3c ####

            layers.update(self._unit_3d(layer_numbers=['30','31','32'], input_layer=layers['29'], kernel_size=[1,1,1,128], name='Mixed_3c/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['33','34','35'], input_layer=layers['29'], kernel_size=[1,1,1,128], name='Mixed_3c/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['36','37','38'], input_layer=layers['35'], kernel_size=[3,3,3,192], name='Mixed_3c/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['39','40','41'], input_layer=layers['29'], kernel_size=[1,1,1,32], name='Mixed_3c/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['42','43','44'], input_layer=layers['41'], kernel_size=[3,3,3,96], name='Mixed_3c/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['45'] = max_pool3d_layer(layers['29'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_3c/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['46','47','48'], input_layer=layers['45'], kernel_size=[1,1,1,64], name='Mixed_3c/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['49'] = tf.concat([layers['32'], layers['38'], layers['44'], layers['48']], 4)

            #### END OF MIXED_3c ####

            layers['50'] = max_pool3d_layer(layers['49'], filter_dims=[1,3,3,3,1], stride_dims=[1,2,2,2,1], padding='SAME', name='RGB/inception_i3d/MaxPool3d_4a_3x3')

            #### Mixed_4b ####

            layers.update(self._unit_3d(layer_numbers=['51','52','53'], input_layer=layers['50'], kernel_size=[1,1,1,192], name='Mixed_4b/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['54','55','56'], input_layer=layers['50'], kernel_size=[1,1,1,96], name='Mixed_4b/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['57','58','59'], input_layer=layers['56'], kernel_size=[3,3,3,208], name='Mixed_4b/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['60','61','62'], input_layer=layers['50'], kernel_size=[1,1,1,16], name='Mixed_4b/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['63','64','65'], input_layer=layers['62'], kernel_size=[3,3,3,48], name='Mixed_4b/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['66'] = max_pool3d_layer(layers['50'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_4b/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['67','68','69'], input_layer=layers['66'], kernel_size=[1,1,1,64], name='Mixed_4b/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['70'] = tf.concat([layers['53'], layers['59'], layers['65'], layers['69']], 4)

            #### END OF MIXED_4b ####

            #### Mixed_4c ####

            layers.update(self._unit_3d(layer_numbers=['71','72','73'], input_layer=layers['70'], kernel_size=[1,1,1,160], name='Mixed_4c/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['74','75','76'], input_layer=layers['70'], kernel_size=[1,1,1,112], name='Mixed_4c/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['77','78','79'], input_layer=layers['76'], kernel_size=[3,3,3,224], name='Mixed_4c/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['80','81','82'], input_layer=layers['70'], kernel_size=[1,1,1,24], name='Mixed_4c/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['83','84','85'], input_layer=layers['82'], kernel_size=[3,3,3,64], name='Mixed_4c/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['86'] = max_pool3d_layer(layers['70'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_4c/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['87','88','89'], input_layer=layers['86'], kernel_size=[1,1,1,64], name='Mixed_4c/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['90'] = tf.concat([layers['73'], layers['79'], layers['85'], layers['89']], 4)

            #### END OF MIXED_4c ####

            #### Mixed_4d ####

            #with tf.variable_scope('Mixed_4d'):
            #    with tf.variable_scope('Branch_0'):
            layers.update(self._unit_3d(layer_numbers=['91','92','93'], input_layer=layers['90'], kernel_size=[1,1,1,128], name='Mixed_4d/Branch_0/Conv3d_0a_1x1', is_training=False))

                # END WITH

                #with tf.variable_scope('Branch_1'):
            layers.update(self._unit_3d(layer_numbers=['94','95','96'], input_layer=layers['90'], kernel_size=[1,1,1,128], name='Mixed_4d/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['97','98','99'], input_layer=layers['96'], kernel_size=[3,3,3,256], name='Mixed_4d/Branch_1/Conv3d_0b_3x3', is_training=False))

                # END WITH

                #with tf.variable_scope('Branch_2'):
            layers.update(self._unit_3d(layer_numbers=['100','101','102'], input_layer=layers['90'], kernel_size=[1,1,1,24], name='Mixed_4d/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['103','104','105'], input_layer=layers['102'], kernel_size=[3,3,3,64], name='Mixed_4d/Branch_2/Conv3d_0b_3x3', is_training=False))

                # END WITH

                #with tf.variable_scope('Branch_3'):
            layers['106'] = max_pool3d_layer(layers['90'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_4d/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['107','108','109'], input_layer=layers['106'], kernel_size=[1,1,1,64], name='Mixed_4d/Branch_3/Conv3d_0b_1x1', is_training=False))

                # END WITH

            layers['110'] = tf.concat([layers['93'], layers['99'], layers['105'], layers['109']], 4)

            # END WITH

            #### END OF MIXED_4d ####

            #### Mixed_4e ####

            layers.update(self._unit_3d(layer_numbers=['111','112','113'], input_layer=layers['110'], kernel_size=[1,1,1,112], name='Mixed_4e/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['114','115','116'], input_layer=layers['110'], kernel_size=[1,1,1,144], name='Mixed_4e/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['117','118','119'], input_layer=layers['116'], kernel_size=[3,3,3,288], name='Mixed_4e/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['120','121','122'], input_layer=layers['110'], kernel_size=[1,1,1,32], name='Mixed_4e/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['123','124','125'], input_layer=layers['122'], kernel_size=[3,3,3,64], name='Mixed_4e/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['126'] = max_pool3d_layer(layers['110'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_4e/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['127','128','129'], input_layer=layers['126'], kernel_size=[1,1,1,64], name='Mixed_4e/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['130'] = tf.concat([layers['113'], layers['119'], layers['125'], layers['129']], 4)

            #### END OF MIXED_4e ####

            #### Mixed_4f ####

            layers.update(self._unit_3d(layer_numbers=['131','132','133'], input_layer=layers['130'], kernel_size=[1,1,1,256], name='Mixed_4f/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['134','135','136'], input_layer=layers['130'], kernel_size=[1,1,1,160], name='Mixed_4f/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['137','138','139'], input_layer=layers['136'], kernel_size=[3,3,3,320], name='Mixed_4f/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['140','141','142'], input_layer=layers['130'], kernel_size=[1,1,1,32], name='Mixed_4f/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['143','144','145'], input_layer=layers['142'], kernel_size=[3,3,3,128], name='Mixed_4f/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['146'] = max_pool3d_layer(layers['130'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_4f/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['147','148','149'], input_layer=layers['146'], kernel_size=[1,1,1,128], name='Mixed_4f/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['150'] = tf.concat([layers['133'], layers['139'], layers['145'], layers['149']], 4)

            #### END OF MIXED_4f ####

            layers['151'] = max_pool3d_layer(layers['150'], filter_dims=[1,2,2,2,1], stride_dims=[1,2,2,2,1], padding='SAME', name='RGB/inception_i3d/MaxPool3d_5a_2x2')

            #### Mixed_5b ####

            layers.update(self._unit_3d(layer_numbers=['152','153','154'], input_layer=layers['151'], kernel_size=[1,1,1,256], name='Mixed_5b/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['155','156','157'], input_layer=layers['151'], kernel_size=[1,1,1,160], name='Mixed_5b/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['158','159','160'], input_layer=layers['157'], kernel_size=[3,3,3,320], name='Mixed_5b/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['161','162','163'], input_layer=layers['151'], kernel_size=[1,1,1,32], name='Mixed_5b/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['164','165','166'], input_layer=layers['163'], kernel_size=[3,3,3,128], name='Mixed_5b/Branch_2/Conv3d_0a_3x3', is_training=False))

            layers['167'] = max_pool3d_layer(layers['151'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_5b/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['168','169','170'], input_layer=layers['167'], kernel_size=[1,1,1,128], name='Mixed_5b/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['171'] = tf.concat([layers['154'], layers['160'], layers['166'], layers['170']], 4)

            #### END OF MIXED_5b ####

            #### Mixed_5c ####

            layers.update(self._unit_3d(layer_numbers=['172','173','174'], input_layer=layers['171'], kernel_size=[1,1,1,384], name='Mixed_5c/Branch_0/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['175','176','177'], input_layer=layers['171'], kernel_size=[1,1,1,192], name='Mixed_5c/Branch_1/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['178','179','180'], input_layer=layers['177'], kernel_size=[3,3,3,384], name='Mixed_5c/Branch_1/Conv3d_0b_3x3', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['181','182','183'], input_layer=layers['171'], kernel_size=[1,1,1,48], name='Mixed_5c/Branch_2/Conv3d_0a_1x1', is_training=False))

            layers.update(self._unit_3d(layer_numbers=['184','185','186'], input_layer=layers['183'], kernel_size=[3,3,3,128], name='Mixed_5c/Branch_2/Conv3d_0b_3x3', is_training=False))

            layers['187'] = max_pool3d_layer(layers['171'], filter_dims=[1,3,3,3,1], stride_dims=[1,1,1,1,1], padding='SAME', name='RGB/inception_i3d/Mixed_5c/Branch_3/MaxPool3d_0a_3x3')

            layers.update(self._unit_3d(layer_numbers=['188','189','190'], input_layer=layers['187'], kernel_size=[1,1,1,128], name='Mixed_5c/Branch_3/Conv3d_0b_1x1', is_training=False))

            layers['191'] = tf.concat([layers['174'], layers['180'], layers['186'], layers['190']], 4)

            #### END OF MIXED_5c ####

            layers['192'] = tf.expand_dims(tf.reduce_mean(avg_pool3d_layer(layers['191'], filter_dims=[1,2,7,7,1], stride_dims=[1,1,1,1,1], padding='VALID', name='RGB/inception_i3d/avg_pooling'), axis=1), 1)

            layers['193'] = dropout(layers['192'], rate=dropout_rate, training=is_training)

            layers.update(self._unit_3d(layer_numbers=['logits_pre'], input_layer=layers['193'], kernel_size=[1,1,1,output_dims], name='Logits/Conv3d_0c_1x1', is_training=is_training, activation_fn=None, use_batch_norm=False, freeze=True))

            layers['logits'] = tf.expand_dims(tf.reduce_mean(tf.squeeze(layers['logits_pre'], [2, 3]), axis=1), 1)


        # END WITH
        return [layers[x] for x in return_layer]

        """ Function to return loss calculated on given network """
    def _loss(self, logits, labels, loss_type='full_loss'):
        """
        Args:
            :logits: Unscaled logits returned from final layer in model
            :labels: True labels corresponding to loaded data
        Return:
            Cross entropy loss value
        """
        labels = tf.cast(labels, tf.int64)

        cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return cross_entropy_loss

    def _input_handler(self, x):
        """
        Pre-processes video to be correct length (and values) for I3D network
                Arguments:
            - data: string file location of video or np.array of video data 
        Returns:
            - x: 4D np.array of shape (1, 250, 224, 224, 3) [batch dimension, frames, height, width, channels]
        """

        #Normalize frames to have RGB values between -1 and 1
        x = (x/255.) * 2. -  1.

        #Resize all frames to be the required dimension
        out_vid = []
        for vid in x:
            out_vid.append(cv2.resize(vid, (self._output_frame_shape[1], self._output_frame_shape[0])))
        x = np.array(out_vid)
        
        if len(x.shape) is 4:
            x = np.expand_dims(x, axis=0)
        
        #For now, shorten sequence length to first 250 frames or repeat until 250 frames in length
        if x.shape[1] > self._input_dims:
            x = x[:,:self._input_dims,:,:,:]
        elif x.shape[1] < self._input_dims:
            x = np.repeat(x,self._input_dims//x.shape[1],axis=1)
            x = np.concatenate((x,x[:,:self._input_dims%x.shape[1],:,:,:]),axis=1)            
        assert x.shape == (1,) + self._output_video_shape  
        return x

    @base.singleton
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        with stopit.ThreadingTimeout(timeout) as timer:

            features = container.List()
            #import pdb; pdb.set_trace();

            num_columns = inputs.metadata.query((metadata_module.ALL_ELEMENTS,))['dimension']['length']
            column_to_use = None

            #Find which column contains the semantic type http://schema.org/VideoObject
            for idx in range(num_columns):
                column_metadata = inputs.metadata.query((metadata_module.ALL_ELEMENTS,idx))
                semantic_types = column_metadata['semantic_types']
                structural_type = column_metadata['structural_type']

                #The column to use will contain the VideoObject semantic type but not the FileName semantic type
                if 'http://schema.org/VideoObject' in semantic_types and structural_type == container.ndarray:
                    column_to_use = idx 

            assert(column_to_use is not None)

            
            #Load model checkpoints
            filename = os.path.join(self._weights_directory, self._weights_filename)
            print('Weights filename: {}'.format(filename))
            ckpt = np.load(filename, encoding='latin1')

            video_placeholder = tf.placeholder(tf.float32, shape=(1,) + self._output_video_shape)
            XX = self._inference(video_placeholder, is_training=False, input_dims=self._input_dims, output_dims=51, seq_length=1, return_layer=[self._output_feature_layer], scope='RGB/inception_i3d')

            init = tf.global_variables_initializer()
            self._sess.run(init)

            if ckpt == None:
                raise IOError('Model weights not found from  %s' % filename)
            else:
                #Initialize model variables
                initialize_from_dict(self._sess, ckpt, self._model_name)

            for data in inputs.iloc[:,column_to_use]:
                x = self._input_handler(data)

                video_features = self._sess.run(XX, feed_dict={video_placeholder: x})[0] #returning only 1 layer
                video_features = np.squeeze(video_features)

                features.append(video_features)

        return base.CallResult(np.array(features))

        if timer.state != timer.EXECUTED:
            raise TimeoutError('I3D produce time out')

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: I3DHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
