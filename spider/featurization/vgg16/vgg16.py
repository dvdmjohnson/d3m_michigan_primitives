import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, featurization
from d3m import container, utils as d3m_utils


import sys
import os
import os.path
import inspect
import urllib.request, urllib.parse, urllib.error
import warnings
import stopit
import numpy as np
from scipy import misc

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

from . import utils

# Params could include option of weights to load into model, but currently
# model is fixed and no constraints change instance of primitive

__all__ = ('AudioFeaturization',)

Inputs = container.ndarray
Outputs = container.ndarray

class VGG16Hyperparams(hyperparams.Hyperparams):
    num_cores = hyperparams.Bounded[int](
        lower=1,
        upper=None,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description='Number of cpu cores this process is allowed to use.')
    num_gpus = hyperparams.Bounded[int](
        lower=0,
        upper=None,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description='Number of gpus (not gpu cores) this process is allowed to use.')
    output_layer = hyperparams.Enumeration[str](
        values=['block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1',
               'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3',
               'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
               'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 'flatten', 'fc1',
               'fc2', 'predictions'],
        default='fc1',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Name of the network layer to output as features.')

class VGG16(featurization.FeaturizationTransformerPrimitiveBase[Inputs, Outputs, VGG16Hyperparams]):

    metadata = metadata_module.PrimitiveMetadata({
        "id": 'fda1e12e-d89e-49f3-86cb-7dfaa82bbb9c',
        'version': '0.0.5',
        'name': 'VGG16',
        'description': """Uses a pre-trained 16-layer convolutional neural network to featurize RGB images.""",
        'keywords': ['feature extraction', 'image', 'deep learning', 'convolutional neural network', 'pre-trained'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                'https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/featurization/vgg16/vgg16.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
            'citation': """@Article{Simonyan14c,
                 author       = "Simonyan, K. and Zisserman, A.",
                 title        = "Very Deep Convolutional Networks for Large-Scale Image Recognition",
                 journal      = "CoRR",
                 volume       = "abs/1409.1556",
                 year         = "2014"}"""
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
        'python_path': 'd3m.primitives.feature_extraction.vgg16.umich',
        'hyperparams_to_tune': ['output_layer'],
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK],
        'primitive_family': metadata_module.PrimitiveFamily.FEATURE_EXTRACTION
    })

    def __init__(self, *, hyperparams: VGG16Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        """
        Darpa D3M VGG16 Image Featurization Primitive

        Arguments:
            - device: Device type ('cpu' or 'gpu') and number of devices as a tuple.
            - num_cores: Integer number of CPU cores to use.
            - output_feature_layer: String layer name whose features to output.

        Return :
            - None
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._output_feature_layer = hyperparams['output_layer']

        # settable vars unrelated to fit/application of primitive
        self._num_cores = hyperparams['num_cores']
        self._num_gpus = hyperparams['num_gpus']

        config = tf.ConfigProto(intra_op_parallelism_threads=self._num_cores,\
            inter_op_parallelism_threads=self._num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : self._num_cores, 'GPU' : self._num_gpus})
        session = tf.Session(config=config)
        K.set_session(session)

        # helper vars
        self._input_shape = (224, 224, 3)
        self._weights_directory = os.path.join(os.path.abspath(os.path.dirname(utils.__file__)), \
            'weights')
        self._weights_filename = 'vgg16_weights.h5'
        self._interpolation_method = 'bilinear'
        self._base_model = self._model()

        # output variable of list of feature vectors
        self._features = []

    def _model(self):

        """
        Implementation of the VGG16 architecture provided by keras and
            fchollet on github with minor alterations. This implementation is
            designed to be used with a Tensorflow backend. As such, ensure that
            the pixel depth channel is last.

        Arguments:
            - None

        Returns:
            - A Keras model instance.
        """

        # Prepare image for input to model
        img_input = Input(shape=self._input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

        inputs = img_input

        # Create model.
        model = Model(inputs, x, name='vgg16')

        filepath = os.path.join(self._weights_directory, self._weights_filename)
        model.load_weights(filepath)
    
        return model

    def _input_handler(self, data, method='bilinear'):
        """
        Performs necessary manipulations to prepare input
            data for implementation in the VGG16 network

        Arguments:
            - data: string file location of image file, or np.array of an image
                to feed into the network
            - method: optional -- default 'bilinear' --, selects which method of
                interpolation to perform when resizing the input is necessary

        Returns:
            - x: 4D np.array to feed into the VGG16 network. Typically the output
                is always of shape (1, 224, 224, 3)
        """

        # interpolation methods handling
        method_list = {'bilinear', 'nearest', 'lanczos', 'bicubic', 'cubic'}
        if method not in method_list:
            raise ValueError('Method for interpolation must be one of the following %s' \
                % str(method_list))

        # handle if input is a path
        if isinstance(data, str):
            if not os.path.isfile(data):
                raise IOError('Input data file does not exist.')
            else:
                img = image.load_img(data, target_size=(self._input_shape[0],
                    self._input_shape[1]))
                x = image.img_to_array(img)
                if len(x.shape) is 3:
                    x = np.expand_dims(x, axis=0)

        # handle if input is an np.ndarray
        elif isinstance(data, np.ndarray):
            x = utils.interpolate(data, self._input_shape, method)
            if len(x.shape) is 3:
                x = np.expand_dims(x, axis=0)

        # if neither raise error
        else:
            raise TypeError('Input must either be a file path to an image, \
                    or an np.ndarray of an image.')

        assert x.shape == (1,) + self._input_shape

        x = preprocess_input(x)

        return x

    def _model_handler(self, output_feature_layer):
        """
        Used to construct and layout the VGG16 model with desired input and output

        Arguments: 
            - output_feature_layer: The layer from VGG16 to output from the forward pass
                of the model

        Returns:
            - A Keras model type with desired input and output
        """ 

        # allow for layer feature selection, but defaulting to output of conv-layers to make
        # dataset agnostic -- intend for SVM to be run after
        try:
            model = Model(inputs=self._base_model.input, outputs=\
                    self._base_model.get_layer(output_feature_layer).output)
        except AttributeError:
            warnings.warn('Improper layer output selected, defaulting to \'fc1\' layer features.')
            model = Model(inputs=self._base_model.input,
                    outputs=self._base_model.get_layer('fc1').output)

        return model


    def _imagenet_predictions(self, data):
        """
        Outputs the predicted imagenet class of the input data
        
        Arguments:
            - data: file location or numpy array containing image data to feed into the network

        Returns:
            - predictions: The top 5 predicted imagenet classes at the output of the model
        """

        model = self._model_handler('predictions')
        x = self._input_handler(data, self._interpolation_method)
        features = model.predict(x)
        predictions = decode_predictions(features)

        return predictions

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        with stopit.ThreadingTimeout(timeout) as timer:

            model = self._model_handler(self._output_feature_layer)
            features = []

            nimage = inputs.shape[0]

            for i in range(nimage):
                datum = np.rollaxis(inputs[i,:,:,:], 2)
                x = self._input_handler(datum, self._interpolation_method)
                image_features = model.predict(x)
                image_features = np.squeeze(image_features)
                features.append(image_features)

        return base.CallResult(container.ndarray(np.asarray(features)))

        if timer.state != timer.EXECUTED:
            raise TimeoutError('VGG16 produce timed out.')

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: VGG16Hyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

