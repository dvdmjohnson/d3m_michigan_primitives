import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, supervised_learning
from d3m import container, utils as d3m_utils
import collections
import os
import warnings
import pickle
import stopit
import copy
import numpy as np

import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from spider.supervised_learning.goturn.utils import GoTurn as GoTurnModel
from spider.supervised_learning.goturn.utils import BoundingBox, CropPadImage, crop_image, convert_state_dict_keys

__all__ = ('GoTurn')

Inputs = container.DataFrame
Outputs = container.ndarray

class GoTurnParams(params.Params):
    fitted: bool

class GoTurnHyperparams(hyperparams.Hyperparams):
    num_gpus = hyperparams.Bounded[int](lower=0,
        upper=None,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description='Number of gpus (not gpu cores) this process is allowed to use.')
    num_epochs = hyperparams.Bounded[int](lower=0,
        upper=None,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Number of epochs to run')
    learning_rate = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=1e-5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Learning rate for Stochastic Gradient Descent Optimizer')
    momentum = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=0.9,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Momentum for Stochastic Gradient Descent Optimizer')
    weight_decay = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=0.0005,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Regularizer for Stochastic Gradient Descent Optimizer')

class GoTurn(supervised_learning.SupervisedLearnerPrimitiveBase[Inputs, Outputs, GoTurnParams, GoTurnHyperparams]):

    """
    GoTurn visual object tracking training
    """
    
    metadata= metadata_module.PrimitiveMetadata({
        'id': '915dcd56-d2ee-4c4c-9c7f-c6ed056fe905',
        'version': '0.0.5',
        'name': 'GoTurn',
        'description': """Siamese-style network architecture that performs generic object tracking by regressing to four bounding box coordinates.""",
        'keywords' : ['video', 'deep learning', 'visual tracking', 'tracking', 'alex net'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/supervised_learning/goturn/goturn.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@inproceedings{held2016learning,
                  title={Learning to track at 100 fps with deep regression networks},
                  author={Held, David and Thrun, Sebastian and Savarese, Silvio},
                  booktitle={European Conference on Computer Vision},
                  pages={749--765},
                  year={2016},
                  organization={Springer}}"""
        },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                'package': 'ffmpeg',
                'version': '7:2.8.11-0ubuntu0.16.04.1'},
            {'type': metadata_module.PrimitiveInstallationType.FILE,
            'key': 'goturn_weights.pth',
            'file_uri': 'https://umich.box.com/shared/static/lbx9uo2cvruamhey0clcit0w7tufis8w.pth',
            'file_digest': '57aed821983b6b083221543a289893c7e5464d7031a0c2ef6f4574a516b652f6'}],
        'python_path': 'd3m.primitives.learner.goturn.Umich',
        'hyperparams_to_tune': ['num_epochs'],
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK],
        'primitive_family': metadata_module.PrimitiveFamily.LEARNER
    })

    def __init__(self, *, hyperparams: GoTurnHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None, volumes: typing.Dict[str,str] = None) -> None:
        """
        GoTurn visual object tracking training
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers, volumes=volumes)

        self._num_gpus = hyperparams['num_gpus']
        self._num_epochs = hyperparams['num_epochs']
        self._lr = hyperparams['learning_rate']
        self._momentum = hyperparams['momentum']
        self._weight_decay = hyperparams['weight_decay']
        self._input_shape = (3,227,227)
        self._scale = 227/10

        self._use_gpu = self._num_gpus>0
        self._model_name = 'goturn'

        self._model = GoTurnModel(train=True, use_gpu=self._use_gpu)
        self._weights_path = volumes['goturn_weights.pth']

        self._criterion = nn.L1Loss()
        self._optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self._model.parameters()), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)

        #Load weights
        weights_state_dict = torch.load(self._weights_path, map_location=lambda storage, location: storage)
        weights_state_dict = convert_state_dict_keys(weights_state_dict)

        #Flexible state dict loading
        state = self._model.state_dict()
        state.update(weights_state_dict)
        self._model.load_state_dict(state)

        try:
            if self._use_gpu:
                self._model = self._model.cuda()
        except:
            self._use_gpu = False
            self._num_gpus = 0

        #Training data
        self._X0: Inputs = None #Expected shape: (num_videos, sequence_length-1, 3, height, width)
        self._X1: Inputs = None #Expected shape: (num_videos, sequence_length-1, 3, height, width)
        self._y_init : InputLabels = None #Excepted shape: (4,)
        self._y: InputLabels = None #Expected shape: (sequence_length-1, 4)
        
        self._fitted: bool = False 

    def set_training_data(self, *, inputs: Inputs, targets: Outputs) -> None:
        #Read in video from DataFrame object
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

        video_data = inputs.iloc[:,column_to_use]

        self._X0 = video_data[0][:-1] #previous frames
        self._X1 = video_data[0][1:] #next frames
        self._y_init = targets[0] #initialize tracker from first frame
        self._y = targets[1:] #next frame coordinates

        self._fitted = False

    def _loss(self, predictions, targets):
        """
        Args:
            :predictions: predicted bounding box coordinates
            :targets: ground truth coordinates
        Return:
            Loss based on criterion defined in _init
        """
        return self._criterion(predictions, targets)

    def _create_bbox_object(self, annotation):
        bbox = BoundingBox()

        #Zero-index all coordinates
        bbox.x1 = annotation[0]-1
        bbox.y1 = annotation[1]-1
        bbox.x2 = annotation[2]-1
        bbox.y2 = annotation[2]-1

        return bbox

    def _preprocess(self, x0, x1, y0, y1):
        """
        Pre-processes data
        Args:
            - data: string file location of video or np.array of video data 
        Returns:
            - x0: Tensor of shape (1, 3, 227, 227) 
            - x1: Tensor of shape (1, 3, 227, 227) 
            - y1: Tensor of shape (1, 4)
        """
        y0_bbox = self._create_bbox_object(y0)
        y1_bbox = self._create_bbox_object(y1)
        unloader = transforms.ToPILImage()

        x0 = np.moveaxis(x0,2,1)
        x1 = np.moveaxis(x1,2,1)

        #Crop to search region (x times bounding box region) - defined as kContextFactor in utils
        search_region, bbox_search_region, edge_spacing_x, edge_spacing_y = CropPadImage(y0_bbox, unloader(torch.Tensor(x0))) 
        x0 = crop_image(search_region) #Get cropped image (as tensor)
        y0_bbox.recenter(bbox_search_region, edge_spacing_x, edge_spacing_y)
        y0_bbox.scale(search_region, 227)

        search_region, bbox_search_region, edge_spacing_x, edge_spacing_y = CropPadImage(y0_bbox, unloader(torch.Tensor(x1))) 
        x1 = crop_image(search_region) #Get cropped image (as tensor)
        y1_bbox.recenter(bbox_search_region, edge_spacing_x, edge_spacing_y)
        y1_bbox.scale(search_region, 227)

        #Bound values to within current window
        y1_bbox.x1 = max(0,y1_bbox.x1)
        y1_bbox.y1 = max(0,y1_bbox.y1)
        y1_bbox.x2 = min(227,y1_bbox.x2)
        y1_bbox.y2 = min(227,y1_bbox.y2)

        if len(x0.shape) == 3:
            x0 = x0.unsqueeze(0)
            x1 = x1.unsqueeze(0)

        #Switch channels to BGR instead of RGB (Per Caffe implementation)
        x0 = x0[:, [2,1,0],:,:]*255 #Pixel values between 0-255
        x1 = x1[:, [2,1,0],:,:]*255 #Pixel values between 0-255
        mean = torch.FloatTensor([[[103.939]],[[116.779]],[[123.68]]])#ImageNet mean values

        y1 = torch.Tensor(y1_bbox.getVector())
        y1.unsqueeze(0)
        if self._use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
            y1 = y1.cuda()
            mean = mean.cuda()

        x0 = x0 - mean
        x1 = x1 - mean 

        return x0, x1, y1

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:

        if self._fitted:
            return base.CallResult(None)

        if not hasattr(self, '_X0') or not hasattr(self, '_X1') or not hasattr(self, '_y'):
            raise ValueError('Missing training data')

        if(not (isinstance(self._X0, container.ndarray) and isinstance(self._X1, container.ndarray) and isinstance(self._y, container.ndarray))):
            raise TypeError('Training inputs and outputs must be D3M numpy arrays')

        #Fit with timeout
        with stopit.ThreadingTimeout(timeout) as timer:
            self._model.train() #Set model to training mode

            for _ in range(self._num_epochs):
                y0 = self._y_init
                for x0,x1,y1 in zip(self._X0, self._X1, self._y):
                    prev_image, curr_image, curr_bbox = self._preprocess(x0,x1,y0,y1)
                    
                    #Perform prediction with model
                    prediction = self._model(Variable(curr_image), Variable(prev_image))
                    
                    self._optimizer.zero_grad()
                    loss = self._loss(prediction, Variable(curr_bbox))
                    loss.backward()
                    self._optimizer.step()
                    
                    y0 = y1

            self._fitted = True

        #If completed on time return, else reset state and raise error
        if timer.state == timer.EXECUTED:
            return base.CallResult(None)
        else:
            raise TimeoutError('GoTurn fit timed out')

    @base.singleton 
    def produce(self, *, inputs: Inputs, targets: Outputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        if self._fitted is False:
            raise ValueError('Calling produce before fitting')

        #Read in video from DataFrame object
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

        video_data = inputs.iloc[:,column_to_use]

        X0 = video_data[0][:-1] #previous frames
        X1 = video_data[0][1:] #next frames
        y_init = targets[0] #initialize tracker from first frame
        y = targets[1:] #next frame coordinates

        with stopit.ThreadingTimeout(timeout) as timer:
            coordinate_predictions = container.List()
            self._model.eval() #Set model to eval mode

            y0 = y_init
            for x0,x1,y1 in zip(X0, X1, y):
                prev_image, curr_image, curr_bbox = self._preprocess(x0,x1,y0,y1)
                
                #Perform prediction with model
                prediction = self._model(Variable(curr_image), Variable(prev_image))
                coordinate_predictions.append(prediction[0].data.cpu().numpy())
                
                y0 = y1

        return base.CallResult(np.array(coordinate_predictions))

        if timer.state != timer.EXECUTED:
            raise TimeoutError('GoTurn produce timed out')
	
    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, targets: Outputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, reference=reference)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, targets: Outputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, left=left, right=right)


    #package up our (potentially) fitted params for external inspection and storage
    def get_params(self) -> GoTurnParams:
        return GoTurnParams(fitted=self._fitted)

    #use an externally-stored set of params to set the state of our primitive
    def set_params(self, *, params: GoTurnParams) -> None:
        self._fitted = params['fitted']

    #package the full state of the primitive (including hyperparameters and random state)
    #as a serializable dict
    def __getstate__(self) -> dict:
        return {
            'constructor': {
                'hyperparams': self.hyperparams,
                'random_seed': self.random_seed,
                'docker_containers': self.docker_containers,
            },
            'params': self.get_params(),
            'random_state': self._random_state,
        }

    #restores the full state of the primitive (as stored by __getstate__())
    def __setstate__(self, state: dict) -> None:
        self.__init__(**state['constructor'])  # type: ignore
        self.set_params(params=state['params'])
        self._random_state = state['random_state']
        self._rf.random_state = self._random_state #we need to reset this reference in case the sklearns object was pickled
  
    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: GoTurnHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
