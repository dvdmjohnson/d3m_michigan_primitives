import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, distance
from d3m import container, utils
import collections
import os
import warnings
import pickle
import stopit
import copy
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from spider.distance.utils import get_random_constraints

__all__ = ('RFD',)

#see https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/container for valid types
Inputs = container.ndarray
InputLabels = container.ndarray
Outputs = container.ndarray

#contains variables that are learned, optimized or otherwise altered during the fitting process
#used to save state information (in serializable form) such that a particular instance of the primitive
#can be stored and recreated later (if also supplied with the same hyperparams)
#See https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/params.py
class RFDParams(params.Params):
    fitted: bool
    d: int
    rf: ExtraTreesRegressor

#describes the hyperparameters of the primitive (i.e. input parameters that affect the primitive's behavior and do not
#change during fitting or producing operations).  See https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/hyperparams.py
class RFDHyperparams(hyperparams.Hyperparams):
    class_cons = hyperparams.Bounded[int](lower=10,
        upper=None,
        default=1000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="the number of pairwise constraints per class to sample from the training labels")
    num_trees = hyperparams.Bounded[int](lower=10,
        upper=2000,
        default=500,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter', 'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="the number of trees the metric forest should contain")
    min_node_size = hyperparams.Bounded[int](lower=1,
        upper=50,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="the stopping criterion for tree splitting")
    n_jobs = hyperparams.Union(configuration=collections.OrderedDict({
            'enum':hyperparams.Enumeration[int](values=[-1], default=-1),
            'bounded':hyperparams.Bounded[int](lower=1, upper=128, default=1)}),
        default='enum',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="the number of separate processes to run; if -1, will be set equal to number of cores.")


#note that private variables (i.e class members not defined in the primitive interfaces) are prefaced with "_"
#also note that hyperparameters are referenced using their name as a key value input to the hyperparams object
#the * argument is used here (and on any other methods that takes inputs) to force callers to use keyword inputs
class RFD(distance.PairwiseDistanceLearnerPrimitiveBase[Inputs, InputLabels, Outputs, RFDParams, RFDHyperparams]):

    """
    Primitive for learning Random Forest Distance metrics and returning a pairwise
    distance matrix between two sets of data.
    """

    #your metadata object must contain all required metadata fields, or your primitive won't validate
    #(and thus won't run.)
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '273b03a2-5c9e-4a2c-93f2-18d293f2994d',
        'version': "0.0.5",
        'name': 'RFD',
        'description': """Learns Random Forest Distance metrics based on pairwise constraints drawn from labeled vector
                           data and applies the learned metric to produce a distance matrix between two given sets of
                           instances (the same set can be given twice to compute distance between each pair in a single
                           dataset).""",
        'keywords': ['distance', 'metric learning', 'weakly supervised', 'decision tree', 'random forest'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/distance/rfd/rfd.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@inproceedings{xiong2012random, 
                title={Random forests for metric learning with implicit pairwise position dependence},
                author={Xiong, Caiming and Johnson, David and Xu, Ran and Corso, Jason J},
                booktitle={Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining},
                pages={958--966},
                year={2012},
                organization={ACM}"""
            },
        #link to install package
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        #entry-points path to the primitive (see setup.py)
        'python_path': 'd3m.primitives.similarity_modeling.rfd.Umich',
        #a list of hyperparameters that are high-priorities for tuning
        'hyperparams_to_tune': ['class_cons', 'num_trees'],
        #search https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json for list of valid algorithm types
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.RANDOM_FOREST,
            metadata_module.PrimitiveAlgorithmType.ENSEMBLE_LEARNING],
        #again search https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json for list of valid primitive families
        'primitive_family': metadata_module.PrimitiveFamily.SIMILARITY_MODELING
        })

    def __init__(self, *, hyperparams: RFDHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        """
        Primitive for learning Random Forest Distance metrics and returning a pairwise
        distance matrix between two sets of data.
        """
        #random_seed and docker_containers inputs should be handled by the superclass
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._fitted: bool = False
        self._class_cons: int = hyperparams['class_cons']

        self._X: Inputs = None
        self._y: InputLabels = None
        self._random_state = np.random.RandomState(random_seed)


        self._rf = ExtraTreesRegressor(n_estimators=hyperparams['num_trees'],
                                        max_features="log2",
                                        min_samples_leaf=hyperparams['min_node_size'],
                                        n_jobs=hyperparams['n_jobs'],
                                        random_state = self._random_state)

    #training data is set here, rather than in the constructor.  For most methods, outputs will be of type Outputs - 
    #PairwiseDistanceLearners are an oddity in that regard because their training data is in a different format
    #from their output
    def set_training_data(self, *, inputs: Inputs, outputs: InputLabels) -> None:
        self._X = inputs
        self._y = outputs
        self._fitted = False

    #timeout MUST be implemented on final primitive submissions
    #see https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/primitive_interfaces/base.py for details on CallResult
    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Fit the random forest distance to a set of labeled data by sampling and fitting
        to pairwise constraints.
        """

        # state/input checking
        if self._fitted:
            return base.CallResult(None)

        if not hasattr(self, '_X') or not hasattr(self, '_y'):
            raise ValueError("Missing training data.")

        if(not (isinstance(self._X, container.ndarray) and isinstance(self._y, container.ndarray))):
            raise TypeError('Training inputs and outputs must be D3M numpy arrays.')

        # store state in case of timeout
        if hasattr(self, '_d'):
            dtemp = self._d
        else:
           dtemp = None
        rftemp = copy.deepcopy(self._rf)

        # do fitting with timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            n = self._X.shape[0]
            self._d = self._X.shape[1]
            assert n > 0
            assert self._d > 0
            assert n == self._y.shape[0]

            constraints = get_random_constraints(
                self._y, self._class_cons // 3, 2 * self._class_cons // 3, self._random_state)

            c1 = self._X[constraints[:, 0], :]
            c2 = self._X[constraints[:, 1], :]
            rfdfeat = np.empty(dtype=np.float32, shape=(constraints.shape[0], self._d * 2))
            rfdfeat[:, :self._d] = np.abs(c1 - c2)
            rfdfeat[:, self._d:] = (c1 + c2) / 2

            self._rf.fit(rfdfeat, constraints[:, 2])
            self._fitted = True

        # if we completed on time, return.  Otherwise reset state and raise error.
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return base.CallResult(None)
        else:
            self._d = dtemp
            self._rf = rftemp
            self._fitted = False
            raise TimeoutError("RFD fitting timed out.")

    #produce generally takes only one input.  Again, PairwiseDistanceLearners are an oddity.
    def produce(self, *, inputs: Inputs, second_inputs: Inputs, 
               timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Compute the distance matrix between vector arrays inputs and
        second_inputs, yielding an output of shape n by m (where n and m are
        the number of instances in inputs and second_inputs respectively).

        Both inputs must match the dimensionality of the training data.
        The same array may be input twice in order to generate an
        (inverted) kernel matrix for use in clustering, etc.
        """
        #first do assorted error checking and initialization
        if self._fitted is False:
            raise ValueError("Calling produce before fitting.")

        X = inputs
        Y = second_inputs

        if(X.shape[1] != self._d or Y.shape[1] != self._d):
            raise ValueError('At least one input has the wrong dimensionality.')

        n1 = X.shape[0]
        n2 = Y.shape[0]

        #start timeout counter
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            # compute distance from each instance in X to all instances in Y
            dist = np.empty(dtype=np.float32, shape=(n1, n2))
            for i in range(0, n1):
                data = np.empty(dtype=np.float32, shape=(n2, self._d * 2))
                data[:, :self._d] = np.abs(X[i, :] - Y)
                data[:, self._d:] = (X[i, :] + Y) / 2
                dist[i, :] = self._rf.predict(data)

            #Return distance.  Note that we "wrap" the numpy array in the appropriate D3M container class,
            #before further wrapping it in the CallResult.
            return base.CallResult(container.ndarray(dist, generate_metadata=True))
        # if we did not finish in time, raise error.
        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("RFD produce timed out.")

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, second_inputs=second_inputs)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, outputs: InputLabels, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, second_inputs=second_inputs, outputs = outputs)


    #package up our (potentially) fitted params for external inspection and storage
    def get_params(self) -> RFDParams:
        return RFDParams(fitted=self._fitted, d=self._d, rf=self._rf)

    #use an externally-stored set of params to set the state of our primitive
    def set_params(self, *, params: RFDParams) -> None:
        self._fitted = params['fitted']
        self._d = params['d']
        self._rf = params['rf']

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
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: RFDHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

