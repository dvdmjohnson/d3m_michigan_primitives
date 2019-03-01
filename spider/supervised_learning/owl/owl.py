import typing
import os
import stopit
import numpy as np

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, supervised_learning

from spider.supervised_learning.utils import proxOWL, costOWL, proxLASSO
#from ..utils import proxOWL, costOWL, proxLASSO

__all__ = ('OWLRegression',)

#see https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/container for valid types
Inputs = container.ndarray
Outputs = container.ndarray


class OWLParams(params.Params):
    fitted: bool
    coef: container.ndarray
    intercept: float


class OWLHyperparams(hyperparams.Hyperparams):
    """
    Weight in OWL norm should be non-increasing:
                    ^
                    |
        max_val:    |-----+
                    |     . \   type: linear
                    |     .   \
                    |     .     \
        min_val:    | . . . . . . +--------
                    |     .       .
                    ------------------------>
                       max_off   min_off
    """

    weight_type = hyperparams.Enumeration[str](values = ["linear"],
        default = "linear",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description= "type of hyperparameters. future: quadratic, logarithm")
    weight_max_val = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=0.01,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="should be greater than weight_min_val")
    weight_max_off = hyperparams.Bounded[int](lower=0,
        upper=None,
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="should be smaller than weight_min_off and n")
    weight_min_val = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=0.01,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="should be no greater than weight_max_val (or equal to, when the two cutoff values are the same)")
    weight_min_off = hyperparams.Bounded[int](lower=0,
        upper=None,
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="should be greater than or equal to weight_max_off and smaller than n")
    fit_intercept = hyperparams.UniformBool(default = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description= "Whether the intercept should be estimated")
    normalize = hyperparams.UniformBool(default = False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description= "Whether the features should be centerized and standardized. Ignored when fit_intercept is set to False")
    tol = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=1e-3,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="tolerance to exit iterations")
    learning_rate = hyperparams.Bounded[float](lower=0,
        upper=None,
        default=1e-3,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="learning rate of proximal gradient descent (FISTA)")
    verbose = hyperparams.Bounded[int](lower=0,
        upper=2,
        default = 1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description= "0: no training output. 1: exit status. 2: every iteration")


class OWLRegression(supervised_learning.SupervisedLearnerPrimitiveBase[Inputs, Outputs, OWLParams, OWLHyperparams]):
    """
    Primitive for linear regression with OWL regularization.
    """

    #your metadata object must contain all required metadata fields, or your primitive won't validate
    #(and thus won't run.)
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'fab94f96-74f4-4987-ab45-2a992cb37e0b',
        'version': "0.0.5",
        'name': 'OWLRegression',
        'description': """Solve linear regression problem with ordered weighted L1 (OWL) regularization""",
        'keywords': ['regularization', 'regression', 'supervised', 'ordered weight L1', 'proximal gradient descent'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:alsoltan@umich.edu',
            'uris': [
                #link to file and repo
                'https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/supervised_learning/owl/owl.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
            'citation': """@ARTICLE{2014arXiv1407.3824B,
                author = {{Bogdan}, M. and {van den Berg}, E. and {Sabatti}, C. and {Su}, W. and 
                {Cand{\`e}s}, E.~J.},
                title = "{SLOPE - Adaptive variable selection via convex optimization}",
                journal = {ArXiv e-prints},
                archivePrefix = "arXiv",
                eprint = {1407.3824},
                primaryClass = "stat.ME",
                keywords = {Statistics - Methodology},
                year = 2014,
                month = jul,
                adsurl = {http://adsabs.harvard.edu/abs/2014arXiv1407.3824B},
                adsnote = {Provided by the SAO/NASA Astrophysics Data System}
                } """
            },
        #link to install package
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
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        #entry-points path to the primitive (see setup.py)
        'python_path': 'd3m.primitives.regression.owl_regression.umich',
        #a list of hyperparameters that are high-priorities for tuning
        'hyperparams_to_tune': ['weight_type', 'weight_max_val', 'weight_max_off', 'weight_min_val', 'weight_min_off', 'fit_intercept', 'normalize', 'tol', 'learning_rate', 'verbose'],
        #search https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json for list of valid algorithm types
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.LINEAR_REGRESSION],
        #again search https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json for list of valid primitive families
        'primitive_family': metadata_module.PrimitiveFamily.REGRESSION
        })

    def __init__(self, *, hyperparams: OWLHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:

        #random_seed and docker_containers inputs should be handled by the superclass
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        # Hyperparameters
        self._tol: float = self.hyperparams['tol']
        self._learning_rate: float = self.hyperparams['learning_rate']
        self._verbose: int = self.hyperparams['verbose']
        self._weight: container.ndarray = []
        self._fit_intercept: bool = self.hyperparams['fit_intercept']
        self._normalize: bool = self.hyperparams['normalize']
        weight_type = self.hyperparams['weight_type']
        max_val = self.hyperparams['weight_max_val']
        max_off = self.hyperparams['weight_max_off']
        min_val = self.hyperparams['weight_min_val']
        min_off = self.hyperparams['weight_min_off']
        assert max_off <= min_off, "max_off={} should be no greater than min_off={}".format(max_off, min_off)
        if max_off == min_off:
            assert max_val == min_val, "when max_off==min_off, max_val and min_val have to be identical"
        assert max_val >= min_val, "max_val={} should be no less than min_val={}".format(max_val, min_val)

        # Training data
        self._X: Inputs = None
        self._y: InputLabels = None

        # Status
        self._n_iter = 0  # number of iterations done in fit
        self._fitted: bool = False
        self._loss_history: container.ndarray = []
        self._random_state = np.random.RandomState(random_seed)


    #training data is set here, rather than in the constructor.  For most methods, outputs will be of type Outputs - 
    #PairwiseDistanceLearners are an oddity in that regard because their training data is in a different format
    #from their output
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Arguments:
            inputs: nSamples x nFeatures
            outputs: nSamples, or nSamples x 1, ... (All dim after the first dim should have length 1)
        """
        # Assertions
        assert inputs.ndim == 2, "X should be 2-dimensional array"
        assert inputs.shape[0] == outputs.shape[0], "shapes of X and y don't match"
        assert outputs.shape[0] == outputs.flatten().shape[0], "All dims of y, except the 1st one, should be 1"
        assert inputs.shape[0] > 0, "empty input"

        # Set the features
        self._X = inputs.astype(float)
        self._y = outputs.astype(float)
        self._fitted = False

        # Set OWL weight
        weight_type = self.hyperparams['weight_type']
        max_val = self.hyperparams['weight_max_val']
        max_off = self.hyperparams['weight_max_off']
        min_val = self.hyperparams['weight_min_val']
        min_off = self.hyperparams['weight_min_off']
        assert min_off < self._X.shape[1], "min_off={} should be smaller than the number of features={}".format(min_off, self._X.shape[1])
        weight = np.zeros(self._X.shape[1])
        weight[:max_off] = max_val
        weight[min_off:] = min_val
        if weight_type == 'linear':
            weight[max_off:min_off+1] = np.linspace(max_val, min_val, min_off-max_off+1)
        self._weight = weight


    @staticmethod
    def _preprocessing(X, y, fit_intercept, normalize):
        """
        Static preprocessing method to centerize and normalize the data on demand.
        Note if fit_intercept is False, then normalize is ignored.
        """
        if fit_intercept:
            X_offset = np.mean(X, axis=0)
            X -= X_offset
            if normalize:
                X_scale = np.linalg.norm(X, axis=0)
                X = X / X_scale
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = np.mean(y, axis=0)
            y -= y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
            y_offset = X.dtype.type(0)

        return X, y, X_offset, y_offset, X_scale


    #timeout MUST be implemented on final primitive submissions
    #see https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/primitive_interfaces/base.py for details on CallResult

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Fit the linear regression problem with OWL regularization
        """

        # State/input checking
        if self._fitted:
            return base.CallResult(None)

        if not hasattr(self, '_X') or not hasattr(self, '_y'):
            raise ValueError("Missing training data.")

        if not (isinstance(self._X, container.ndarray) and isinstance(self._y, container.ndarray)):
            raise TypeError('Training inputs and outputs must be D3M numpy arrays.')

        # Preprocessing
        self._X, self._y, X_offset, y_offset, X_scale = self._preprocessing(
                self._X, self._y, self._fit_intercept, self._normalize)

        # Store state in case of timeout
        if hasattr(self, '_coef'):
            coef = self._coef
            intercept = self._intercept
        else:
            coef = None

        # Do fitting with timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            self._coef = np.zeros(self._X.shape[1], dtype=float)
            assert self._X.shape[0] > 0
            assert self._X.shape[0] == self._y.shape[0], "shapes of input X and y don't match"

            # FISTA with proxOWL as proximal operator
            if iterations is None:
                iterations = float('inf')

            n_iter = 0
            loss_history = []
            loss = lambda beta: costOWL(beta, self._X, self._y, self._weight)
            coef_old = np.zeros(self._X.shape[1])
            z = coef_old.copy()
            t_old = 1

            if self._verbose >= 1:
                print("FISTA training begins:")
            eps = np.nan
            while True:
                if n_iter >= iterations:
                    break

                _, grad_fit = loss(z)
                z = z - self._learning_rate * grad_fit
                coef = proxOWL(z, self._learning_rate * self._weight)
                t = (1 + np.sqrt(1+4*t_old**2)) / 2
                z = coef + ((t_old-1)/t) * (coef - coef_old)

                eps = np.linalg.norm(coef - coef_old) / (np.linalg.norm(coef_old) + 1e-10)
                if eps < self._tol:
                    break

                coef_old = coef
                t_old = t

                cur_loss, _ = loss(coef)
                loss_history.append(cur_loss)
                n_iter += 1

                if self._verbose >= 2 and n_iter % max(1, iterations//20) == 0:
                    print("iter = {:4d}, loss = {:f}".format(n_iter, loss_history[-1]))
            if self._verbose >= 1:
                print("FISTA training exits after {} iterations, with loss={:f} and eps={:f}".format(
                    n_iter, loss_history[-1], eps))

            self._loss_history = loss_history
            self._n_iter = n_iter
            self._coef = coef / X_scale
            if self._fit_intercept:
                self._intercept = float(y_offset - X_offset.dot(self._coef))
            else:
                self._intercept = float(0)
            self._fitted = True

        # If we completed on time, return.  Otherwise reset state and raise error.
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return base.CallResult(None)
        else:
            self._coef = coef
            self._intercept = intercept
            self._fitted = False
            raise TimeoutError("OWL fitting timed out.")


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Compute the predictions given inputs with shape n by m,
        yielding an array of size n.

        Inputs must match the dimensionality of the training data.
        """
        # First do assorted error checking and initialization
        if self._fitted is False:
            raise ValueError("Calling produce before fitting.")

        if(inputs.shape[1] != self._coef.shape[0]):
            raise ValueError('Input dimension is wrong.')

        # Start timeout counter.
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            outputs: container.ndarray = container.ndarray(
                   inputs.dot(self._coef) + self._intercept)
            outputs.metadata = inputs.metadata.clear(for_value=outputs, source=self)
            return base.CallResult(outputs)
        # If we did not finish in time, raise error.
        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("OWL regression produce timed out.")

    # Package up our (potentially) fitted params for external inspection and storage
    def get_params(self) -> OWLParams:
        return OWLParams(fitted=self._fitted, coef=self._coef, intercept=self._intercept)

    #use an externally-stored set of params to set the state of our primitive
    def set_params(self, *, params: OWLParams) -> None:
        self._fitted = params['fitted']
        self._coef = params['coef']
        self._intercept = params['intercept']

    # Package the full state of the primitive (including hyperparameters and random state)
    # as a serializable dict
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

    # Restores the full state of the primitive (as stored by __getstate__())
    def __setstate__(self, state: dict) -> None:
        self.__init__(**state['constructor'])  # type: ignore
        self.set_params(params=state['params'])
        self._random_state = state['random_state']

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: OWLHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
