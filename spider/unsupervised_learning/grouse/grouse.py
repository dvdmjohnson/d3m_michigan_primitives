import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, transformer, unsupervised_learning
from d3m import container, utils
import os
import numpy as np

__all__ = ('GROUSE',)

Inputs = container.ndarray
Outputs = container.ndarray


class GROUSEHyperparams(hyperparams.Hyperparams):

    rank = hyperparams.Bounded[int](lower=1,
                                    upper=None,
                                    default=5,
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                    description="Rank of learned low-rank matrix")

    constant_step = hyperparams.Bounded[float](lower=0, upper=None, default=0, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                               description="Make nonzero for contant step size instead of multi-level adaptive step")

    max_train_cycles = hyperparams.Bounded[int](lower=1, upper=None, default=10, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                                description="Number of times to cycle over training data")
    subsample = hyperparams.Bounded[float](lower=0.01, upper=1, default=1, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                                description="Matrix sub-sampling parameter during training")

### GROUSE OPTIONS CLASS
class _OPTIONS(object):
    def __init__(self, rank, constant_step=0, subsample=1):
        self.rank = rank
        self.constant_step = constant_step
        self.subsample = subsample

class GROUSEParams(params.Params):
    OPTIONS: _OPTIONS
    U: np.ndarray


##  GROUSE class
#
#   uses GROUSE to perform online dimensionality reduction of (possibly) sub-sampled data
class GROUSE(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[
                        Inputs, Outputs, GROUSEParams, GROUSEHyperparams]):
    """
    Uses GROUSE to perform online dimensionality reduction of (possibly) sub-sampled data
    """
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'a5afb336-608b-4d04-bbab-e8921f605d04',
        'version': "0.0.5",
        'name': 'GROUSE',
        'description': """Performs online, unsupervised dimensionality reduction by computing PCA on the Grassmannian manifold with missing data.""",
        'keywords': ['unsupervised learning', 'dimensionality reduction', 'PCA', 'low-rank', 'online',
                     'streaming', 'Grassmannian manifold', 'subspace tracking', 'matrix completion', 'missing data'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                # link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/unsupervised_learning/grouse/grouse.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            #CHANGE THIS BELOW
            'citation': """@inproceedings{balzano2010online,
  title={Online identification and tracking of subspaces from highly incomplete information},
  author={Balzano, Laura and Nowak, Robert and Recht, Benjamin},
  booktitle={Communication, Control, and Computing (Allerton), 2010 48th Annual Allerton Conference on},
  pages={704--711},
  year={2010},
  organization={IEEE}
}"""
        },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
                 git_commit=utils.current_git_commit(os.path.dirname(__file__)))
             },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
             'package': 'ffmpeg',
             'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.data_compression.grouse.Umich',
        'hyperparams_to_tune': ['rank', 'constant_step','max_train_cycles'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_COMPRESSION
    })

    # GROUSE class constructor and instantiation
    def __init__(self, *, hyperparams: GROUSEHyperparams, random_seed: int = 0,
                 docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._rank = hyperparams['rank']

        self._constant_step = hyperparams['constant_step']
        self._max_train_cycles = hyperparams['max_train_cycles']
        self._subsample = hyperparams['subsample']

        self._X: Inputs = None
        self._U = None
        self._random_state = np.random.RandomState(random_seed)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._X = inputs
        self._dim = inputs.shape[1]
        self._training_size = inputs.shape[0]

    # GROUSE fit function: learns low-rank subspace from training data
    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:

        # Internal function to generate low-rank random matrix
        def generateLRMatrix(d, r):
            rando_mat = self._random_state.randn(d, d)
            Q, R = np.linalg.qr(rando_mat)
            U = Q[:, 0:r]
            return U

        assert self._X is not None, "No training data provided."
        assert self._X.ndim == 2, "Data is not in the right shape."
        assert self._rank <= self._X.shape[1], "Dim_subspaces should be less than ambient dimension."

        _X = self._X.T  # Get the training data

        # Begin training
        # Instantiate a random low-rank subspace
        d = _X.shape[0]
        r = self._rank
        U = generateLRMatrix(d, r)

        # Set the training control params
        self._grouseOPTIONS = _OPTIONS(self._rank, self._constant_step, self._subsample)

        U = self._train_grouse(_X, U)
        self._U = U  # update global variable

        return base.CallResult(None)

    # GROUSE training internal function
    def _train_grouse(self, X, U):

        max_cycles = self._max_train_cycles
        train_size = self._training_size
        for i in range(0, max_cycles):
            perm = self._random_state.choice(train_size, train_size, replace=False)  # randomly permute training data
            for j in range(0, train_size):
                _x = X[:, perm[j]]
                if(self._grouseOPTIONS.subsample < 1):
                    _xidx = self._random_state.choice(self._dim, int(np.ceil(self._grouseOPTIONS.subsample * self._dim)),replace=False)
                else:
                    _xidx = np.where(~np.isnan(_x))[0]

                U, _ = self._grouse_stream(U, _x, _xidx)
                self._U = U

        return U

    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:

        # Get the vector input, and the subspace
        _X = self._X.T  # Get the data
        # _Mask = self._Mask.T       #Get the mask indicating observed indices
        d, numVectors = _X.shape
        U = self._U

        for i in range(0, numVectors):
            _x = _X[:, i]

            if(self._grouseOPTIONS.subsample < 1):
                _xidx = self._random_state.choice(self._dim, int(np.ceil(self._grouseOPTIONS.subsample * self._dim)),replace=False)
            else:
                _xidx = np.where(~np.isnan(_x))[0]

            # Call GROUSE iteration
            U, w = self._grouse_stream(U, _x, _xidx)
            self._U = U

        return base.CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        U = self._U
        _X = inputs.T
        d,numVectors = _X.shape

        Y = np.zeros((d,numVectors))
        for i in range(0,numVectors):
            _x = _X[:, i]
            if(self._grouseOPTIONS.subsample < 1):
                _xidx = self._random_state.choice(self._dim, int(np.ceil(self._grouseOPTIONS.subsample * self._dim)),replace=False)
            else:
                _xidx = np.where(~np.isnan(_x))[0]

            # Call GROUSE iteration
            _, w = self._grouse_stream(U, _x, _xidx)
            Y[:,i] = U @ w

        return base.CallResult(container.ndarray(Y.T, generate_metadata=True))

    def produce_Subspace(self, *, inputs: Inputs, iterations: int = None) -> base.CallResult[
        Outputs]:
        X = inputs
        U = self._U

        return base.CallResult(container.ndarray(U, generate_metadata=True))


    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    ### MAIN GROUSE UPDATE FUNCTION
    def _grouse_stream(self, U, v, xIdx):


        ### Main GROUSE update

        step = self._grouseOPTIONS.constant_step
        
        U_Omega = U[xIdx,:]
        v_Omega = v[xIdx]
        w_hat = np.linalg.pinv(U_Omega)@v_Omega

        r = v_Omega - U_Omega @ w_hat

        rnorm = np.linalg.norm(r)
        wnorm = np.linalg.norm(w_hat)
        sigma = rnorm * np.linalg.norm(w_hat)

        if step > 0:
            t = step
        else:
            t = np.arctan(rnorm / wnorm)

        if t < np.pi / 2:
            alpha = (np.cos(t) - 1) / wnorm**2
            beta = np.sin(t) / sigma
            Ustep = U @ (alpha * w_hat)
            Ustep[xIdx] = Ustep[xIdx] + beta * r
            Uhat = U + np.outer(Ustep, w_hat)

        
        return Uhat, w_hat

    def get_params(self) -> GROUSEParams:
        return GROUSEParams(OPTIONS=self._grouseOPTIONS, U=self._U)

    def set_params(self, *, params: GROUSEParams) -> None:
        self._grouseOPTIONS = params['OPTIONS']
        self._U = params['U']

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

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state['constructor'])  # type: ignore
        self.set_params(params=state['params'])
        self._random_state = state['random_state']




