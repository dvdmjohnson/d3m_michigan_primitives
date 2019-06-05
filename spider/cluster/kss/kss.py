import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, clustering
from d3m import container, utils
import numpy as np
from scipy.linalg import orth
import os

Inputs = container.ndarray
Outputs = container.ndarray
DistanceMatrixOutput = container.ndarray

class KSSParams(params.Params):
    U: container.ndarray

class KSSHyperparams(hyperparams.Hyperparams):
    n_clusters = hyperparams.Bounded[int](lower=2,
        upper=None,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="number of subspaces/clusters to learn")
    dim_subspaces = hyperparams.Bounded[int](lower=1,
        upper=50,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="dimensionality of learned subspaces")

class KSS(clustering.ClusteringDistanceMatrixMixin[Inputs, Outputs, KSSParams, KSSHyperparams, DistanceMatrixOutput],
          clustering.ClusteringLearnerPrimitiveBase[Inputs, Outputs, KSSParams, KSSHyperparams]):

    metadata = metadata_module.PrimitiveMetadata({
        'id': '044e5c71-7507-4f58-a139-bc5481179d62',
        'version': "0.0.5",
        'name': 'KSS',
        'description': """Does clustering via the k-subspaces method.""",
        'keywords': ['clustering', 'k-subspaces', 'subspace'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/cluster/kss/kss.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@inproceedings{agarwal2004k, title={K-means projective clustering}, author={Agarwal, Pankaj K and Mustafa, Nabil H}, booktitle={Proceedings of the twenty-third ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems}, pages={155--165}, year={2004}, organization={ACM}}"""
            },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.clustering.kss.Umich',
        'hyperparams_to_tune': ['n_clusters', 'dim_subspaces'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.SUBSPACE_CLUSTERING],
        'primitive_family': metadata_module.PrimitiveFamily.CLUSTERING
        })

    def __init__(self, *, hyperparams: KSSHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._dim_subspaces = hyperparams['dim_subspaces']
        self._k = hyperparams['n_clusters']
        self._X: Inputs = None
        self._U = None
        self._random_state = np.random.RandomState(random_seed)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._X = inputs
        self._U = None

    def fit(self, *, iterations: int = None) -> base.CallResult[None]:
        assert self._X is not None, "No training data provided."
        assert self._X.ndim == 2, "Data is not in the right shape."
        assert self._dim_subspaces <= self._X.shape[1], "Dim_subspaces should be less than ambient dimension."

        _X = self._X.T
        n_features, n_samples = _X.shape

        # randomly initialize subspaces
        U_init = np.zeros((self._k, n_features, self._dim_subspaces))
        for kk in range(self._k):
            U_init[kk] = orth(self._random_state.randn(n_features, self._dim_subspaces))

        # compute residuals
        full_residuals = np.zeros((n_samples, self._k))
        for kk in range(self._k):
            tmp1 = np.dot(U_init[kk].T, _X)
            tmp2 = np.dot(U_init[kk], tmp1)
            full_residuals[:,kk] = np.linalg.norm(_X-tmp2, ord=2, axis=0)

        # label by nearest subspace
        estimated_labels = np.argmin(full_residuals, axis=1)

        # alternate between subspace estimation and assignment
        prev_labels = -1 * np.ones(estimated_labels.shape)
        it = 0
        while np.sum(estimated_labels != prev_labels) and (iterations is None or it < iterations):
            # first update residuals after labels obtained
            U = np.empty((self._k, n_features, self._dim_subspaces))
            for kk in range(self._k):
                Z = _X[:,estimated_labels == kk]
                D, V = np.linalg.eig(np.dot(Z, Z.T))
                D_idx = np.argsort(-D) # descending order
                U[kk] = V[:,D_idx[list(range(self._dim_subspaces))]]
                tmp1 = np.dot(U[kk,:].T, _X)
                tmp2 = np.dot(U[kk,:], tmp1)
                full_residuals[:,kk] = np.linalg.norm(_X-tmp2, ord=2, axis=0)
            # update prev_labels
            prev_labels = estimated_labels
            # label by nearest subspace
            estimated_labels = np.argmin(full_residuals, axis=1)

            it = it + 1

        self._U = U
        return base.CallResult(None)

    def produce(self, *, inputs: Inputs) -> base.CallResult[Outputs]:
        if self._U is None:
            raise ValueError("Calling produce before fitting.")
        full_residuals = np.empty((inputs.shape[0], self._k))
        for kk in range(self._k):
            tmp1 = np.dot(self._U[kk,:].T, inputs.T)
            tmp2 = np.dot(self._U[kk,:], tmp1)
            full_residuals[:,kk] = np.linalg.norm(inputs.T-tmp2, ord=2, axis=0)
        labels = np.argmin(full_residuals, axis=1)

        return base.CallResult(Outputs(labels))

    def produce_distance_matrix(self, *, inputs: Inputs) -> base.CallResult[DistanceMatrixOutput]:
        """
            Returns a generic result representing the cluster assignment labels in distance matrix form (i.e. distance is 0
            if the two instances are in the same class, and 1 if they are not).
        """
        if self._U is None:
            raise ValueError("Calling produce before fitting.")

        full_residuals = np.empty((inputs.shape[0], self._k))
        for kk in range(self._k):
            tmp1 = np.dot(self._U[kk,:].T, inputs.T)
            tmp2 = np.dot(self._U[kk,:], tmp1)
            full_residuals[:,kk] = np.linalg.norm(inputs.T-tmp2, ord=2, axis=0)
        labels = np.argmin(full_residuals, axis=1)

        n = labels.shape[0]
        labmat = np.empty((n,n))
        for i in range(0,n):
            labmat[i,:] = labels != labels[i]

        return base.CallResult(DistanceMatrixOutput(labmat))

    def get_params(self) -> KSSParams:
        return KSSParams(U = self._U)

    def set_params(self, *, params: KSSParams) -> None:
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
        
    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: KSSHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
