import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, clustering
from d3m import container, utils
import collections
import numpy as np
from scipy.linalg import orth
from sklearn.cluster import SpectralClustering
import stopit
import os

from ..kss import KSS, KSSHyperparams

Inputs = container.ndarray
Outputs = container.ndarray
DistanceMatrixOutput = container.ndarray

class EKSSHyperparams(hyperparams.Hyperparams):
    n_clusters = hyperparams.Bounded[int](lower=2,
        upper=None,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="number of clusters to learn")
    dim_subspaces = hyperparams.Bounded[int](lower=1,
        upper=50,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="dimensionality of learned subspaces")
    n_base = hyperparams.Bounded[int](lower=10,
        upper=1000,
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter', 'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="number of 'base' KSS clusterings to use in the ensemble - larger values generally yield better results but longer computation time")
    thresh = hyperparams.Union(configuration=collections.OrderedDict({
            'enum':hyperparams.Enumeration[int](values=[-1], default=-1),
            'bounded':hyperparams.Bounded[int](lower=1, upper=10000, default=5)}),
        default='bounded',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="if >=1, only the top <thresh> values from each column/row of the affinity matrix are used in spectral clustering")

class EKSS(clustering.ClusteringDistanceMatrixMixin[Inputs, Outputs, type(None), EKSSHyperparams, DistanceMatrixOutput],
          clustering.ClusteringTransformerPrimitiveBase[Inputs, Outputs, EKSSHyperparams]):

    metadata = metadata_module.PrimitiveMetadata({
        'id': '6d94cfb0-4225-4446-b5b1-afd8803f2bf5',
        'version': "0.0.5",
        'name': 'EKSS',
        'description': """Does clustering via the ensemble k-subspaces method.""",
        'keywords': ['clustering', 'k-subspaces', 'subspace', 'ensemble'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/cluster/ekss/ekss.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@article{DBLP:journals/corr/abs-1709-04744, author = {John Lipor and David Hong and Dejiao Zhang and Laura Balzano}, title = {Subspace Clustering using Ensembles of {\textdollar}K{\textdollar}-Subspaces}, journal = {CoRR}, volume = {abs/1709.04744}, year = {2017}, url = {http://arxiv.org/abs/1709.04744}, archivePrefix = {arXiv}, eprint = {1709.04744}, timestamp = {Thu, 05 Oct 2017 09:43:01 +0200}, biburl = {https://dblp.org/rec/bib/journals/corr/abs-1709-04744}, bibsource = {dblp computer science bibliography, https://dblp.org}
}"""
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
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.clustering.ekss.Umich',
        'hyperparams_to_tune': ['n_clusters', 'dim_subspaces'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.SUBSPACE_CLUSTERING,
            metadata_module.PrimitiveAlgorithmType.ENSEMBLE_LEARNING],
        'primitive_family': metadata_module.PrimitiveFamily.CLUSTERING
        })

    def __init__(self, *, hyperparams: EKSSHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._dim_subspaces = hyperparams['dim_subspaces']
        self._n_base = hyperparams['n_base']
        self._thresh = hyperparams['thresh']
        self._k = hyperparams['n_clusters']
        self._random_state = np.random.RandomState(random_seed)

    def set_training_data(self, *, inputs: Inputs) -> None:
        pass

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        assert inputs is not None, "No training data provided."
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert self._dim_subspaces <= inputs.shape[1], "Dim_subspaces should be less than ambient dimension"
        assert self._thresh <= inputs.shape[0], "Threshold should be in range 1:n_samples"

        _X = inputs.T
        n_features, n_samples = _X.shape

        affinity_matrix = np.zeros((n_samples, n_samples))

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            # for each base clustering
            for b in range(self._n_base):
                # run K-Subspaces
                ksshp = KSSHyperparams(n_clusters = self._k, dim_subspaces = self._dim_subspaces)
                kss = KSS(hyperparams=ksshp, random_seed=self._random_state.randint(0, 2**32-1))
                kss.set_training_data(inputs=_X.T)
                kss.fit(iterations=iterations)
                est_labels = kss.produce(inputs=_X.T).value
                # update affinity matrix
                for i in range(n_samples):
                    affinity_matrix[i][i] += 1
                    for j in range(i+1, n_samples):
                        if est_labels[i] == est_labels[j]:
                            affinity_matrix[i][j] += 1
                            affinity_matrix[j][i] += 1

            affinity_matrix = 1.0 * affinity_matrix / self._n_base

            # if thresh is positive, threshold affinity_matrix
            if self._thresh > 0:
                A_row = np.copy(affinity_matrix)
                A_col = np.copy(affinity_matrix.T)
                for i in range(n_samples):
                    # threshold rows
                    idx = np.argsort(A_row[i])[list(range(self._thresh))]
                    A_row[i][idx] = 0
                    # threshold columns
                    idx = np.argsort(A_col[i])[list(range(self._thresh))]
                    A_col[i][idx] = 0
                # average
                affinity_matrix = (A_row + A_col.T) / 2.0

            # apply Spectral Clustering with affinity_matrix
            sc = SpectralClustering(n_clusters= self._k, affinity='precomputed', random_state=self._random_state)
            estimated_labels = sc.fit_predict(affinity_matrix)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return base.CallResult(Outputs(estimated_labels))
        else:
            affinity_matrix = np.zeros((n_samples, n_samples))
            raise TimeoutError("EKSS fitting has timed out.")

    def produce_distance_matrix(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[DistanceMatrixOutput]:
        """
            Returns the affinity matrix generated from the ensemble of KSS clustering results.
        """
        assert inputs is not None, "No training data provided."
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert self._dim_subspaces <= inputs.shape[1], "Dim_subspaces should be less than ambient dimension"
        assert self._thresh <= inputs.shape[0], "Threshold should be in range 1:n_samples"

        _X = inputs.T
        n_features, n_samples = _X.shape

        affinity_matrix = np.zeros((n_samples, n_samples))

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            # for each base clustering
            for b in range(self._n_base):
                # run K-Subspaces
                ksshp = KSSHyperparams(n_clusters = self._k, dim_subspaces = self._dim_subspaces)
                kss = KSS(hyperparams=ksshp, random_seed=self._random_state.randint(0, 2**32-1))
                kss.set_training_data(inputs=_X.T)
                kss.fit(iterations=iterations)
                est_labels = kss.produce(inputs=_X.T).value
                # update affinity matrix
                for i in range(n_samples):
                    affinity_matrix[i][i] += 1
                    for j in range(i+1, n_samples):
                        if est_labels[i] == est_labels[j]:
                            affinity_matrix[i][j] += 1
                            affinity_matrix[j][i] += 1

            affinity_matrix = 1.0 * affinity_matrix / self._n_base

            # if thresh is positive, threshold affinity_matrix
            if self._thresh > 0:
                A_row = np.copy(affinity_matrix)
                A_col = np.copy(affinity_matrix.T)
                for i in range(n_samples):
                    # threshold rows
                    idx = np.argsort(A_row[i])[list(range(self._thresh))]
                    A_row[i][idx] = 0
                    # threshold columns
                    idx = np.argsort(A_col[i])[list(range(self._thresh))]
                    A_col[i][idx] = 0
                # average
                affinity_matrix = (A_row + A_col.T) / 2.0

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return base.CallResult(DistanceMatrixOutput(affinity_matrix))
        else:
            affinity_matrix = np.zeros((n_samples, n_samples))
            raise TimeoutError("EKSS fitting has timed out.")

    def __getstate__(self) -> dict:
        return {
            'constructor': {
                'hyperparams': self.hyperparams,
                'random_seed': self.random_seed,
                'docker_containers': self.docker_containers,
            },
            'random_state': self._random_state,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state['constructor'])  # type: ignore
        self._random_state = state['random_state']
        
    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: EKSSHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)