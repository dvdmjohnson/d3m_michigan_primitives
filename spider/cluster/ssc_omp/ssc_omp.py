import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, clustering
from d3m import container, utils
import os
import numpy as np
from scipy.linalg import lstsq
import scipy.sparse as sps
from sklearn.cluster import KMeans


Inputs = container.ndarray
Outputs = container.ndarray
DistanceMatrixOutput = container.ndarray

class SSC_OMPHyperparams(hyperparams.Hyperparams):
    n_clusters = hyperparams.Bounded[int](lower=2,
        upper=None,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="number of subspaces/clusters to learn")
    sparsity_level = hyperparams.UniformInt(lower=3, 
        upper=50,
        default=3,
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The algorithm terminates when it has selected this many regression coefficients.")
    thresh = hyperparams.LogUniform(lower=1e-10, 
        upper=1,
        default=1e-6,
        upper_inclusive = True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Minimum regression residual for termination.")


class SSC_OMP(clustering.ClusteringDistanceMatrixMixin[Inputs, Outputs, type(None), SSC_OMPHyperparams, DistanceMatrixOutput],
          clustering.ClusteringTransformerPrimitiveBase[Inputs, Outputs, SSC_OMPHyperparams]):
    """
    This code implements the subspace clustering algorithm described in
    Chong You, Daniel Robinson, Rene Vidal,
    "Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit", CVPR 2016.

    It performs the OMP algorithm on every column of X using all other columns as a
    dictionary

    :param data: A dxN numpy array
    :param K: The maximum subspace dimension
    :param thres: termination condition
    :return: the SSC-OMP representation of the data
    """

    metadata = metadata_module.PrimitiveMetadata({
        'id': '50f89f90-7cef-4bb6-b56f-642f85bd1d58',
        'version': "0.0.5",
        'name': 'SSC_OMP',
        'description': """Does sparse subspace clustering using orthogonal matching pursuit.""",
        'keywords': ['clustering', 'subspace', 'sparse', 'orthogonal matching pursuit'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/cluster/ssc_omp/ssc_omp.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@inproceedings{you2016scalable,
  title={Scalable sparse subspace clustering by orthogonal matching pursuit},
  author={You, Chong and Robinson, Daniel and Vidal, Ren{\'e}},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3918--3927},
  year={2016}}"""
            },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.clustering.ssc_omp.Umich',
        'hyperparams_to_tune': ['n_clusters', 'sparsity_level'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.SUBSPACE_CLUSTERING],
        'primitive_family': metadata_module.PrimitiveFamily.CLUSTERING
        })
    
    def __init__(self, *, hyperparams: SSC_OMPHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._k = hyperparams['n_clusters']
        self._max_subspace_dim = hyperparams['sparsity_level']
        self._thres = hyperparams['thresh']
        self._random_state = np.random.RandomState(random_seed)

    def set_training_data(self, *, inputs: Inputs) -> None:
        pass

    @staticmethod
    def _cNormalize(data, norm=2):
        """
        This method performs the column wise normalization of the input data
        :param data: A dxN numpy array
        :param norm: the desired norm value (This has to be in accordance with the accepted numpy
         norm values
        :return: Returns the column wise normalised data
        """
        return data / (np.linalg.norm(data, ord=norm, axis = 0) + 2.220446049250313e-16)

    @staticmethod
    def _OMPMatFunction(data, K, thres):

        memory_total = 0.1 * 10**9
        _, n = data.shape
        data_normalised = SSC_OMP._cNormalize(data)
        support_set = np.ones((n, K), dtype=np.int64)
        indices = np.arange(n, dtype=np.int64).reshape(n, 1) * np.ones((1, K))
        values = np.zeros((n, K))
        t_vector = np.ones((n, 1), dtype=np.int64) * K
        residual = np.copy(data_normalised)

        for t in range(K):
            counter = 0
            block_size = np.ceil(memory_total / n)
            while True:
                mask = np.arange(counter, min(counter+block_size, n))
                iMat = np.abs(np.matmul(data.T, residual[:, mask]))
                np.fill_diagonal(iMat, 0.0)
                jMat = np.argmax(iMat, axis=0)
                support_set[mask, t] = jMat
                counter = counter + block_size
                if counter >= n:
                    break

            if t+1 != K:
                for iN in range(n):
                    if t_vector[iN] == K:
                        B = data_normalised[:, support_set[iN, 0:(t+1)]]
                        mat_tmp, _, _, _ = lstsq(B, data_normalised[:, iN])

                        residual[:, iN] = data_normalised[:, iN] - np.matmul(B, mat_tmp)

                        if np.sum(residual[:, iN]**2) < thres:
                            t_vector[iN] = t

            if not np.any(K == t_vector):
                break

        for iN in range(n):
            tmp, _, _, _ = lstsq(data[:, support_set[iN, 0:np.asscalar(t_vector[iN] + 1)]], (data[:, iN]))
            values[iN, 0:np.asscalar(t_vector[iN])] = tmp.T

        sparse_mat = sps.coo_matrix((values.flat, (support_set.flat, indices.flat)), shape=(n, n))
        sparse_mat = sparse_mat.toarray()
        return sparse_mat

    def _spectral_clustering(self, W, n_clusters = 10, max_iter = 1000, n_init = 20):
        N,_ = W.shape
        eps = 2.220446049250313e-16
        DN = np.diag(1/np.sqrt(np.sum(W, axis = 0) + eps))
        LapN = np.identity(N) - np.matmul(np.matmul(DN, W), DN)
        _, _, VN = np.linalg.svd(LapN)
        kerN = VN.T[:,(N - n_clusters):N]
        normN = np.sqrt(np.sum(np.square(kerN), axis = 1));
        kerNS = (kerN.T / (normN + eps).T).T
        l = KMeans(n_clusters, n_init = n_init, max_iter = max_iter, random_state = self._random_state).fit(kerNS)
        labels = l.labels_.reshape((N,))
        return labels

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert self._max_subspace_dim <= inputs.shape[1], "max_subspace dim can't be greater than the" + \
        "input feature space"

        if iterations is None or iterations < 5:
            iterations = 200

        data = inputs.T
        R = SSC_OMP._OMPMatFunction(data, self._max_subspace_dim, self._thres)
        np.fill_diagonal(R, 0)
        A = np.abs(R) + np.abs(R.T)
        labels = self._spectral_clustering(A, n_clusters=self._k, max_iter=iterations, n_init=20)

        return base.CallResult(Outputs(labels))

    def produce_distance_matrix(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[DistanceMatrixOutput]:
        """
            Returns 1 - the affinity matrix generated from the subspace-transformed data
        """
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert self._max_subspace_dim <= inputs.shape[1], "max_subspace dim can't be greater than the" + \
        "input feature space"

        data = inputs.T
        R = SSC_OMP._OMPMatFunction(data, self._max_subspace_dim, self._thres)
        np.fill_diagonal(R, 0)
        A = np.abs(R) + np.abs(R.T)

        return base.CallResult(DistanceMatrixOutput(1 - A))


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
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: SSC_OMPHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)