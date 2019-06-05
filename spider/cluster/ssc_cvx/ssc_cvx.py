import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, clustering
from d3m import container, utils
import collections
import os
import numpy as np
from sklearn.cluster import KMeans
from cvxpy import *

Inputs = container.ndarray
Outputs = container.ndarray
DistanceMatrixOutput = container.ndarray

class SSC_CVXHyperparams(hyperparams.Hyperparams):
    n_clusters = hyperparams.Bounded[int](lower=2,
        upper=None,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="number of subspaces/clusters to learn")
    use_affine = hyperparams.Enumeration[bool](values=[True, False],
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should be True if data is derived from affine subspaces rather than linear subspaces")
    use_outliers = hyperparams.Enumeration[bool](values=[True, False],
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should be True if you believe the data countains instances that are outliers to all subspaces")
    use_noise = hyperparams.Enumeration[bool](values=[True, False],
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should be True if you believe the data is noisy")
    alpha = hyperparams.Union(configuration=collections.OrderedDict({
            'enum':hyperparams.Enumeration[int](values=[-1], default=-1),
            'bounded':hyperparams.UniformInt(lower=20, upper=800, default=800, upper_inclusive = True)}),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default='enum',
        description="Tuning parameter that balances regression and sparsity terms.  If -1, will be initialized to 20 (if no outliers) or 800 (with outliers).")


class SSC_CVX(clustering.ClusteringDistanceMatrixMixin[Inputs, Outputs, type(None), SSC_CVXHyperparams, DistanceMatrixOutput],
          clustering.ClusteringTransformerPrimitiveBase[Inputs, Outputs, SSC_CVXHyperparams]):
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'a36f322f-e02d-4618-b022-d741d672366f',
        'version': "0.0.5",
        'name': 'SSC_CVX',
        'description': """Does sparse subspace clustering using convex optimization.""",
        'keywords': ['clustering', 'subspace', 'sparse', 'convex optimization'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/cluster/ssc_cvx/ssc_cvx.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@article{elhamifar2013sparse,
  title={Sparse subspace clustering: Algorithm, theory, and applications},
  author={Elhamifar, Ehsan and Vidal, Rene},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={35},
  number={11},
  pages={2765--2781},
  year={2013},
  publisher={IEEE}}"""
            },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.clustering.ssc_cvx.Umich',
        'hyperparams_to_tune': ['n_clusters', 'alpha'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.SUBSPACE_CLUSTERING],
        'primitive_family': metadata_module.PrimitiveFamily.CLUSTERING
        })
    
    def __init__(self, *, hyperparams: SSC_CVXHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._use_affine = hyperparams['use_affine']
        self._use_outliers = hyperparams['use_outliers']
        self._use_noise = hyperparams['use_noise']
        self._alpha = hyperparams['alpha'] if hyperparams['alpha'] != -1 else (20 if self._use_outliers else 800)
        self._k = hyperparams['n_clusters']
        self._random_state = np.random.RandomState(random_seed)

    def set_training_data(self, *, inputs: Inputs) -> None:
        pass

    ##  computes adjacency matrix given coefficient matrix
    #   @param C NxN coefficient matrix (NumPy matrix)
    #   @return NxN adjacency matrix (NumPy matrix)
    def _build_adjacency_matrix(self, C):
        eps = 2.220446049250313e-16
        N = C.shape[0]
        CAbs = np.absolute(C)
        for i in range(N):
            CAbs[:,i] = CAbs[:,i] / (np.amax(CAbs[:,i]) + eps)
        A = CAbs + np.transpose(CAbs) + eps
        np.fill_diagonal(A,0.0)
        return A

    ##  spectral clustering algorithm
    #   @param W NxN adjacency matrix (NumPy matrix)
    #   @param n_clusters number of clusters
    #   @param max_iter maximum number of iterations for KMeans
    #   @param n_init number of replications for KMeans
    #   @return labels for N points
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

    ##  computes sparse coefficient matrix using SSC algorithm with convex optimization
    #   @param X NxD NumPy array/matrix representing N points in D-dimensional space
    #   @return sparse coefficient matrix
    def _compute_sparse_coefficient_matrix(self, X):
    
        Y = np.transpose(np.array(X))
        D,N = Y.shape
        a = self._alpha if self._alpha != -1 else (20.0 if self._use_outliers else 800.0)
        le = a / np.min([np.max([np.linalg.norm(Y[:,j],1) for j in range(N) if j != i]) for i in range(N)]) if self._use_outliers else 0.0
        lz = a / np.min([np.max([np.absolute(np.dot(Y[:,i],Y[:,j])) for j in range(N) if j != i]) for i in range(N)]) if self._use_noise else 0.0
        C = np.zeros((N,N))
    
        # find sparse coefficient matrix C using convex optimization
        for i in range(N):
            # since cii = 0, we can treat ci as (N-1)x1 vector
            ci = Variable(shape=(N-1,1))
            ei = Variable(shape=(D,1))
            zi = Variable(shape=(D,1))
            Yi = np.delete(Y,i,axis=1)
            yi = Y[:,i]
            objective = None
            constraints = []
            if self._use_outliers and self._use_noise:
                objective = Minimize(norm(ci,1) + le * norm(ei,1) + 0.5 * lz * (norm(zi,2) ** 2))
                constraints = [yi == ((Yi * ci) + ei + zi).flatten()]
            elif self._use_outliers and not self._use_noise:
                objective = Minimize(norm(ci,1) + le * norm(ei,1))
                constraints = [yi == Yi * ci + ei]
            elif not self._use_outliers and self._use_noise:
                objective = Minimize(norm(ci,1) + 0.5 * lz * (norm(zi,2) ** 2))
                constraints = [yi == Yi * ci + zi]
            else:
                objective = Minimize(norm(ci,1))
                constraints = [yi == Yi * ci]
            if self._use_affine:
                constraints.append(np.ones((1,N-1)) * ci == 1)
            prob = Problem(objective, constraints)
            result = prob.solve()
            # turn ci into Nx1 vector by setting cii = 0 and set column i of C equal to ci
            C[:,i] = np.insert(np.asarray(ci.value),i,0,axis=0).flatten()

        return C
   
    def produce(self, *, inputs: Inputs, iterations: int = None) -> base.CallResult[Outputs]:

        if iterations == None or iterations < 5:
            iterations = 1000

        C = self._compute_sparse_coefficient_matrix(inputs)
        W = self._build_adjacency_matrix(C)
        labels = self._spectral_clustering(W, n_clusters = self._k, max_iter = iterations)
        labels = np.array(labels)

        return base.CallResult(Outputs(labels))

    def produce_distance_matrix(self, *, inputs: Inputs) -> base.CallResult[DistanceMatrixOutput]:
        """
            Returns 1 - the affinity matrix generated from the subspace-transformed data
        """

        C = self._compute_sparse_coefficient_matrix(inputs)
        W = self._build_adjacency_matrix(C)

        return base.CallResult(DistanceMatrixOutput(1 - W))

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
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: SSC_CVXHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
