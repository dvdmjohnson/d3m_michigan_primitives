import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, clustering
from d3m import container, utils
import collections
import stopit
import os
import numpy as np
import numpy.matlib
from sklearn.cluster import KMeans

Inputs = container.ndarray
Outputs = container.ndarray
DistanceMatrixOutput = container.ndarray

class SSC_ADMMHyperparams(hyperparams.Hyperparams):
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
    alpha = hyperparams.Union(configuration=collections.OrderedDict({
            'enum':hyperparams.Enumeration[int](values=[-1], default=-1),
            'bounded':hyperparams.UniformInt(lower=20, upper=800, default=800, upper_inclusive = True)}),
        default='enum',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Tuning parameter that balances regression and sparsity terms.  If -1, will be initialized to 20 (if no outliers) or 800 (with outliers).")


class SSC_ADMM(clustering.ClusteringDistanceMatrixMixin[Inputs, Outputs, type(None), SSC_ADMMHyperparams, DistanceMatrixOutput],
          clustering.ClusteringTransformerPrimitiveBase[Inputs, Outputs, SSC_ADMMHyperparams]):
    metadata = metadata_module.PrimitiveMetadata({
        'id': '83083e82-088b-47f4-9c0b-ba29adf5a51d',
        'version': "0.0.5",
        'name': 'SSC_ADMM',
        'description': """Does sparse subspace clustering, using the Alternating Direction Method of Multipliers framework for optimization.""",
        'keywords': ['clustering', 'subspace', 'sparse', 'Alternating Direction Method of Multipliers'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://gitlab.datadrivendiscovery.org/michigan/spider/raw/master/spider/cluster/ssc_admm/ssc_admm.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
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
        'python_path': 'd3m.primitives.clustering.ssc_admm.umich',
        'hyperparams_to_tune': ['n_clusters', 'alpha'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.SUBSPACE_CLUSTERING],
        'primitive_family': metadata_module.PrimitiveFamily.CLUSTERING
        })

    def __init__(self, *, hyperparams: SSC_ADMMHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._use_affine = hyperparams['use_affine']
        self._use_outliers = hyperparams['use_outliers']
        self._alpha = hyperparams['alpha'] if hyperparams['alpha'] != -1 else (20 if self._use_outliers else 800)
        self._epsilon = 0.0002
        self._k = hyperparams['n_clusters']
        self._random_state = np.random.RandomState(random_seed)

    def set_training_data(self, *, inputs: Inputs) -> None:
        pass
    
    ##  computes regularization paramater lambda to be used in ADMM algorithm
    #   @param Y DxN data matrix
    #   @param P Dx? modified data matrix
    #   @return regularization paramater lambda for ADMM algorithm
    def _compute_lambda(self, Y, P):
        T = P.T * Y
        np.fill_diagonal(T,0.0)
        T = np.absolute(T)
        l = np.min(np.amax(T, axis = 0))
        return l

    ##  shrinkage threshold operator
    #   @param eta number
    #   @param M NumPy matrix
    #   @return NumPy matrix resulting from applying shrinkage threshold operator to each entry of M
    def _shrinkage_threshold(self, eta, M):
        ST = np.matrix(np.maximum(np.zeros(M.shape), np.array(np.absolute(M)) - eta) * np.array(np.sign(M)))
        return ST

    ##  computes maximum L2-norm error among columns of residual of linear system
    #   @param P DxN NumPy matrix
    #   @param Z NxN NumPy matrix
    #   @return maximum L2-norm of columns of P-P*Z
    def _error_linear_system(self, P, Z):
        R,N = Z.shape
        Y = P[:,:N] if R > N else P
        Y0 = Y - P[:,N:] * Z[N:,:] if R > N else P
        C = Z[:N,:] if R > N else Z
        n = np.linalg.norm(Y0, 2, axis = 0)
        S = np.array((Y0 / n) - Y * (C / n))
        err = np.sqrt(np.max(sum(S*S)))
        return err

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
        normN = np.sqrt(np.sum(np.square(kerN), axis = 1))
        kerNS = (kerN.T / (normN + eps).T).T
        l = KMeans(n_clusters, n_init = n_init, max_iter = max_iter, random_state = self._random_state).fit(kerNS)
        labels = l.labels_.reshape((N,))
        return labels

    ##  ADMM algorithm with outliers
    #   @param X DxN NumPy array/matrix representing N points in D-dimensional space
    #   @param use_affine whether or not data points come from union of affine subspaces instead of linear subspaces
    #   @param alpha constant used in calculating updates
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    #   @return sparse coefficient matrix (NumPy array)
    def _outlier_admm(self, X, use_affine = False, alpha = 20.0, epsilon = 0.0002, max_iter = 200):
    
        Y = np.matrix(X)
        D,N = Y.shape
        gamma = alpha / np.linalg.norm(Y,1)
        P = np.concatenate((Y, np.matlib.eye(D) / gamma), axis = 1)
        mu1 = alpha / self._compute_lambda(Y,P)
        mu2 = alpha
        C = np.matlib.zeros((N+D,N))
    
        if not use_affine:
        
            # initializations
            k = 1
            A = np.linalg.pinv(mu1*P.T*P + mu2*np.matlib.eye(N+D))
            Lambda1 = np.matlib.zeros((D,N))
            Lambda2 = np.matlib.zeros((N+D,N))
            err1 = 10.0 * epsilon
            err2 = 10.0 * epsilon
        
            # main loop
            while k < max_iter and (err1 > epsilon or err2 > epsilon):
                Z = A * (mu1*P.T*(Y+Lambda1/mu1) + mu2*(C-Lambda2/mu2))
                np.fill_diagonal(Z,0.0)
                C = self._shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda1 = Lambda1 + mu1 * (Y - P * Z)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                err1 = np.amax(np.absolute(Z-C))
                err2 = self._error_linear_system(P,Z)
                k += 1

        else:
    
            # initializations
            k = 1
            delta = np.matrix([[float(i < N)] for i in range(N+D)])
            A = np.linalg.pinv(mu1*P.T*P + mu2*np.matlib.eye(N+D) + mu2*delta*delta.T)
            Lambda1 = np.matlib.zeros((D,N))
            Lambda2 = np.matlib.zeros((N+D,N))
            lambda3 = np.matlib.zeros((1,N))
            err1 = 10.0 * epsilon
            err2 = 10.0 * epsilon
            err3 = 10.0 * epsilon
        
            # main loop
            while k < max_iter and (err1 > epsilon or err2 > epsilon or err3 > epsilon):
                Z = A * (mu1*P.T*(Y+Lambda1/mu1) + mu2*(C-Lambda2/mu2) + mu2*delta*(1.0-lambda3/mu2))
                np.fill_diagonal(Z,0.0)
                C = self._shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda1 = Lambda1 + mu1 * (Y - P * Z)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                lambda3 = lambda3 + mu2 * (delta.T * Z - 1.0)
                err1 = np.amax(np.absolute(Z-C))
                err2 = self._error_linear_system(P,Z)
                err3 = np.amax(np.absolute(delta.T * Z - 1.0))
                k += 1

        C = np.array(C[:N,:])
        return C
                
    ##  ADMM algorithm without outliers
    #   @param X DxN NumPy array/matrix representing D points in N-dimensional space
    #   @param use_affine whether or not data points come from union of affine subspaces instead of linear subspaces
    #   @param alpha constant used in calculating updates
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    #   @return sparse coefficient matrix (NumPy array)
    def _lasso_admm(self, X, use_affine = False, alpha = 800.0, epsilon = 0.0002, max_iter = 200):
        
        Y = np.matrix(X)
        N = Y.shape[1]
        mu1 = alpha / self._compute_lambda(Y,Y)
        mu2 = alpha
        C = np.matlib.zeros((N,N))
        
        if not use_affine:
            
            # initializations
            k = 1
            A = np.linalg.pinv(mu1*Y.T*Y + mu2*np.matlib.eye(N))
            Lambda2 = np.matlib.zeros((N,N))
            err1 = 10.0 * epsilon
            
            # main loop
            while k < max_iter and err1 > epsilon:
                Z = A * (mu1*Y.T*Y + mu2*(C-Lambda2/mu2))
                np.fill_diagonal(Z,0.0)
                C = self._shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                err1 = np.amax(np.absolute(Z-C))
                k += 1

        else:
        
            # initializations
            k = 1
            A = np.linalg.pinv(mu1*Y.T*Y + mu2*np.matlib.eye(N) + mu2)
            Lambda2 = np.matlib.zeros((N,N))
            lambda3 = np.matlib.zeros((1,N))
            err1 = 10.0 * epsilon
            err3 = 10.0 * epsilon
            
            # main loop
            while k < max_iter and (err1 > epsilon or err3 > epsilon):
                Z = A * (mu1*Y.T*Y + mu2*(C-Lambda2/mu2) + mu2*np.matlib.ones((N,1))*(1.0-lambda3/mu2))
                np.fill_diagonal(Z,0.0)
                C = self._shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                lambda3 = lambda3 + mu2 * (np.matlib.ones((1,N)) * Z - 1.0)
                err1 = np.amax(np.absolute(Z-C))
                err3 = np.amax(np.absolute(np.matlib.ones((1,N)) * Z - 1.0))
                k += 1

        C = np.array(C)
        return C

    ##  computes sparse coefficient matrix using SSC algorithm with ADMM
    #   @param X NxD NumPy array/matrix representing N points in D-dimensional space
    #   @return sparse coefficient matrix (NumPy array)
    def _compute_sparse_coefficient_matrix(self, X, max_iter):
        XX = np.transpose(X)
        a = self._alpha
        C = self._outlier_admm(XX, self._use_affine, a, self._epsilon, max_iter) if self._use_outliers else self._lasso_admm(XX, self._use_affine, a, self._epsilon, max_iter)
        return C

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        assert inputs.ndim == 2, "Inputs are not in the right shape"

        if iterations == None or iterations < 5:
            iterations = 200

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            C = self._compute_sparse_coefficient_matrix(inputs, iterations)
            W = self._build_adjacency_matrix(C)
            labels = self._spectral_clustering(W, self._k)
            labels = np.array(labels)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return base.CallResult(Outputs(labels))
        else:
            raise TimeoutError("SSC ADMM produce has timed out.")

    def produce_distance_matrix(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[DistanceMatrixOutput]:
        """
            Returns 1 - the affinity matrix generated from the subspace-transformed data
        """
        assert inputs.ndim == 2, "Inputs are not in the right shape"

        if iterations == None or iterations < 5:
            iterations = 200

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            C = self._compute_sparse_coefficient_matrix(inputs, iterations)
            W = self._build_adjacency_matrix(C)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return base.CallResult(DistanceMatrixOutput(1 - W))
        else:
            raise TimeoutError("SSC ADMM produce has timed out.")

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
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: SSC_ADMMHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)