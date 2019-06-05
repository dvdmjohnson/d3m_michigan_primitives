import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, transformer
from d3m import container, utils
import os
import numpy as np
import numpy.matlib as ml
from numpy.linalg import norm
from numpy.linalg import svd

__all__ = ('RPCA_LBD',)

Inputs = container.ndarray
Outputs = container.ndarray

class RPCA_LBDHyperparams(hyperparams.Hyperparams):
    kappa = hyperparams.Bounded[float](lower=1e-10,
        upper=None,
        default=1.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="column-sparsity regularization parameter")
    lamb = hyperparams.Bounded[float](lower=1e-10,
        upper=None,
        default=0.61,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="block sparse regularization parameter")
    rho = hyperparams.Bounded[float](lower=1e-10,
        upper=None,
        default=1.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="outer loop update parameter")
    beta = hyperparams.Bounded[float](lower=1e-10,
        upper=None,
        default=0.2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="inner loop update parameter b")
    alpha = hyperparams.Bounded[float](lower=1e-10,
        upper=1.999999999,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="inner loop update parameter a")
    outer_epsilon = hyperparams.LogUniform(lower=1e-10,
        upper=1.0,
        default=1e-7,
        upper_inclusive=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter', 'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="termination constant for outer loop")
    inner_epsilon = hyperparams.LogUniform(lower=1e-10,
        upper=1.0,
        default=1e-6,
        upper_inclusive=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter', 'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="termination constant for inner loop")
    inner_max_iter = hyperparams.Bounded[int](lower=3,
        upper=500,
        default=20,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter', 'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="iteration limit for inner loop")

##  RPCA_LBD class
#
#   uses RPCA-LBD to perform dimensionality reduction
class RPCA_LBD(transformer.TransformerPrimitiveBase[Inputs, Outputs, RPCA_LBDHyperparams]):

    """
    RPCA based on the Low-Rank and Block-Sparse Decomposition (LBD) model
    @param inputs data matrix (NumPy array/matrix where rows are samples and columns are features)
    @return W low-rank component of input matrix (NumPy array with same shape as inputs)
    """
    
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'a1d3dc5d-fe3a-4e9c-b661-bd9198a934cd',
        'version': "0.0.5",
        'name': 'RPCA_LBD',
        'description': """Does unsupervised dimensionality reduction by computing robust PCA via low-rank block-sparse decomposition.""",
        'keywords': ['dimensionality reduction', 'PCA', 'robust PCA', 'low-rank', 'block-sparse decomposition'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/dimensionality_reduction/rpca_lbd/rpca_lbd.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@inproceedings{tang2011robust, title={Robust principal component analysis based on low-rank and block-sparse matrix decomposition}, author={Tang, Gongguo and Nehorai, Arye}, booktitle={Information Sciences and Systems (CISS), 2011 45th Annual Conference on}, pages={1--5}, year={2011}, organization={IEEE}}"""
            },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.data_compression.rpca_lbd.Umich',
        'hyperparams_to_tune': ['kappa', 'lamb'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.ROBUST_PRINCIPAL_COMPONENT_ANALYSIS],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_COMPRESSION
        })

    def __init__(self, *, hyperparams: RPCA_LBDHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._kappa = hyperparams['kappa']
        self._lamb = hyperparams['lamb']
        self._rho = hyperparams['rho']
        self._beta = hyperparams['beta']
        self._alpha = hyperparams['alpha']
        self._epsilon = [hyperparams['outer_epsilon'], hyperparams['inner_epsilon']]
        self._inner_max_iter = hyperparams['inner_max_iter']

    ##  RPCA based on the Low-Rank and Block-Sparse Decomposition (LBD) model
    #   @param inputs data matrix (NumPy array/matrix where rows are samples and columns are features)
    #   @return W low-rank component of input matrix (NumPy array with same shape as inputs)
    def produce(self, *, inputs: Inputs, iterations: int = None) -> base.CallResult[Outputs]:
        def SVD_thresh(X, tau):
            U,s,V = svd(X, full_matrices = False)
            s_thresh = np.array([max(abs(sig) - tau, 0.0) * np.sign(sig) for sig in s])
            return U * np.matrix(np.diag(s_thresh)) * V
        
        def col_thresh(X, tau):
            C = ml.zeros(X.shape)
            for i in range(X.shape[1]):
                Xi_norm = norm(X[:,i],2)
                if Xi_norm > tau:
                    C[:,i] = X[:,i] * (1.0 - (tau / Xi_norm))
            return C

        D = np.transpose(np.matrix(inputs))
        D_norm = norm(D,'fro')
        A = np.matrix(D)
        E = ml.zeros(D.shape)
        Y = ml.zeros(D.shape)
        mu = 30.0 / norm(np.sign(D),2)
        kappa = self._kappa
        lamb = self._lamb
        rho = self._rho
        beta = self._beta
        alpha = self._alpha
        err_outer = 10.0 * self._epsilon[0] * D_norm
        k = 0
        num_iter = 0
        while k < iterations and err_outer > self._epsilon[0] * D_norm:
            GA = D - E + Y / mu
            A = GA
            A_save = ml.zeros(D.shape)
            err_inner = 10.0 * self._epsilon[1]
            j = 0
            while j < self._inner_max_iter and err_inner > self._epsilon[1]:
                A_save = SVD_thresh(A,beta)
                A_old = A
                Mat = (2.0 * A_save - A + beta * mu * GA) / (1.0 + beta * mu)
                A = A + alpha * (col_thresh(Mat, beta*kappa*(1.0-lamb)/(1.0+beta*mu)) - A_save)
                err_inner = norm(A-A_old,'fro')
                j += 1
            num_iter += j
            A = A_save
            GE = D - A + Y / mu
            E = col_thresh(GE, kappa*lamb/mu)
            Y = Y + mu * (D-A-E)
            mu *= rho
            err_outer = norm(D-A-E,'fro')
            k += 1
        W = np.array(A.T)
        return base.CallResult(container.ndarray(W, generate_metadata=True))

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: RPCA_LBDHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)