import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, transformer
from d3m import container, utils
import collections
import os
import numpy as np
import numpy.matlib as ml
from numpy.linalg import norm
from numpy.linalg import svd

__all__ = ('PCP_IALM',)

Inputs = container.ndarray
Outputs = container.ndarray

class PCP_IALMHyperparams(hyperparams.Hyperparams):
    lamb = hyperparams.Union(configuration=collections.OrderedDict({
            'boundedfloat': hyperparams.Bounded[float](lower=1e-10, upper=0.9999, default=0.03),
            'enum': hyperparams.Enumeration[int](values=[-1], default=-1)}),
        default='enum',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="regularization parameter for sparse component")
    mu = hyperparams.Union(configuration=collections.OrderedDict({
            'boundedfloat': hyperparams.Bounded[float](lower=1e-10, upper=0.9999, default=0.1),
            'enum': hyperparams.Enumeration[int](values=[-1], default=-1)}),
        default='enum',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="penalty parameter in Lagrangian function for noise")
    rho = hyperparams.Bounded[float](lower=1.0,
        upper=None,
        default=1.5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="constant used to update mu in each iteration")
    epsilon = hyperparams.LogUniform(lower=1e-10,
        upper=1.0,
        default=1e-7,
        upper_inclusive=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="termination constant")

##  PCP_IALM class
#
#   uses RPCA via PCP-IALM to perform dimensionality reduction
class PCP_IALM(transformer.TransformerPrimitiveBase[Inputs, Outputs, PCP_IALMHyperparams]):

    """"
    Uses RPCA via PCP-IALM to perform dimensionality reduction

    """
    
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'ff4056f0-9644-458a-ba24-66fe3a3bc53a',
        'version': "0.0.5",
        'name': 'PCP_IALM',
        'description': """Does unsupervised dimensionality reduction by computing robust PCA via Principal Component Pursuit with the Inexact Augmented Lagrange Multiplier""",
        'keywords': ['dimensionality reduction', 'PCA', 'robust PCA', 'principal component pursuit', 'inexact augmented lagrange multiplier'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/dimensionality_reduction/pcp_ialm/pcp_ialm.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@article{lin2010augmented,
                           title={The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices},
                           author={Lin, Zhouchen and Chen, Minming and Ma, Yi},
                           journal={arXiv preprint arXiv:1009.5055},
                           year={2010}
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
        'python_path': 'd3m.primitives.data_compression.pcp_ialm.Umich',
        'hyperparams_to_tune': ['rho'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.ROBUST_PRINCIPAL_COMPONENT_ANALYSIS],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_COMPRESSION
        })


    ##  constructor for PCP_IALM class
    #   @param name name associated with primitive
    #   @param lamb regularization parameter for sparse component
    #   @param mu penalty parameter in Lagrangian function for noise
    #   @param rho constant used to update mu in each iteration
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    def __init__(self, *, hyperparams: PCP_IALMHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        # type: (float, float, float, float, int) -> None
        self._lamb = hyperparams['lamb']
        self._mu = hyperparams['mu']
        self._rho = hyperparams['rho']
        self._epsilon = hyperparams['epsilon']
    
    def produce(self, *, inputs: Inputs, iterations: int = None) -> base.CallResult[Outputs]:
        
        def shrink(T, zeta):
            return np.matrix(np.maximum(np.zeros(T.shape), np.array(np.absolute(T)) - zeta) * np.array(np.sign(T)))
        
        def SVD_thresh(X, tau):
            U,s,V = svd(X, full_matrices = False)
            s_thresh = np.array([max(abs(sig) - tau, 0.0) * np.sign(sig) for sig in s])
            return U * np.matrix(np.diag(s_thresh)) * V
        
        D = np.transpose(np.matrix(inputs))
        d,n = inputs.shape
        k = 0
        D_norm_fro = norm(D,'fro')
        D_norm_2 = norm(D,2)
        D_norm_inf = norm(D,np.inf)
        mu = self._mu if self._mu != -1 else 1.25 / D_norm_2
        rho = self._rho
        lamb = self._lamb if self._lamb != -1 else 1.0 / np.sqrt(d)
        Y = D / max(D_norm_2, D_norm_inf / lamb)
        E = ml.zeros(D.shape)
        A = ml.zeros(D.shape)
        while (iterations == None or k < iterations) and norm(D-A-E,'fro') > self._epsilon * D_norm_fro:
            A = SVD_thresh(D - E + Y / mu, 1 / mu)
            E = shrink(D - A + Y / mu, lamb / mu)
            Y = Y + mu * (D - A - E)
            mu = rho * mu
            k += 1
        W = np.array(A.T)
        return base.CallResult(container.ndarray(W, generate_metadata=True))

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: PCP_IALMHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)