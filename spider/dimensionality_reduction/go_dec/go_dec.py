import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, transformer
from d3m import container, utils
import collections
import os
import stopit
import numpy as np
import numpy.matlib as ml
from numpy.linalg import norm
from scipy.linalg import qr

__all__ = ('GO_DEC',)

Inputs = container.ndarray
Outputs = container.ndarray

class GO_DECHyperparams(hyperparams.Hyperparams):
    c = hyperparams.Union(configuration=collections.OrderedDict({
            'boundedfloat': hyperparams.Bounded[float](lower=0.0, upper=0.9999, default=0.03),
            'boundedint': hyperparams.Bounded[int](lower=1, upper=None, default=10)}),
        default='boundedfloat',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="maximum cardinality of sparse component (if c>=1 then max card = c; if 0<c<1 then max card = c*m*n where (m,n) is shape of data matrix)")
    r = hyperparams.Union(configuration=collections.OrderedDict({
            'boundedfloat': hyperparams.Bounded[float](lower=0.0, upper=0.9999, default=0.1),
            'boundedint': hyperparams.Bounded[int](lower=1, upper=None, default=3)}),
        default='boundedfloat',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="maximum rank of low-rank component (if r>=1 then max rank = r; if 0<r<1 then max rank = r*min(m,n) where (m,n) is shape of data matrix)")
    power = hyperparams.Bounded[int](lower=1,
        upper=10,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter', 'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        description="power scheme modification (larger power leads to higher accuracy and higher time cost)")
    epsilon = hyperparams.LogUniform(lower=0.0,
        upper=1.0,
        default=1e-3,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        upper_inclusive=True,
        description="termination constant")

##  GO_DEC class
#
#   uses RPCA via the Go Decomposition (GoDec) to perform dimensionality reduction
class GO_DEC(transformer.TransformerPrimitiveBase[Inputs, Outputs, GO_DECHyperparams]):
    
    metadata = metadata_module.PrimitiveMetadata({
        'id': '0b2edee8-df52-49f2-85b6-470abb28f4eb',
        'version': "0.0.5",
        'name': 'GO_DEC',
        'description': """Does unsupervised dimensionality reduction by computing robust PCA using the Go Decomposition algorithm.""",
        'keywords': ['dimensionality reduction', 'PCA', 'robust PCA', 'Go Decomposition'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                #link to file and repo
                'https://gitlab.datadrivendiscovery.org/michigan/spider/raw/master/spider/dimensionality_reduction/go_dec/go_dec.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
            'citation': """@inproceedings{zhou2011godec,
                           title={Godec: Randomized low-rank \& sparse matrix decomposition in noisy case},
                           author={Zhou, Tianyi and Tao, Dacheng},
                           booktitle={International conference on machine learning},
                           year={2011},
                           organization={Omnipress}
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
             'package_uri': 'git+https://gitlab.datadrivendiscovery.org/michigan/spider.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.data_compression.go_dec.umich',
        'hyperparams_to_tune': ['c', 'r'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.ROBUST_PRINCIPAL_COMPONENT_ANALYSIS],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_COMPRESSION
        })

    ##  constructor for GO_DEC class
    def __init__(self, *, hyperparams: GO_DECHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._card = hyperparams['c']
        self._r = hyperparams['r']
        self._power = hyperparams['power']
        self._epsilon = hyperparams['epsilon']
        self._random_state = np.random.RandomState(random_seed)
    
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        
        def largest_entries(M, kk):
            k = int(kk)
            L = np.array(np.absolute(M)).ravel()
            ind = np.argpartition(L, -k)[-k:]
            Y = np.zeros(len(L))
            LL = np.array(M).ravel()
            Y[ind] = LL[ind]
            return np.matrix(Y.reshape(M.shape))

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            X = np.matrix(np.transpose(inputs))
            m,n = X.shape
            rank = self._r if self._r >= 1 else np.floor(self._r * min(m, n))
            card = self._card if self._card >= 1 else np.floor(self._card * m * n)
            r = self._r
            L = np.matrix(X)
            S = ml.zeros(X.shape)
            itr = 1
            while True:
                Y2 = np.matrix(self._random_state.normal(size = (n,int(rank))))
                for i in range(self._power + 1):
                    Y1 = L * Y2
                    Y2 = L.T * Y1
                Q,R = qr(Y2,mode='economic')
                L_new = L * Q * Q.T
                T = L - L_new + S
                L = L_new
                S = largest_entries(T,card)
                T = T - S
                if norm(T,'fro') < self._epsilon or (iterations != None and itr > iterations):
                    break
                L = L + T
                itr += 1
            W = np.array(L.T)
        
            return base.CallResult(container.ndarray(W))

        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("GO_DEC produce timed out.")

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
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: GO_DECHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)