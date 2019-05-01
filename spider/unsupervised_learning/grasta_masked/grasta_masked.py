import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, transformer, unsupervised_learning
from d3m import container, utils
import collections
import os
import stopit
import numpy as np


__all__ = ('GRASTA_MASKED',)

Inputs = container.ndarray
Outputs = container.ndarray


class GRASTA_MASKEDHyperparams(hyperparams.Hyperparams):
    # dim = hyperparams.Bounded[int](lower=1,
    #                                    upper=None,
    #                                    default=500,
    #                                     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    #                                    description="Ambient dimension of data")
    rank = hyperparams.Bounded[int](lower=1,
                                      upper=None,
                                      default=5,
                                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                      description="Rank of learned low-rank matrix")
    sampling = hyperparams.Bounded[float](lower=0.01,
                                     upper=1,
                                     default=1,
                                      semantic_types=[
                                          'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                     description="Matrix sub-sampling parameter")

    train_sampling = hyperparams.Bounded[float](lower=0.01, upper=1, default = 1,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description="Matrix sub-sampling parameter during training")

    admm_max_iter = hyperparams.Bounded[int](lower=1,
                                      upper=None,
                                      default=20,
                                     semantic_types=[
                                         'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                      description="Maximum ADMM iterations")
    admm_min_iter = hyperparams.Bounded[int](lower=1,
                                       upper=None,
                                       default=20,
                                         semantic_types=[
                                             'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                       description="Minimum ADMM iterations")

    admm_rho = hyperparams.Bounded[float](lower = 1, upper = None,default = 1.8, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description="ADMM rho parameter")

    max_level = hyperparams.Bounded[int](lower=1,
                                           upper=None,
                                           default=20,
                                            semantic_types=[
                                                 'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                           description="Maximum level in multi-level adaptive step size")
    max_mu = hyperparams.Bounded[int](lower=1,
                                           upper=None,
                                           default=15,
                                          semantic_types=[
                                              'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                           description="Maximum mu in multi-level adaptive step size")
    min_mu = hyperparams.Bounded[int](lower=1,
                                              upper=None,
                                              default=1,
                                              semantic_types=[
                                                  'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                              description="Minimum mu in multi-level adaptive step size")

    constant_step = hyperparams.Bounded[float](lower=0, upper=None, default = 0,semantic_types=[
                                          'https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = "Make nonzero for contant step size instead of multi-level adaptive step")

    max_train_cycles = hyperparams.Bounded[int](lower = 1, upper = None, default = 10,semantic_types=[
                                          'https://metadata.datadrivendiscovery.org/types/TuningParameter'],description="Number of times to cycle over training data")


    ###Don't know how to set the upper bound here
    # training_size = hyperparams.Bounded[int](lower = 1, upper = None, default = 50, semantic_types=[
    #                                       'https://metadata.datadrivendiscovery.org/types/TuningParameter'], description= "Number of random training samples")


### GRASTA OPTIONS CLASS
class _OPTIONS(object):
    def __init__(self, dim_m, rank, subsampling=1, admm_max_itr = 20, admm_min_itr = 20, max_level = 20, max_mu = 15, min_mu = 1, constant_step = 0):
        self.admm_max_itr = admm_max_itr
        self.admm_min_itr = admm_min_itr
        self.dim_m = dim_m
        self.rank = rank
        self.subsampling = subsampling

        self.max_level = max_level
        self.max_mu = max_mu
        self.min_mu = min_mu
        self.constant_step = constant_step

class _STATUS(object):
    def __init__(self, last_mu, last_w, last_gamma, level = 0, step_scale = 0, train = 0):
        self.last_mu = last_mu
        self.level = level
        self.step_scale = step_scale
        self.last_w = last_w
        self.last_gamma = last_gamma
        self.train = train

class _OPTS(object):
    def __init__(self,max_iter=20,rho=1.8,tol=1e-8):
        self.max_iter = max_iter
        self.rho = rho
        self.tol = tol

class GRASTA_MASKEDParams(params.Params):
    OPTIONS: _OPTIONS
    STATUS: _STATUS
    OPTS: _OPTS
    U: np.ndarray


##  GRASTA class
#
#   uses GRASTA to perform online dimensionality reduction of (possibly) sub-sampled data
class GRASTA_MASKED(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, GRASTA_MASKEDParams, GRASTA_MASKEDHyperparams]):
    """
    Uses GRASTA to perform online dimensionality reduction of (possibly) sub-sampled data
    """
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'acf412e2-1087-4257-a26e-73c39c2c8d7d',
        'version': "0.0.5",
        'name': 'GRASTA_MASKED',
        'description': """Performs online, unsupervised dimensionality reduction by computing robust PCA on the Grassmannian manifold with missing data.""",
        'keywords': ['unsupervised learning','dimensionality reduction', 'robust PCA', 'low-rank', 'online','streaming','Grassmannian manifold','subspace tracking','matrix completion','video surveillance', 'missing data'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                # link to file and repo
                'https://gitlab.datadrivendiscovery.org/michigan/spider/raw/master/spider/unsupervised_learning/GRASTA/GRASTA_MASKED.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
            'citation': """@inproceedings{he2014grasta, title={Incremental Gradient on the Grassmannian for Online Foreground and Background Separation in Subsampled Video}, author={He, Balzano and Lui}, booktitle={Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference On}, pages={1568–1575}, year={2014}, organization={IEEE}}"""
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
        'python_path': 'd3m.primitives.data_compression.grasta_masked.Umich',
        'hyperparams_to_tune': ['rank','admm_rho', 'max_level','max_mu','min_mu','constant_step','max_train_cycles'],
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.ROBUST_PRINCIPAL_COMPONENT_ANALYSIS],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_COMPRESSION
    })

    #GRASTA class constructor and instantiation
    def __init__(self, *, hyperparams: GRASTA_MASKEDHyperparams, random_seed: int = 0,
                 docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        # self._dim = hyperparams['dim']
        self._rank = hyperparams['rank']
        self._sampling = hyperparams['sampling']
        self._train_sampling = hyperparams['train_sampling']
        self._admm_max_iter = hyperparams['admm_max_iter']
        self._admm_min_iter = hyperparams['admm_min_iter']
        self._admm_rho = hyperparams['admm_rho']
        self._max_level = hyperparams['max_level']
        self._max_mu = hyperparams['max_mu']
        self._min_mu = hyperparams['min_mu']
        self._constant_step = hyperparams['constant_step']
        self._max_train_cycles = hyperparams['max_train_cycles']
        # self._training_size = hyperparams['training_size']

        self._X: Inputs = None
        self._Mask: Inputs = None
        self._U = None
        self._random_state = np.random.RandomState(random_seed)

        #Instantiate GRASTA status and admm control params
        # self._grastaSTATUS = _STATUS(last_mu=self._min_mu, last_w=np.zeros(self._rank),
        #                       last_gamma=np.zeros(self._dim), train=1)
        self._admm_OPTS = _OPTS()


    def set_training_data(self, *, inputs: Inputs, mask: Inputs) -> None:
        self._X = inputs
        self._Mask = mask
        self._dim = inputs.shape[1]
        self._training_size = inputs.shape[0]
        self._grastaSTATUS = _STATUS(last_mu=self._min_mu, last_w=np.zeros(self._rank),
                      last_gamma=np.zeros(self._dim), train=1)

    #GRASTA fit function: learns low-rank subspace from training data
    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:

        #Internal function to generate low-rank random matrix
        def generateLRMatrix(d, r):
            rando_mat = self._random_state.randn(d, d)
            Q, R = np.linalg.qr(rando_mat)
            U = Q[:, 0:r]
            return U

        assert self._X is not None, "No training data provided."
        assert self._X.ndim == 2, "Data is not in the right shape."
        assert self._rank <= self._X.shape[1], "Dim_subspaces should be less than ambient dimension."
        assert self._min_mu < self._max_mu, "Min mu cannot be greater than max mu"
        assert self._admm_min_iter <= self._admm_max_iter, "Min admm iterations cannot exceed max admm iterations"

        _X = self._X.T  #Get the training data
        _Mask = self._Mask.T  #Get the mask

        #Begin training
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:

            # Instantiate a random low-rank subspace
            d = self._dim
            r = self._rank
            U = generateLRMatrix(d, r)

            # Set the training control params
            self._grastaOPTIONS = _OPTIONS(self._dim, self._rank, self._train_sampling, self._admm_max_iter,
                                          self._admm_min_iter,
                                          self._max_level, self._max_mu, self._min_mu, self._constant_step)

            U = self._train_grasta(_X,_Mask,U)
            self._U = U #update global variable

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:

            return base.CallResult(None)
        else:
            raise TimeoutError("GRASTA fit() has timed out.")

    # GRASTA training internal function
    def _train_grasta(self, X, Mask, U):

        max_cycles = self._max_train_cycles
        train_size = self._training_size
        #train_size = self._training_size
        # train_size = Mask.shape[1]

        for i in range(0, max_cycles):
            perm = self._random_state.choice(train_size, train_size, replace=False)  # randomly permute training data
            for j in range(0, train_size):
                x = X[:, perm[j]]  # get a column of data
                xidx = np.where(Mask[:,perm[j]])[0]
                U, w, s, STATUS_new, admm_OPTS_new = self._grasta_stream(U,x,xidx)

                # Update subspace and control variables
                self._grastaSTATUS = STATUS_new
                self._admm_OPTS = admm_OPTS_new

        return U

    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:

        #Get the vector input, and the subspace
        _X = self._X.T  # Get the data
        _Mask = self._Mask.T #Get the mask indicating observed indices
        d,numVectors = _X.shape
        Uhat = self._U

        #Set the proper subsampling for streaming
        self._grastaOPTIONS.subsampling = self._sampling

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:

            for i in range(0,numVectors):
                _x = _X[:,i]
                _xidx = np.where(_Mask[:,i])[0]

                #Call GRASTA iteration
                U, w, s, STATUS_new, admm_OPTS_new  = self._grasta_stream(Uhat,_x,_xidx)

                #Update subspace and control variables
                self._grastaSTATUS = STATUS_new
                self._admm_OPTS = admm_OPTS_new
                self._U = U

                Uhat = U

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:

            return base.CallResult(None)
        else:
            raise TimeoutError("GRASTA continue_fit() has timed out.")

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        X = inputs
        U = self._U
        
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            Y = U @ (U.T @ X.T)

            return base.CallResult(container.ndarray(Y, generate_metadata=True))

        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("GRASTA produce timed out.")


    def produce_subspace(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        X = inputs
        U = self._U

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            
            return base.CallResult(container.ndarray(U, generate_metadata=True))

        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("GRASTA produce timed out.")

    def produce_sparse(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        X = inputs
        U = self._U
        
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            S = X.T - U @ (U.T @ X.T)
            
            return base.CallResult(container.ndarray(S, generate_metadata=True))

        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("GRASTA produce timed out.")

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        result = base.MultiCallResult()
        X = inputs
        U = self._U
        
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            result.values = {}
            if 'produce' in produce_methods:
                result.values['produce'] = U @ (U.T @ X.T)
            if 'produce_subspace' in produce_methods:
                result.values['produce_subspace'] = U
            if 'produce_sparse' in produce_methods:
                result.values['produce_sparse'] = X.T - U @ (U.T @ X.T)
            result.has_finished = True
            return result

        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("GRASTA (masked) multiproduce timed out.")

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], mask: Inputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        #set training data

        self._X = inputs
        self._Mask = mask

        #fit

        #Internal function to generate low-rank random matrix
        def generateLRMatrix(d, r):
            rando_mat = self._random_state.randn(d, d)
            Q, R = np.linalg.qr(rando_mat)
            U = Q[:, 0:r]
            return U

        assert self._X is not None, "No training data provided."
        assert self._X.ndim == 2, "Data is not in the right shape."
        assert self._rank <= self._X.shape[1], "Dim_subspaces should be less than ambient dimension."
        assert self._min_mu < self._max_mu, "Min mu cannot be greater than max mu"
        assert self._admm_min_iter <= self._admm_max_iter, "Min admm iterations cannot exceed max admm iterations"

        _X = self._X.T  #Get the training data
        _Mask = self._Mask.T  #Get the mask

        #Begin training
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:

            # Instantiate a random low-rank subspace
            d = self._dim
            r = self._rank
            U = generateLRMatrix(d, r)

            # Set the training control params
            self._grastaOPTIONS = _OPTIONS(self._dim, self._rank, self._train_sampling, self._admm_max_iter,
                                          self._admm_min_iter,
                                          self._max_level, self._max_mu, self._min_mu, self._constant_step)

            U = self._train_grasta(_X,_Mask,U)
            self._U = U #update global variable

            #produce

            result = base.MultiCallResult()
            X = _X
            U = self._U

            result.values = {}
            if 'produce' in produce_methods:
                result.values['produce'] = U @ (U.T @ X.T)
            if 'produce_subspace' in produce_methods:
                result.values['produce_subspace'] = U
            if 'produce_sparse' in produce_methods:
                result.values['produce_sparse'] = X.T - U @ (U.T @ X.T)
            result.has_finished = True
            return result

        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("GRASTA (masked) multiproduce timed out.")

    ### MAIN GRASTA UPDATE FUNCTION
    def _grasta_stream(self,Uhat, x, xIdx):

        ### ELEMENTWISE SOFT THRESHOLDING FUNCTION
        def shrinkage(a, kappa):
            y = np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)
            return y

        ### SIGMOID FUNCTION
        def sigmoid(x):
            FMIN = -1
            FMAX = 1
            omega = 0.1
            y = FMIN + (FMAX - FMIN) / (1 - (FMAX / FMIN) * np.exp(-x / omega))
            return y

        ### ADMM FUNCTION
        def admm(Uomega, xOmega, ADMM_OPTS):
            tol = ADMM_OPTS.tol
            y = np.zeros(xOmega.size)
            s = np.zeros(xOmega.size)
            rho = ADMM_OPTS.rho
            mu = rho

            converged = False
            itrs = 0

            pinv_U = np.linalg.pinv(Uomega)
            max_itrs = ADMM_OPTS.max_iter

            while not converged and itrs <= max_itrs:
                w = pinv_U @ (xOmega - s + y / mu)
                s = shrinkage(xOmega - Uomega @ w + y / mu, 1 / mu)
                h = xOmega - Uomega @ w - s
                y = y + mu * h
                h_norm = np.linalg.norm(h, 2)
                itrs += 1
                if (h_norm < tol):
                    converged = True
                else:
                    mu = mu * rho
            return w, s, y, h

        ### Multi-level Adaptive Step Size Calculation Function
        def calculate_mla_step(grastaSTATUS, grastaOPTIONS, admm_OPTS, gamma, w, sG):

            level_factor = 2

            MAX_MU = grastaOPTIONS.max_mu
            MIN_MU = grastaOPTIONS.min_mu
            MAX_LEVEL = grastaOPTIONS.max_level

            ITER_MAX = grastaOPTIONS.admm_max_itr
            MIN_ITER = grastaOPTIONS.admm_min_itr

            last_w = grastaSTATUS.last_w
            last_gamma = grastaSTATUS.last_gamma
            last_mu = grastaSTATUS.last_mu
            level = grastaSTATUS.level
            step_scale = grastaSTATUS.step_scale

            DEFAULT_MU_HIGH = (MAX_MU - 1) / 2
            DEFAULT_MU_LOW = MIN_MU + 2

            # 1. Determine step-scale from 1st observation
            if step_scale == 0:
                step_scale = 0.5 * np.pi * (1 + MIN_MU) / sG

            # 2. Inner product of previous and current gradients
            test_mat = (last_gamma.T @ gamma) * np.multiply.outer(last_w, w)
            grad_ip = np.trace(test_mat)

            # Avoid too large of inner products
            normalization = np.linalg.norm(np.multiply.outer(last_gamma, last_w.T), 'fro') * np.linalg.norm(
                np.multiply.outer(gamma, w.T), 'fro')
            if normalization == 0:
                grad_ip_normalization = 0
            else:
                grad_ip_normalization = grad_ip / normalization

            # 3. Take step by sigmoid rule. If gradients in same direction, take a larger step; o.w. small step
            mu = max(last_mu + sigmoid(-grad_ip_normalization), MIN_MU)

            if grastaOPTIONS.constant_step > 0:
                step = grastaOPTIONS.constant_step
                MAX_ITER = ITER_MAX
            else:
                step = step_scale * level_factor ** (-level) * sG / (1 + mu)

                if step >= np.pi / 3:
                    step = np.pi / 3

                bShrUpd = 0

                if mu <= MIN_MU:
                    if level > 1:
                        bShrUpd = 1
                        level = level - 1

                    mu = DEFAULT_MU_LOW

                elif mu > MAX_MU:
                    if level < MAX_LEVEL:
                        bShrUpd = 1
                        level = level + 1
                        mu = DEFAULT_MU_HIGH
                    else:
                        mu = MAX_MU

                if bShrUpd:
                    if level >= 0 and level < 4:
                        MAX_ITER = grastaOPTIONS.admm_min_itr
                    elif level >= 4 and level < 7:
                        MAX_ITER = min(MIN_ITER * 2, ITER_MAX)
                    elif level >= 7 and level < 10:
                        MAX_ITER = min(MIN_ITER * 4, ITER_MAX)
                    elif level >= 10 and level < 14:
                        MAX_ITER = min(MIN_ITER * 8, ITER_MAX)
                    else:
                        MAX_ITER = ITER_MAX
                else:
                    MAX_ITER = ITER_MAX

            STATUS_new = _STATUS(mu, w, gamma, level, step_scale)
            ADMM_OPTS_new = _OPTS(MAX_ITER, admm_OPTS.rho, admm_OPTS.tol)

            return step, STATUS_new, ADMM_OPTS_new


        ### Main GRASTA update

        DIM = self._grastaOPTIONS.dim_m

#xIdx = self._random_state.choice(DIM, int(np.ceil(self._grastaOPTIONS.subsampling * DIM)), replace=False)

        xOmega = x[xIdx]
        Uomega = Uhat[xIdx, :]

        w_hat, s_hat, y_hat, h = admm(Uomega, xOmega, self._admm_OPTS)

        gamma1 = y_hat + (xOmega - Uomega @ w_hat - s_hat)
        gamma2 = Uomega.T @ gamma1
        gamma = np.zeros(DIM)
        gamma[xIdx] = gamma1
        gamma = gamma - Uhat @ gamma2

        w_norm = np.linalg.norm(w_hat)
        gamma_norm = np.linalg.norm(gamma, 2)

        sG = gamma_norm * w_norm

        t, STATUS_new, admm_OPTS_new = calculate_mla_step(self._grastaSTATUS, self._grastaOPTIONS, self._admm_OPTS,
                                                           gamma, w_hat, sG)

        step = np.multiply.outer([((np.cos(t) - 1) * (Uhat @ w_hat) / w_norm) + (np.sin(t) * gamma / gamma_norm)],
                                 (w_hat / w_norm))
        step = np.squeeze(step)
        Unew = Uhat + step

        return Unew, w_hat, s_hat, STATUS_new, admm_OPTS_new

    def get_params(self) -> GRASTA_MASKEDParams:
        return GRASTA_MASKEDParams(OPTIONS = self._grastaOPTIONS, STATUS = self._grastaSTATUS, OPTS = self._admm_OPTS, U = self._U)

    def set_params(self, *, params: GRASTA_MASKEDParams) -> None:
        self._grastaOPTIONS = params['OPTIONS']
        self._grastaSTATUS = params['STATUS']
        self._admm_OPTS = params['OPTS']
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





