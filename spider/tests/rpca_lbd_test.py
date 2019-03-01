'''
    rpca_lbd_test.py

    @author agitlin

    Unit tests for the spider.dimensionality_reduction.rpca_lbd module
'''

import unittest
import numpy as np
import numpy.matlib as ml
import scipy.stats
from scipy import sparse
from random import randint

from spider.dimensionality_reduction.rpca_lbd import RPCA_LBD, RPCA_LBDHyperparams

class Test_RPCA_LBD(unittest.TestCase):
    
    def test(self):
        D = 10 # ambient dimension
        n = 20 # number of data vectors
        rank = 4 # rank of low-rank component
        dens = 0.1 # density of sparse component
        np.random.seed(0)
        L = np.matrix(np.random.rand(n,rank)) * np.matrix(np.random.rand(rank,D)) # low-rank component
        S = np.matrix(sparse.random(n,D,density=dens,random_state=0,data_rvs=scipy.stats.uniform(loc=-50,scale=100).rvs).A) # sparse component
        A = np.array(L+S) # data matrix

        hp = RPCA_LBDHyperparams(
            kappa = 1.1,
            lamb = 0.61,
            rho = 1.1,
            beta = 0.2,
            alpha = 1.0,
            outer_epsilon = 1e-7,
            inner_epsilon = 1e-6,
            inner_max_iter = 20)

        rpca = RPCA_LBD(hyperparams=hp, random_seed=randint(0, 2**32-1))
        W = rpca.produce(inputs = A, iterations = 100).value
        self.assertEqual(len(W.shape), 2)
        self.assertEqual(W.shape[0], n)
        self.assertEqual(W.shape[1], D)


