'''
    pcp_ialm_test.py

    @author agitlin

    Unit tests for the spider.dimensionality_reduction.pcp_ialm module
'''

import unittest
import numpy as np
import numpy.matlib as ml
import scipy.stats
from scipy import sparse
from random import randint

from spider.dimensionality_reduction.pcp_ialm import PCP_IALM, PCP_IALMHyperparams

class Test_PCP_IALM(unittest.TestCase):
    
    def test(self):
        D = 10 # ambient dimension
        n = 20 # number of data vectors
        rank = 4 # rank of low-rank component
        dens = 0.1 # density of sparse component
        np.random.seed(0)
        L = np.matrix(np.random.rand(n,rank)) * np.matrix(np.random.rand(rank,D)) # low-rank component
        S = np.matrix(sparse.random(n,D,density=dens,random_state=0,data_rvs=scipy.stats.uniform(loc=-50,scale=100).rvs).A) # sparse component
        A = np.array(L+S) # data matrix

        hp = PCP_IALMHyperparams(
            lamb = -1, 
            mu = -1,
            rho = 1.5, 
            epsilon = 1e-7)

        rpca = PCP_IALM(hyperparams=hp, random_seed=randint(0, 2**32-1))
        W = rpca.produce(inputs = A, iterations = 100).value
        self.assertEqual(len(W.shape), 2)
        self.assertEqual(W.shape[0], n)
        self.assertEqual(W.shape[1], D)


