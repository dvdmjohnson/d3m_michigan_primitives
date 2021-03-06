'''
    ssc_cvx_test.py

    Unit tests for the spider.cluster.ssc_cvx module
'''

import unittest
import numpy as np
import numpy.matlib as ml
from sklearn.preprocessing import normalize
from scipy.linalg import orth
from random import randint
from spider.cluster.ssc_omp import SSC_OMP, SSC_OMPHyperparams

class Test_SSC_OMP(unittest.TestCase):

    def test_ssc_cvx(self):
        """ Test runnability and verify shapes of estimated_labels

        """

        # parameters
        D = 100  # dimension of ambient space
        K = 5  # number of subspaces
        Nk = 100  # points per subspace
        d = 1  # dimension of subspace
        varn = 0.01   # noise variance
        N = K * Nk

        # generate data
        X = np.zeros((D, N))
        true_labels = np.zeros(N)
        true_U = np.zeros((K, D, d))
        for kk in range(K):
            true_U[kk] = orth(np.random.randn(D, d))
            x = np.dot(true_U[kk], np.random.randn(d, Nk))
            X[:,list(range(Nk*kk, Nk*(kk+1)))] = x
            true_labels[list(range(Nk*kk, Nk*(kk+1)))] = kk * np.ones(Nk)

        noise = np.sqrt(varn) * np.random.randn(D, N)
        X = X + noise
        X = normalize(X, norm='l2', axis=0)

        # run ssc_cvx
        hp = SSC_OMPHyperparams(
            n_clusters = K,
            sparsity_level = 3,
            thresh = 1e-6)
        cvx_sub = SSC_OMP(hyperparams=hp, random_seed=randint(0, 2**32-1))
        estimated_labels = cvx_sub.produce(inputs = X.T).value
        self.assertEqual(len(estimated_labels), N)

if __name__ == '__main__':
    unittest.main()

