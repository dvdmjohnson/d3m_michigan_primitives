'''
    spider.tests: cluster_kss_test.py

    Unit tests for the spider.cluster.kss module
'''

import unittest
import tempfile
import os.path
import shutil

import numpy as np
from scipy.linalg import orth
from sklearn.preprocessing import normalize
from random import randint

from spider.cluster.kss import KSS, KSSHyperparams


class testKSS(unittest.TestCase):


    def test_kss(self):
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

        # run KSS
        hp = KSSHyperparams(
            n_clusters = K, 
            dim_subspaces = d)
        ksub = KSS(hyperparams=hp, random_seed=randint(0, 2**32-1))
        ksub.set_training_data(inputs = X.T)
        ksub.fit()
        estimated_labels = ksub.produce(inputs = X.T).value
        self.assertEqual(len(estimated_labels), N)

        dmat = ksub.produce_distance_matrix(inputs = X.T).value
        self.assertEqual(dmat.shape[0], N)
        self.assertEqual(dmat.shape[1], N)

if __name__ == '__main__':
    unittest.main()
