'''
    grouse_test.py

    @author kgilman

    Unit tests for the spider.unsupervised_learning.grouse module
'''

import unittest
import numpy as np
import numpy.matlib as ml
import scipy.stats
from scipy import sparse
from random import randint

from spider.unsupervised_learning.grouse import GROUSE, GROUSEHyperparams


class Test_GROUSE(unittest.TestCase):

    def test(self):
        D = 500  # ambient dimension
        n = 150  # number of data vectors
        rank = 5  # rank of low-rank component

        np.random.seed(0)

        # Test training capability
        # Generate low-rank matrix
        rando_mat = np.random.randn(D, D)
        Q, R = np.linalg.qr(rando_mat)
        Utrue = Q[:, 0:rank]

        # Generate observation: low-rank with missing data

        Y = Utrue @ np.random.randn(rank,n)

        Mask = np.zeros([D, n])
        for i in range(0, n):
            numObservations = int(np.ceil(0.7 * D))
            midx = np.random.choice(D, numObservations, replace=False)
            Mask[midx, i] = 1

        hp = GROUSEHyperparams(
            rank=5,
            constant_step=0,
            max_train_cycles=10,
        )

        grouse = GROUSE(hyperparams=hp, random_seed=randint(0, 2 ** 32 - 1))
        grouse.set_training_data(inputs=Y.T, mask=Mask.T)
        grouse.fit()
        Uhat = grouse._U

        err = np.linalg.norm(Uhat @ Uhat.T - Utrue @ Utrue.T, 'fro')
        determ_discrep = abs(1 - np.linalg.det((Utrue.T @ Uhat) @ (Uhat.T @ Utrue)))
        # print("Subspace frobenius norm error: ",err)
        # print("Determinant discrepancy: ", determ_discrep)

        self.assertLess(err, 1e-6)
        self.assertLess(determ_discrep, 1e-6)

        # Test streaming capability
        n = 1500  # number of "streaming" test data
        Y = np.zeros([D, n])
        Mask = np.zeros([D, n])
        for i in range(0, n):

            # Random subspace change at 500 samples: test tracking
            if i == 500:
                rando_mat = np.random.randn(D, D)
                Q, R = np.linalg.qr(rando_mat)
                Utrue = Q[:, 0:rank]

            wtrue = np.random.randn(rank)
            Y[:,i] = Utrue @ wtrue

            numObservations = int(np.ceil(0.7 * D))
            midx = np.random.choice(D, numObservations, replace=False)
            Mask[midx, i] = 1

        grouse.set_training_data(inputs=Y.T, mask=Mask.T)
        grouse.continue_fit()

        Ustream = grouse._U

        U_err = np.linalg.norm(Ustream @ Ustream.T - Utrue @ Utrue.T, 'fro')
        determ_discrep = abs(1 - np.linalg.det((Utrue.T @ Ustream) @ (Ustream.T @ Utrue)))

        self.assertLess(U_err, 1e-6)
        self.assertLess(determ_discrep, 1e-6)
        # print("Streaming subspace error: ",U_err)
        # print("Streaming determinant discrepancy: ",determ_discrep)

        # TEST PRODUCE FUNCTION
        n = 500;
        W = np.random.randn(rank, n)
        Y = Utrue @ W

        dens = 0.1;
        Mask = np.zeros([D, n])
        for i in range(0, n):
            numObservations = int(np.ceil(0.7 * D))
            midx = np.random.choice(D, numObservations, replace=False)
            Mask[midx, i] = 1

        Yhat = grouse.produce(inputs=Y.T).value
        Yerr = np.linalg.norm(Yhat - Y, 'fro') / np.linalg.norm(Y, 'fro')
        # print('Lerr error is: ',Lerr)
        Yerr_tolerance = 0.2
        self.assertLess(Yerr, Yerr_tolerance)

        U = grouse.produce_Subspace(inputs=Y.T).value

        test = grouse.get_params()
        test2 = grouse.__getstate__()
        grouse.__setstate__(state=test2)
        # print(test.STATUS)


if __name__ == '__main__':
    unittest.main()

