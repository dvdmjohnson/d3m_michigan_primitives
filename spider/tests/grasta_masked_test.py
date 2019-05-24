'''
    grasta_test.py

    @author kgilman

    Unit tests for the spider.unsupervised_learning.grasta_masked module
'''

import unittest
import numpy as np
import numpy.matlib as ml
import scipy.stats
from scipy import sparse
from random import randint

from spider.unsupervised_learning.grasta_masked import GRASTA_MASKED, GRASTA_MASKEDHyperparams

class Test_GRASTA_MASKED(unittest.TestCase):

    def test(self):
        D = 500  # ambient dimension
        n = 150  # number of data vectors
        rank = 5  # rank of low-rank component
        dens = 0.1  # density of sparse component

        np.random.seed(0)

        #Test training capability
        #Generate low-rank matrix
        rando_mat = np.random.randn(D, D)
        Q, R = np.linalg.qr(rando_mat)
        Utrue = Q[:, 0:rank]

        #Generate observation: low-rank + sparse corruptions
        Y = np.zeros([D,n])
        Mask = np.zeros([D,n])
        for i in range(0,n):
            wtrue = np.random.randn(rank)
            l = Utrue @ wtrue

            strue = np.zeros(D)
            numOutliers = int(np.ceil(dens * D))
            idx = np.random.choice(D, numOutliers, replace=False)
            strue[idx] = strue[idx] +  max(abs(l)) * np.random.randn(strue.size)[idx]
            Y[:,i] = l + strue
            
            numObservations = int(np.ceil(0.7*D))
            midx = np.random.choice(D, numObservations, replace = False)
            Mask[midx,i] = 1

        hp = GRASTA_MASKEDHyperparams(
            rank = 5,
            sampling = 1,
            train_sampling = 1,
            admm_max_iter = 20,
            admm_min_iter = 20,
            admm_rho = 1.8,
            max_level = 20,
            max_mu = 15,
            min_mu = 1,
            constant_step = 0,
            max_train_cycles = 10,
        )

        grasta = GRASTA_MASKED(hyperparams=hp, random_seed=randint(0, 2 ** 32 - 1))
        grasta.set_training_data(inputs=Y.T, mask = Mask.T)
        grasta.fit()
        Uhat = grasta._U

        err = np.linalg.norm(Uhat@Uhat.T - Utrue@Utrue.T,'fro')
        determ_discrep = abs(1 - np.linalg.det((Utrue.T @ Uhat)@(Uhat.T @ Utrue)))
        #print("Subspace frobenius norm error: ",err)
        #print("Determinant discrepancy: ", determ_discrep)

        self.assertLess(err, 1e-4)
        self.assertLess(determ_discrep, 1e-6)


        #Test streaming capability
        n = 1500 #number of "streaming" test data
        Y = np.zeros([D,n])
        Mask = np.zeros([D,n])
        for i in range(0,n):

            #Random subspace change at 500 samples: test tracking
            if i == 500:
                rando_mat = np.random.randn(D, D)
                Q, R = np.linalg.qr(rando_mat)
                Utrue = Q[:, 0:rank]

            wtrue = np.random.randn(rank)
            l = Utrue @ wtrue

            strue = np.zeros(D)
            numOutliers = int(np.ceil(dens * D))
            idx = np.random.choice(D, numOutliers, replace=False)
            strue[idx] = strue[idx] + max(abs(l)) * np.random.randn(strue.size)[idx]

            Y[:,i] = l + strue

            numObservations = int(np.ceil(0.7*D))
            midx = np.random.choice(D, numObservations, replace = False)
            Mask[midx,i] = 1

        grasta.set_training_data(inputs = Y.T, mask = Mask.T)
        grasta.continue_fit()

        Ustream = grasta._U

        U_err = np.linalg.norm(Ustream @ Ustream.T - Utrue @ Utrue.T,'fro')
        determ_discrep = abs(1 - np.linalg.det((Utrue.T @ Ustream)@(Ustream.T @ Utrue)))

        self.assertLess(U_err , 2e-4)
        self.assertLess(determ_discrep , 2e-6)
        #print("Streaming subspace error: ",U_err)
        #print("Streaming determinant discrepancy: ",determ_discrep)


        #TEST PRODUCE FUNCTION
        n = 500;
        W = np.random.randn(rank,n)
        L = Utrue @ W
        
        S = np.zeros([D,n])
        dens = 0.1;
        Mask = np.zeros([D,n])
        for i in range(0,n):
            l = L[:,i]
            strue = np.zeros(D)
            numOutliers = int(np.ceil(dens * D))
            idx = np.random.choice(D, numOutliers, replace=False)
            strue[idx] = strue[idx] + max(abs(l)) * np.random.randn(strue.size)[idx]
            S[:,i] = strue;

        numObservations = int(np.ceil(0.7*D))
        midx = np.random.choice(D, numObservations, replace = False)
        Mask[midx,i] = 1
        
        Y = L + S
        
        Lhat = grasta.produce(inputs = Y.T).value
        Lerr = np.linalg.norm(Lhat - L,'fro') / np.linalg.norm(L,'fro')
        #print('Lerr error is: ',Lerr)
        Lerr_tolerance = 0.2
        self.assertLess(Lerr, Lerr_tolerance)
        
        Shat = grasta.produce_sparse(inputs = Y.T).value
        Serr = np.linalg.norm(np.multiply(Mask,Shat - S),'fro')/np.linalg.norm(np.multiply(Mask,S),'fro')
        Serr_tolerance = 0.2
        self.assertLess(Serr,Serr_tolerance)
        
        U = grasta.produce_subspace(inputs = Y.T).value

        test = grasta.get_params()
        test2 = grasta.__getstate__()
        grasta.__setstate__(state = test2)
        #print(test.STATUS)


if __name__ == '__main__':
    unittest.main()

