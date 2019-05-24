'''
    spider.tests: rfd.py

    @tsnowak

    Unit tests for the spider.distance.rfd module
'''
import os
import sys
from random import randint
from scipy.io import savemat, loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import unittest

from d3m import container

from spider.distance.rfd import RFD, RFDHyperparams
from spider.distance.utils import normalize_labels

class testRFD(unittest.TestCase):
    
    def test_param_get_set(self):
        ''' Test RFD Parameter retrieving and setting.
        '''
        # load data
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, "data/iris.mat")
        temp = loadmat(data_path)
        dat = temp["da_iris"]
        labs = normalize_labels(temp["la_iris"])
        
        self.assertEqual(labs.min(), 0)
        self.assertEqual(labs.max(), 2)
        
        # make a training/test split
        all = list(range(len(labs)))
        test = all[0::5]
        testset = set(test)
        trainset = set(all)
        trainset -= testset
        train = list(trainset)

        # generate training and test data
        traindat = container.ndarray(dat[train, :])
        testdat = container.ndarray(dat[test, :])
        trainlabs = container.ndarray(labs[train])
        testlabs = container.ndarray(labs[test])
        
        self.assertEqual(len(test), 30)
        self.assertEqual(len(train), 120)
        self.assertEqual(traindat.shape[1], testdat.shape[1])
        self.assertEqual(traindat.shape[1], 4)

        # train metric
        hp = RFDHyperparams(class_cons=450,
                 num_trees=500,
                 min_node_size=1,
                 n_jobs=-1)
        rfd_metric = RFD(hyperparams=hp, random_seed=randint(0, 2**32-1))
        rfd_metric.set_training_data(inputs=traindat, outputs=trainlabs)
        rfd_metric.fit()
        
        # get output
        test_kernel = rfd_metric.produce(inputs=testdat, second_inputs=traindat).value

        #now make another rfd object from this one's parameters and compare
        params = rfd_metric.get_params()
        rfd_metric2 = RFD(hyperparams=hp, random_seed=randint(0, 2**32-1))
        rfd_metric2.set_params(params=params)
        test_kernel2 = rfd_metric2.produce(inputs=testdat, second_inputs=traindat).value

        diff = (test_kernel - test_kernel2).sum()
        
        self.assertLess(diff, 0.001)
    
    def test_knn_classification(self):
        ''' Test RFD knn classification
        '''
        # load data
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, "data/iris.mat")
        temp = loadmat(data_path)
        dat = temp["da_iris"]
        labs = normalize_labels(temp["la_iris"])
        
        self.assertEqual(labs.min(), 0)
        self.assertEqual(labs.max(), 2)
        
        # make a training/test split
        all = list(range(len(labs)))
        test = all[0::5]
        testset = set(test)
        trainset = set(all)
        trainset -= testset
        train = list(trainset)

        # generate training and test data
        traindat = container.ndarray(dat[train, :])
        testdat = container.ndarray(dat[test, :])
        trainlabs = container.ndarray(labs[train])
        testlabs = container.ndarray(labs[test])
        
        self.assertEqual(len(test), 30)
        self.assertEqual(len(train), 120)
        self.assertEqual(traindat.shape[1], testdat.shape[1])
        self.assertEqual(traindat.shape[1], 4)

        # train metric
        hp = RFDHyperparams(class_cons=450,
                 num_trees=500,
                 min_node_size=1,
                 n_jobs=-1)
        rfd_metric = RFD(hyperparams=hp, random_seed=randint(0, 2**32-1))
        rfd_metric.set_training_data(inputs=traindat, outputs=trainlabs)
        rfd_metric.fit()
        
        # get output
        test_kernel = rfd_metric.produce(inputs=testdat, second_inputs=traindat).value
        
        self.assertEqual(test_kernel.shape, (30, 120))
        self.assertLess(test_kernel.min(), 0.1)
        self.assertGreaterEqual(test_kernel.min(), 0)
        self.assertLessEqual(test_kernel.max(), 1.0)
        self.assertGreater(test_kernel.max(), 0.9)
        
        # do nearest neighbor classification
        classifier = KNeighborsClassifier(n_neighbors=5, metric="precomputed", n_jobs=-1)
        # use dummy training matrix, because it's not needed
        classifier.fit(np.zeros((len(train), len(train))), trainlabs)
        prediction = classifier.predict(test_kernel)
        
        self.assertEqual(prediction.shape, (30,))
        self.assertTrue((np.unique(prediction) == np.array([0, 1, 2], dtype=np.int32)).all())
        
        # compute accuracy
        acc = np.float32(testlabs == prediction).sum() / len(testlabs)
        
        self.assertGreater(acc, 0.6)

if __name__ == '__main__':
    unittest.main()





