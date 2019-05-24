'''
    Use random forest distance (RFD) to learn a nonlinear distance metric
    (technically pseudo-semimetric) from labeled training data and apply it
    to k-nearest neighbor classification.
'''

import os
from scipy.io import savemat, loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=4, suppress=True)

import resource
import time

from spider.distance.rfd import RFD
from spider.distance.utils import get_random_constraints, normalize_labels

# load data
this_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(this_dir, "data/iris.mat")
temp = loadmat(data_path)
dat = temp["da_iris"]
labs = normalize_labels(temp["la_iris"])

# make a training/test split
all = list(range(len(labs)))
test = all[0::5]
testset = set(test)
trainset = set(all)
trainset -= testset
train = list(trainset)

# generate training and test data
traindat = dat[train, :]
testdat = dat[test, :]
trainlabs = labs[train]
testlabs = labs[test]

# train metric
rfd_metric = RFD(class_cons=450,
                 num_trees=500,
                 min_node_size=1,
                 n_jobs=-1,
                 verbose=1)
rfd_metric.set_training_data(inputs=traindat, outputs=trainlabs)
rfd_metric.fit()

# get output
test_kernel = rfd_metric.produce(inputs=(testdat, traindat))

# do nearest neighbor classification
classifier = KNeighborsClassifier(
    n_neighbors=5, metric="precomputed", n_jobs=-1)
# use dummy training matrix, because it's not needed
classifier.fit(np.zeros((len(train), len(train))), trainlabs)
prediction = classifier.predict(test_kernel)

# compute and print accuracy
acc = np.float32(testlabs == prediction).sum() / len(testlabs)
print("Nearest-neighbor classification accuracy: " + str(acc))
