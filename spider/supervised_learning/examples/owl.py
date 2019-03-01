import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import d3m
from d3m import container, utils
from d3m.metadata import base

from spider.supervised_learning.owl import OWLRegression, OWLHyperparams

def get_multivariate_normal(nSamples, nFeatures):
    param_var, param_cov = [1, 0.9]
    mu = np.zeros(nFeatures)
    cov = np.full((nFeatures,nFeatures), param_cov)
    for i in range(nFeatures):
        cov[i,i] = param_var
    X = np.random.multivariate_normal(mu, cov, nSamples)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized
    return X

def generate_data_1():
    """
    y = X * coef + noise
    """
    nSamples = 100
    nFeatures = 100

    # design matrix
    param_var = 1
    X = np.random.randn(nSamples, nFeatures) * np.sqrt(param_var)
    X[:, 20:30] = get_multivariate_normal(nSamples, 10)
    X[:, 60:70] = get_multivariate_normal(nSamples, 10)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized
    X = container.ndarray(X)
    
    # noise: variance 0.01
    noise = np.random.randn(nSamples) * 0.1

    # coef
    coef = np.zeros(nFeatures)
    coef[20:30] = -1
    coef[60:70] = 1
    
    y = X.dot(coef) + noise
    y = container.ndarray(y)
    return coef, X, y

def generate_data_2():
    """
    Zhong, Kwok. Efficient sparse modeling with automatic features grouping, IEEE Signal Processing Letters, 2012
    
    p = 40, 15 nonzeros and 25 zeros
    """
    nFeatures = 40
    nSamples = 20

    coef = np.zeros(nFeatures)
    coef[:15] = 3

    z = np.random.randn(nSamples, 3)
    eps_std = 0.4
    eps = np.random.randn(nSamples, 15) * eps_std
    X = np.zeros((nSamples, nFeatures))
    for i in range(3):
        X[:,i*5:(i+1)*5] = z[:,i:i+1] + eps[:,i*5:(i+1)*5]
    X[:,15:] = np.random.randn(nSamples, 25)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized
    X = container.ndarray(X)

    y = X.dot(coef) # + noise
    y = container.ndarray(y)
    return coef, X, y

def owl_regression(inputs, outputs, hp, fname):
    plt.figure()
    plt.plot(inputs[:3,:].T)
    plt.title("First 3 features of the design matrix")
    plt.savefig(fname + "_features.png")
    plt.show()

    primitive = OWLRegression(hyperparams=hp)
    primitive.set_training_data(inputs=inputs, outputs=outputs)
    primitive.fit(iterations=1000)

    plt.figure()
    plt.plot(primitive._weight)
    plt.title("OWL weights")
    plt.savefig(fname + "_OWL_weights.png")
    plt.show()

    plt.figure()
    plt.plot(primitive._loss_history)
    plt.title("Training loss history")
    plt.savefig(fname + "_loss.png")
    plt.show()

    plt.figure()
    plt.plot(true_param, label='True')
    plt.plot(primitive._coef, label='Estimated')
    plt.title('Coeffcients, MSE={}'.format(np.mean((true_param - primitive._coef)**2)))
    plt.legend()
    plt.savefig(fname + "_coef.png")
    plt.show()
    
    return primitive._coef

if __name__ == "__main__":
    res_dir = 'Results'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    np.random.seed(0)
    true_param, inputs, outputs = generate_data_1()
    hp = OWLHyperparams(OWLHyperparams.defaults(),
                    weight_type = "linear",
                    weight_max_val = 0.015,
                    weight_max_off = 0,
                    weight_min_val = 0.01,
                    weight_min_off = 99,
                    learning_rate = 0.001)
    owl_regression(inputs, outputs, hp, os.path.join(res_dir, "eg1"))
    
    np.random.seed(0)
    true_param, inputs, outputs = generate_data_2()
    hp = OWLHyperparams(OWLHyperparams.defaults(),
                    weight_type = "linear",
                    weight_max_val = 0.015,
                    weight_max_off = 0,
                    weight_min_val = 0.01,
                    weight_min_off = 30,
                    learning_rate = 0.001)
    owl_regression(inputs, outputs, hp, os.path.join(res_dir, "eg2"))
