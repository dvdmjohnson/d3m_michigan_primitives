import unittest
import os.path
import numpy as np

import d3m
from d3m import container, utils
from d3m.metadata import base

from spider.supervised_learning.owl import OWLRegression, OWLHyperparams

def generate_linear_data(nSamples, nFeatures):
    """
    y = X * coef + noise
    noise = 0, for simplicity of unittest
    """
    # design matrix
    X = np.random.randn(nSamples, nFeatures)
    X = X - np.mean(X, 0) # centered
    X = X / np.linalg.norm(X, ord=2, axis=0) # normalized

    # noise with variance 0.01
    #noise = np.random.randn(nSamples) * 0.1
    noise = np.zeros(nSamples)

    # coefficients
    coef = np.random.randn(nFeatures)
    
    y = X.dot(coef) + noise
    return coef, container.ndarray(X, generate_metadata=True), container.ndarray(y, generate_metadata=True)

class TestOWLRegression(unittest.TestCase):

    def test_no_regularization(self):
        # Data
        X_train = container.ndarray([1,2]).reshape(-1,1)
        y_train = container.ndarray([2,3])

        X_test = container.ndarray([0, 1.5, 3]).reshape(-1,1)
        y_test_affine = container.ndarray([1, 2.5, 4])
        y_test_linear = container.ndarray([0, 12/5, 24/5])

        # Different ControlHyperparameters; no regularization
        hp_1 = OWLHyperparams(OWLHyperparams.defaults(),
                              tol=1e-8,
                              weight_max_val=0,
                              weight_min_val=0,
                              fit_intercept=True,
                              normalize=False)
        coef_1, intercept_1, pred_1 = [container.ndarray([1]),
                                       float(1),
                                       y_test_affine]

        hp_2 = OWLHyperparams(OWLHyperparams.defaults(),
                              tol=1e-8,
                              weight_max_val=0,
                              weight_min_val=0,
                              fit_intercept=True,
                              normalize=True)
        coef_2, intercept_2, pred_2 = [coef_1,
                                       intercept_1,
                                       pred_1] # same with 1

        hp_3 = OWLHyperparams(OWLHyperparams.defaults(),
                              tol=1e-8,
                              weight_max_val=0,
                              weight_min_val=0,
                              fit_intercept=False)
        # linear regression instead of affine. normalize should be ignored
        coef_3, intercept_3, pred_3 = [container.ndarray([8/5]),
                                       float(0),
                                       y_test_linear]

        # Test primitive
        def almost_equal(arr_1, arr_2, tol):
            # test almost-equality of two float arrays
            eps = np.linalg.norm(arr_1 - arr_2) / (np.linalg.norm(arr_2) + 1e-10)
            if eps < tol:
                return True, None
            else:
                return False, "Error {} >= tolerance {}\narr_1 = {}\narr_2 = {}".format(
                        eps, tol, arr_1, arr_2)

        hps = [hp_1, hp_2, hp_3]
        coefs = [coef_1, coef_2, coef_3]
        intercepts = [intercept_1, intercept_2, intercept_3]
        preds = [pred_1, pred_2, pred_3]
        for i, (hp, coef, intercept, pred) in enumerate(zip(hps, coefs, intercepts, preds)):
            print(i)
            primitive = OWLRegression(hyperparams=hp)
            primitive.set_training_data(inputs=X_train, outputs=y_train)
            primitive.fit()
            ps = primitive.get_params()
            TOL = 1e-2
            self.assertTrue(*almost_equal(ps['coef'], coef, TOL))
            self.assertTrue(*almost_equal(ps['intercept'], intercept, TOL))
            self.assertTrue(ps['fitted'])
            self.assertTrue(*almost_equal(primitive.produce(inputs=X_test).value, pred, TOL))

        
    def test_regularization(self):
        # Generate data, well-posed problem with nSamples > nFeatures
        np.random.seed(0)
        nSamples = 10
        nFeatures = 5
        true_coef, inputs, outputs = generate_linear_data(nSamples, nFeatures)

        # Test fitting with default hyperparams
        hp = OWLHyperparams(OWLHyperparams.defaults())
        primitive = OWLRegression(hyperparams=hp)
        primitive.set_training_data(inputs=inputs, outputs=outputs)
        primitive.fit()
        ps = primitive.get_params()
        self.assertEqual(np.all(ps['coef'] == primitive._coef), True)
        self.assertEqual(ps['intercept'] == primitive._intercept, True)
        self.assertEqual(ps['fitted'], True)

        relative_error = np.linalg.norm(ps['coef'] - true_coef) / np.linalg.norm(true_coef)
        #print("relative_error = {}".format(relative_error))
        self.assertEqual(relative_error < 0.2, True)
        self.assertEqual(np.abs(ps['intercept']) < 0.1, True)

        # Test fitting with customized hyperparams: OSCAR
        hp = OWLHyperparams(OWLHyperparams.defaults(),
                            weight_type='linear',
                            weight_max_val=0.01,
                            weight_max_off=0,
                            weight_min_val=0.005,
                            weight_min_off=nFeatures-1,
                            learning_rate=0.001)
        primitive = OWLRegression(hyperparams=hp)
        primitive.set_training_data(inputs=inputs, outputs=outputs)
        primitive.fit()
        ps = primitive.get_params()
        self.assertEqual(np.all(ps['coef'] == primitive._coef), True)
        self.assertEqual(ps['intercept'] == primitive._intercept, True)
        self.assertEqual(ps['fitted'], True)
        relative_error = np.linalg.norm(ps['coef'] - true_coef) / np.linalg.norm(true_coef)
        #print("relative_error = {}".format(relative_error))
        self.assertEqual(relative_error < 0.2, True)
        self.assertEqual(np.abs(ps['intercept']) < 0.1, True)
   
        # Test single / multiple produce
        inputs_produce = container.ndarray(np.random.randn(1, nFeatures))
        outputs_produce = primitive.produce(inputs=inputs_produce).value
        self.assertEqual(outputs_produce.shape, (1,))

        inputs_produce = container.ndarray(np.random.randn(2, nFeatures))
        outputs_produce = primitive.produce(inputs=inputs_produce).value
        self.assertEqual(outputs_produce.shape, (2,))

if __name__ == '__main__':
    unittest.main()

