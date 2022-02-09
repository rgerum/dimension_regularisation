# test_with_unittest.py

from unittest import TestCase
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dimension_regularisation.pca_variance import flatten, get_pca_variance, linear_fit
from dimension_regularisation.dimension_reg_layer import get_alpha
from sklearn.decomposition import PCA


class TryTesting(TestCase):

    def test_flatten(self):
        t = keras.Input([10, 30])
        model = keras.models.Sequential([
            t,
            keras.layers.Lambda(lambda x: flatten(x)),
        ])
        self.assertEqual(model.output.shape[0], None)
        self.assertEqual(model.output.shape[1], 300)
        self.assertEqual(flatten(np.zeros([100, 200])).shape, (100, 200))
        self.assertEqual(flatten(np.zeros([300, 1])).shape, (300, 1))
        self.assertEqual(flatten(np.zeros([300, 10, 30])).shape, (300, 300))

    def test_get_pca(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2.]]).astype(np.float32)
        pca = PCA()
        pca.fit(X)
        np.testing.assert_almost_equal(pca.explained_variance_ratio_, get_pca_variance(X).numpy())

        np.random.seed(1234)
        X = np.random.rand(400, 200)
        pca = PCA()
        pca.fit(X)
        np.testing.assert_almost_equal(pca.explained_variance_ratio_, get_pca_variance(X).numpy())

    def test_linear_fit(self):
        np.random.seed(1234)
        params = [4.123, 5.5334]
        X = np.random.rand(400)
        y = params[0] + params[1]*X
        np.testing.assert_almost_equal(params, [a.numpy() for a in linear_fit(X, y)])

    def test_alpha(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        np.testing.assert_almost_equal(1.2, get_alpha(x_train[:28*28].astype(np.float32)), decimal=1)
