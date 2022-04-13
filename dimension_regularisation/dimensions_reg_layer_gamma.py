import tensorflow as tf
from tensorflow import keras
from .pca_variance import flatten, get_pca_variance, linear_fit, get_eigen_vectors, get_eigen_values_from_vectors
from .dimension_reg_layer import get_alpha, get_alpha_from_lambdas

@tf.function
def get_alpha_regularizer(data, alpha=1.):
    lambdas = get_pca_variance(data)
    lambdas = tf.cast(lambdas, tf.float32)
    N = lambdas.shape[0]
    N = 100
    tau = 5
    lambdas = lambdas[tau:N]
    kappa = lambdas[0] * tf.math.pow(float(tau), alpha)
    gammas = kappa * tf.math.pow(tf.range(tau, N, dtype=tf.float32), -alpha)
    loss = 1/N * tf.reduce_sum((lambdas/gammas - 1) ** 2 + tf.nn.relu(lambdas/gammas - 1))
    return loss

@tf.function
def get_alpha_regularizer_from_lambdas(lambdas, alpha=1.):
    lambdas = tf.cast(lambdas, tf.float32)
    N = lambdas.shape[0]
    N = 100
    tau = 5
    lambdas = lambdas[tau:N]
    kappa = lambdas[0] * tf.math.pow(float(tau), alpha)
    gammas = kappa * tf.math.pow(tf.range(tau, N, dtype=tf.float32), -alpha)
    loss = 1/N * tf.reduce_sum((lambdas/gammas - 1) ** 2 + tf.nn.relu(lambdas/gammas - 1))
    return loss


class DimensionRegGammaWeights(keras.layers.Layer):

    def __init__(self, strength=0.01, target_value=1, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg_gamma_weights", "alpha")
        self.metric_name = metric_name

        self.calc_alpha = True

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name}

    def call(self, x):
        if x.shape[0] == None:
            return x

        if self.calc_alpha:
            x2 = flatten(x)
            if x2.shape[1] > 10000:
                x2 = tf.gather(x2, tf.random.uniform(shape=[10000], maxval=x2.shape[1], dtype=tf.int32, seed=10), axis=1)

            loss = get_alpha_regularizer(x2, self.target_value) * self.strength

            alpha = get_alpha(x2)
        else:
            loss = 0
            alpha = 0

        self.add_loss(loss)
        self.add_metric(loss, self.metric_name + "_loss")

        # record it as a metric
        self.add_metric(alpha, self.metric_name)

        return x



class DimensionRegGammaWeightsPreComputedBase(tf.keras.layers.Layer):

    def __init__(self, strength=0.01, target_value=1, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg_gamma_weights", "alpha")
        self.metric_name = metric_name

        self.calc_alpha = True
        self.eigen_vectors = None

        self.calculate_eigenvectors = True

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name}

    def call(self, x):
        if x.shape[0] == None:
            return x

        if self.calc_alpha:
            x2 = flatten(x)
            if x2.shape[1] > 10000:
                x2 = tf.gather(x2, tf.random.uniform(shape=[10000], maxval=x2.shape[1], dtype=tf.int32, seed=10), axis=1)

            if self.calculate_eigenvectors is True:
                self.eigen_vectors = get_eigen_vectors(x2)
                print("self.eigen_vectors", self.eigen_vectors.shape)

            eigen_values = get_eigen_values_from_vectors(x2, self.eigen_vectors)

            loss = get_alpha_regularizer_from_lambdas(eigen_values, self.target_value) * self.strength
            loss = tf.cast(loss, tf.float32)
            alpha = get_alpha_from_lambdas(eigen_values)
        else:
            loss = 0
            alpha = 0

        self.add_loss(loss)
        self.add_metric(loss, self.metric_name + "_loss")
        # record it as a metric
        self.add_metric(alpha, self.metric_name)

        return x



class CalcEigenVectors(keras.callbacks.Callback):
    def __init__(self, data):
        self.data = data

    def on_epoch_start(self, epoch, logs={}):
        for layer in self.model.layers:
            layer.calculate_eigenvectors = True
        self.model(self.data)
        for layer in self.model.layers:
            layer.calculate_eigenvectors = False
