import tensorflow as tf
from tensorflow import keras
from .pca_variance import flatten, get_pca_variance, linear_fit
from .dimension_reg_layer import get_alpha

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


class DimensionRegGammaWeights(keras.layers.Layer):

    def __init__(self, strength=0.01, target_value=1, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg", "alpha")
        self.metric_name = metric_name

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name}

    def call(self, x):
        if x.shape[0] == None:
            return x
        x2 = flatten(x)
        if x2.shape[1] > 10000:
            x2 = tf.gather(x2, tf.random.uniform(shape=[10000], maxval=x2.shape[1], dtype=tf.int32, seed=10), axis=1)

        loss = get_alpha_regularizer(x2, self.target_value) * self.strength
        self.add_loss(loss)
        self.add_metric(loss, self.metric_name+"_loss")

        if self.calc_alpha:
            alpha = get_alpha(x2)
        else:
            alpha = 0
        # record it as a metric
        self.add_metric(alpha, self.metric_name)

        return x
