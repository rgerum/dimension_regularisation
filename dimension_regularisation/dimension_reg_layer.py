import tensorflow as tf
from tensorflow import keras
from .pca_variance import flatten, get_pca_variance, linear_fit


@tf.function
def get_alpha(data, min_x=5, max_x=50):
    """ get the power law exponent of the PCA value distribution """
    # flatten the non-batch dimensions
    data = flatten(data)
    # get the eigenvalues of the covariance matrix
    eigen_values = get_pca_variance(data)

    # ensure that eigenvalues are slightly positive (prevents log from giving nan)
    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    # get the logarithmic x and y values to fit
    y = tf.math.log(eigen_values)
    x = tf.math.log(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype))
    a, b = linear_fit(x[min_x:max_x], y[min_x:max_x])
    # return the negative of the slope
    return -b


@tf.function
def get_alpha_from_lambdas(eigen_values, min_x=5, max_x=50):
    """ get the power law exponent of the PCA value distribution """
    # ensure that eigenvalues are slightly positive (prevents log from giving nan)
    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    # get the logarithmic x and y values to fit
    y = tf.math.log(eigen_values)
    x = tf.math.log(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype))
    a, b = linear_fit(x[min_x:max_x], y[min_x:max_x])
    # return the negative of the slope
    return -b


class DimensionReg(keras.layers.Layer):
    """ a layer to calculate and regularize the exponent of the eigenvalue spectrum """
    def __init__(self, strength=0.01, target_value=1, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg", "alpha")
        self.metric_name = metric_name
        self.calc_alpha = True

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name}

    def call(self, x):
        # flatten the non-batch dimensions
        x2 = flatten(x)
        # if the array is too big create a random sampled sub set
        if x2.shape[1] > 10000:
            x2 = tf.gather(x2, tf.random.uniform(shape=[10000], maxval=x2.shape[1], dtype=tf.int32, seed=10), axis=1)
        # get the alpha value
        if self.calc_alpha:
            alpha = get_alpha(x2)
        else:
            alpha = 0
        # record it as a metric
        self.add_metric(alpha, self.metric_name)
        # calculate the loss and add is a metric
        loss = tf.math.abs(alpha-self.target_value)*self.strength
        self.add_metric(loss, self.metric_name+"_loss")
        self.add_loss(loss)

        # return the unaltered x
        return x
