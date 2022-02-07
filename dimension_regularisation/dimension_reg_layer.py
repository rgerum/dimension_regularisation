import tensorflow as tf
from tensorflow import keras


@tf.function
def flatten(inputs):
    """ flatten the non batch dimensions of a tensor. Works also with None as the batch dimension. """
    from tensorflow.python.framework import constant_op
    import functools
    import operator
    from tensorflow.python.ops import array_ops
    input_shape = inputs.shape
    non_batch_dims = input_shape[1:]
    last_dim = int(functools.reduce(operator.mul, non_batch_dims))
    flattened_shape = constant_op.constant([-1, last_dim])
    return array_ops.reshape(inputs, flattened_shape)


@tf.function
def get_pca_variance(data):
    """ calculate the eigenvalues of the covariance matrix """
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    # Finding the Eigen Values and Vectors for the data
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)
    eigen_values, eigen_vectors = tf.linalg.eigh(sigma)

    if data.shape[0] is None:
        eigen_values_normed = eigen_values[::-1]
    else:
        eigen_values_normed = eigen_values[::-1] / data.shape[0]
    return eigen_values_normed


@tf.function
def linear_fit(x_data, y_data):
    """ calculate the linear regression fit for a list of xy points. """
    x_mean = tf.reduce_mean(x_data)
    y_mean = tf.reduce_mean(y_data)
    b = tf.reduce_sum((x_data - x_mean) * (y_data - y_mean)) / tf.reduce_sum((x_data - x_mean) ** 2)
    a = y_mean - (b * x_mean)
    return a, b


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
