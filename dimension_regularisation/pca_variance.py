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