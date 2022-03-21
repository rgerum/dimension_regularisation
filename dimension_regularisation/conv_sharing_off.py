import numpy as np
import tensorflow as tf
from tensorflow import keras
from dimension_regularisation.pca_variance import flatten

@tf.function
def convolution_unlinked(x, kernel, strides):
    rank = len(x.shape)-2
    y_shapes = [x.shape[0]] + [(x.shape[1+i] - kernel.shape[i])//strides[i] + 1 for i in range(rank)] + [kernel.shape[rank+rank+1]]
    y_sum = np.prod([(x.shape[1+i] - kernel.shape[i])//strides[i] + 1 for i in range(rank)])
    y2 = tf.TensorArray(x.dtype, y_sum)

    total_index = 0
    def summed(index, xtupel, ytupel, pos, y2, total_index):
        if index == rank:
            multplied = x[xtupel + (slice(None), None)] * kernel[(None,)+(slice(None),)*rank+pos+(slice(None),)*2]
            sumed = tf.reduce_sum(multplied, np.arange(1, rank+2))
            y2 = y2.write(total_index, sumed)
            total_index += 1
        else:
            for i in range(y_shapes[index+1]):
                y2, total_index = summed(index+1, xtupel + (slice(i, i + kernel.shape[index]*strides[index], strides[index]),), ytupel + (slice(i, i + 1),), pos+(i,), y2, total_index)
        return y2, total_index
    y2, total_index = summed(0, (slice(None),), (slice(None),), (), y2, total_index)
    y_shapes = [(x.shape[1 + i] - kernel.shape[i])//strides[i] + 1 for i in range(rank)][::] + [x.shape[0]] + [
        kernel.shape[rank + rank + 1]]
    y_shapes = tuple([tf.shape(x)[0]] + [(x.shape[1 + i] - kernel.shape[i]) // strides[i] + 1 for i in range(rank)][::] + [
        kernel.shape[rank + rank + 1]])
    order = [rank] + list(range(rank))[::] + [len(y_shapes)-1]

    stack = y2.stack()
    stack_transposed = tf.transpose(stack, [1, 0, 2])
    stack_transposed_reshaped = tf.reshape(stack_transposed, y_shapes)
    return stack_transposed_reshaped


def kernel_unlink(x_shape, kernel):
    rank = len(x_shape) - 2
    kernel2 = tf.tile(kernel[(slice(None),)*rank+(None,)*rank], [1]*rank+[x_shape[i+1] - kernel.shape[i] + 1 for i in range(rank)] + [1, 1])
    return kernel2

from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.framework import tensor_shape
class ConvNew(Conv):
    def __init__(self, *args, weights_shared=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_shared = weights_shared

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.input_shape_ = input_shape
        add = [(input_shape[i + 1] - self.kernel_size[i])//self.strides[i] + 1 for i in range(self.rank)]
        kernel_size = self.kernel_size
        self.kernel_size = self.kernel_size + tuple(add)

        super().build(input_shape)
        self.kernel_size = kernel_size
        self._convolution_op_old = self._convolution_op
        self._update_conv_function()

        weights = self.get_weights()
        print("initial", weights[0].shape)

    def disable_weightshare(self):
        if self.weights_shared is False:
            return
        self.weights_shared = False
        weights = self.get_weights()
        weights[0] = kernel_unlink(self.input_shape_, weights[0][(slice(None),) * self.rank + (1,) * self.rank])
        print("disabled", weights[0].shape)
        self.set_weights(weights)

        self._update_conv_function()

    def enable_weightshare(self):
        if self.weights_shared is True:
            return
        self.weights_shared = True
        self._update_conv_function()

    def _update_conv_function(self):
        if self.weights_shared is True:
            def conv(x, kernel):
                return self._convolution_op_old(x, kernel[(slice(None),) * self.rank + (1,) * self.rank])

            self._convolution_op = conv
        else:
            def conv(x, kernel):
                return convolution_unlinked(x, kernel, self.strides)

            self._convolution_op = conv


class Conv1DNew(ConvNew):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

class Conv2DNew(ConvNew):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

class Conv3DNew(ConvNew):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

##
from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import conv_utils
class FreeConv2D(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=5, strides=2, activation=None, **kwargs):
        #filters,
        #kernel_size,
        #strides = (1, 1),
        super().__init__(**kwargs)
        self.units = filters
        rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')

        self.activation = activations.get(activation)

    def build(self, input_shape):
        h, w, c = input_shape[-3], input_shape[-2], input_shape[-1]
        u = self.units
        k1, k2 = self.kernel_size
        s1, s2 = self.strides
        b = 2

        i1 = np.tile(np.arange(np.ceil((h - k1 + 1) / s1) * np.ceil((w - k2 + 1) / s2) * u)[:, None],
                     (1, k1 * k2 * c)).ravel()
        i2 = np.lib.stride_tricks.as_strided(np.arange(w * h * c, dtype=np.uint16), shape=(
        int(np.ceil((h - k1 + 1) / s1)), int(np.ceil((w - k2 + 1) / s2)), u, k1, k2 * c),
                                             strides=(b * w * c * s1, b * c * s2, b * 0, b * w * c, b * 1)).flatten()

        self.toep_indices = np.array((i1, i2)).T
        self.output_shape2 = (int(np.ceil((h - k1 + 1) / s1)), int(np.ceil((w - k2 + 1) / s2)), u)
        self.input_shape2 = (h, w, c)
        self.toepl_shape = [(h*w*c), int(np.ceil((h - k1 + 1) / s1))*int(np.ceil((w - k2 + 1) / s2))*u]

        self.w = self.add_weight(
            shape=(int(np.ceil((h - k1 + 1) / s1)), int(np.ceil((w - k2 + 1) / s2)), u, k1, k2, c),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(int(np.ceil((h - k1 + 1) / s1)), int(np.ceil((w - k2 + 1) / s2)), u), initializer="random_normal", trainable=True
        )

    def call(self, x):
        shape = x.shape
        x = flatten(x)

        t = tf.sparse.SparseTensor(self.toep_indices, tf.reshape(self.w, [-1]), dense_shape=self.toepl_shape[::-1])
        x = tf.transpose(tf.sparse.sparse_dense_matmul(tf.cast(t, tf.float32), tf.transpose(x)))

        outputs = tf.reshape(x, [-1, self.output_shape2[0], self.output_shape2[1], self.output_shape2[2]]) + self.b
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
