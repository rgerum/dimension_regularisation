import numpy as np
import tensorflow as tf
from tensorflow import keras


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
    order = [rank] + list(range(rank))[::] + [len(y_shapes)-1]
    return tf.transpose(tf.reshape(y2.stack(), y_shapes), order)


def kernel_unlink(x_shape, kernel):
    rank = len(x_shape) - 2
    kernel2 = tf.tile(kernel[(slice(None),)*rank+(None,)*rank], [1]*rank+[x_shape[i+1] - kernel.shape[i] + 1 for i in range(rank)] + [1, 1])
    return kernel2

from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.framework import tensor_shape
class ConvNew(Conv):
    def __init__(self, *args, weights_shared=True, **kwargs):
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

if __name__ == "__main__":
    if 0:
        # Get the dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # convert it to one-hot encoding
        num_classes = np.max(y_test)+1
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # build the model
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=x_train.shape[1:]),
            keras.layers.Lambda(lambda x: x/255),
            Conv2DNew(10, 5, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(units=num_classes, activation='softmax'),
        ])
        # compile it
        model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # train some time with linked weights
        history = model.fit(x_train, y_train, batch_size=1000, epochs=5, validation_data=(x_test, y_test))

        # disable the weight sharing and compile again
        model.layers[1].disable_weightshare()
        model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

        # fit again without shared weights
        history = model.fit(x_train, y_train, batch_size=1000, epochs=20, validation_data=(x_test, y_test))

        # plot the weights
        kernel = model.layers[1].weights[0]
        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.imshow(kernel[:, :, 0, 0, 0, 0])
        plt.subplot(222)
        plt.imshow(kernel[:, :, 10, 0, 0, 0])
        plt.subplot(223)
        plt.imshow(kernel[:, :, 0, 10, 0, 0])
        plt.subplot(224)
        plt.imshow(kernel[:, :, 10, 10, 0, 0])
        plt.show()
    else:
        # Get the dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # convert it to one-hot encoding
        num_classes = np.max(y_test) + 1
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # build the model
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=x_train.shape[1:]),
            keras.layers.Lambda(lambda x: x / 255),
            Conv2DNew(10, 5, weights_shared=True, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(units=num_classes, activation='softmax'),
        ])
        print("---------------")

        model.layers[1].weights_shared = False
        model.layers[1]._update_conv_function()

        # compile it
        model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # train some time with linked weights
        history = model.fit(x_train, y_train, batch_size=1000, epochs=25, validation_data=(x_test, y_test))

        # plot the weights
        kernel = model.layers[1].weights[0]
        import matplotlib.pyplot as plt

        plt.subplot(221)
        plt.imshow(kernel[:, :, 0, 0, 0, 0])
        plt.subplot(222)
        plt.imshow(kernel[:, :, 10, 0, 0, 0])
        plt.subplot(223)
        plt.imshow(kernel[:, :, 0, 10, 0, 0])
        plt.subplot(224)
        plt.imshow(kernel[:, :, 10, 10, 0, 0])
        plt.show()
