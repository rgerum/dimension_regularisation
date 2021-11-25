import numpy as np
from tensorflow import keras

from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce, DimensionRegGammaWeights
from dimension_regularisation.conv_sharing_off import Conv2DNew
from dimension_regularisation.dim_includes import command_line_parameters as p


# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, p.dataset("cifar10")).load_data()

if p.pca_reduce(0):
    PCAreduce(x_train, x_test, p.pca_reduce())

num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if p.weight_share(True) is True:
    Conv2DNew = keras.layers.Conv2D

if p.augmentation(True) is True:
    augmentation = keras.models.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        #keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ], name="augmentation")
else:
    augmentation = keras.models.Sequential([], name="no augmentation")

p.reg_type("gamma")
def RegLayer(reg, regvalue):
    if p.reg_type() == "gamma":
        return DimensionRegGammaWeights(reg, regvalue)
    else:
        return DimensionReg(reg, regvalue)

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=x_train.shape[1:]),
    keras.layers.Lambda(lambda x: x/255),

    augmentation,

    Conv2DNew(p.conv1(32), 5, 2, activation='relu', kernel_initializer='he_uniform'),
    RegLayer(p.reg1(1.), p.reg1value(1.)),
    #DimensionReg(p.reg1(0.), p.reg1value(1.)),

    Conv2DNew(p.conv2(64), 5, 2, activation='relu', kernel_initializer='he_uniform'),
    RegLayer(p.reg2(1.), p.reg2value(1.)),
    #DimensionRegGammaWeights(p.reg2(0.), p.reg2value(0.)),

    Conv2DNew(p.conv3(128), 3, 1, activation='relu', kernel_initializer='he_uniform'),
    RegLayer(p.reg3(1.), p.reg3value(1.)),
    #DimensionRegGammaWeights(p.reg3(0.), p.reg3value(0.)),

    keras.layers.Flatten(),
    keras.layers.Dense(units=p.dense1(1024), activation='relu'),
    RegLayer(p.reg4(1.), p.reg4value(1.)),
    #DimensionReg(p.reg4(0.), p.reg4value(0.)),
    keras.layers.Dense(units=num_classes, activation='softmax'),
])
import tensorflow as tf
cifar100_superclass_mapping = tf.constant([4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13], dtype=tf.int32)
def superclass_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.gather(cifar100_superclass_mapping, y_pred)
    y_true = tf.gather(cifar100_superclass_mapping, y_true)
    return tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if p.weight_share() is False:
    for layer in model.layers:
        if isinstance(layer, Conv2DNew):
            layer.weights_shared = False
            layer._update_conv_function()

print("test")
output = model(x_train[:200])
print("test done")
print(output.shape)


path = getOutputPath(p)#+"_"+p.reg_type()
history = model.fit(x_train, y_train, batch_size=500, epochs=100, validation_data=(x_test, y_test),
                    callbacks=[PlotAlpha(path, x_train, batch_size=500)])
from pathlib import Path
model.save(Path(path) / "model.h5")
#loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
