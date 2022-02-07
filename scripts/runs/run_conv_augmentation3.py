import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce
from dimension_regularisation.conv_sharing_off import Conv2DNew
from dimension_regularisation.dim_includes import command_line_parameters as p

import tensorflow_datasets as tfds
ds = tfds.load('cifar10_corrupted', download_and_prepare_kwargs={"download_dir": Path(__file__).parent / "tensorflowdatasets"})
ds = ds["test"]

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, p.dataset("cifar10")).load_data()

if p.pca_reduce(0):
    PCAreduce(x_train, x_test, p.pca_reduce())

num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

cb = PlotAlpha(getOutputPath(p), x_train, batch_size=200)
if cb.started() and 0:
    model, initial_epoch = cb.load()
else:
    initial_epoch = 0

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=x_train.shape[1:]),
        keras.layers.Lambda(lambda x: x/255),

        keras.layers.Conv2D(p.conv1(32), 3, 2, activation='relu', kernel_initializer='he_uniform'),
        #DimensionReg(p.reg1(0.1), p.reg1value(1.)),
        keras.layers.Conv2D(p.conv2(32), 3, 2, activation='relu', kernel_initializer='he_uniform'),
#        DimensionReg(p.reg2(0.), p.reg2value(0.)),
        keras.layers.Conv2D(p.conv3(32), 3, 1, activation='relu', kernel_initializer='he_uniform'),
#        DimensionReg(p.reg3(0.), p.reg3value(0.)),

        keras.layers.Flatten(),
        keras.layers.Dense(units=p.dense1(128), activation='relu'),
#        DimensionReg(p.reg4(0.), p.reg4value(0.)),
        keras.layers.Dense(units=num_classes, activation='softmax'),
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

history = model.fit(x_train, y_train, batch_size=200, epochs=22, validation_data=(x_test, y_test),
                    initial_epoch=initial_epoch,
                    callbacks=[cb]
)

@tf.function
def mask_to_categorical(data):
    data_label = tf.one_hot(tf.cast(data["label"], tf.int32), num_classes)
    data_label = tf.cast(data_label, tf.float32)
    return data["image"], data_label


results = []
for i in range(1, 6):
    ds = tfds.load(f'cifar10_corrupted/brightness_{i}', download_and_prepare_kwargs={"download_dir": Path(__file__).parent / "tensorflowdatasets"})
    ds = ds["test"]
    res = model.evaluate(ds.batch(200).map(mask_to_categorical))
    results.append(res)
print(results)


#loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
