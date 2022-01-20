import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce, hostname
from dimension_regularisation.conv_sharing_off import Conv2DNew
from dimension_regularisation.dim_includes import command_line_parameters as p

import tensorflow_datasets as tfds
if hostname() == "richard-lassonde-linux":
    download_dir = Path(__file__).parent / "tensorflowdatasets"
else:
    download_dir = "/home/rgerum/scratch/tensorflowdatasets"


# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, p.dataset("cifar10")).load_data()


num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

cb = PlotAlpha(getOutputPath(p), x_train, batch_size=200, download_dir=download_dir)
if cb.started() and 1:
    model, initial_epoch = cb.load()
else:
    initial_epoch = 0

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=x_train.shape[1:]),
        keras.layers.Lambda(lambda x: x/255),

        keras.layers.Conv2D(p.conv1(32), 3, 2, activation='relu', kernel_initializer='he_uniform'),
        DimensionReg(p.reg1(0), p.reg1value(1.)),
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

history = model.fit(x_train, y_train, batch_size=200, epochs=50, validation_data=(x_test, y_test),
                    initial_epoch=initial_epoch,
                    callbacks=[cb]
)




#loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
