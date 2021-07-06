import numpy as np
from tensorflow import keras

from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce
from dimension_regularisation.conv_sharing_off import Conv2DNew

import argparse
parser = argparse.ArgumentParser(description='Settings to run dimensions')
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--pca_reduce', default=0, type=int)
parser.add_argument('--conv1', default=32, type=int)
parser.add_argument('--reg1', default=0, type=float)
parser.add_argument('--conv2', default=32, type=int)
parser.add_argument('--reg2', default=0, type=float)
parser.add_argument('--dense1', default=256, type=float)
parser.add_argument('--reg3', default=0, type=float)
parser.add_argument('--weight_share', default=False, type=bool)
parser.add_argument('--output', default='logs', help='the output directory')
args = parser.parse_args()

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, args.dataset).load_data()

if args.pca_reduce:
    PCAreduce(x_train, x_test, args.pca_reduce)

num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if args.weight_share is True:
    Conv2DNew = keras.layers.Conv2D

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=x_train.shape[1:]),
    keras.layers.Lambda(lambda x: x/255),

    Conv2DNew(args.conv1, 3, weights_shared=True, activation='relu', kernel_initializer='he_uniform'),#, padding='same'),
#    keras.layers.Conv2D(args.conv1, 3, activation='relu', kernel_initializer='he_uniform', padding='same'),
    keras.layers.MaxPool2D(2),
    DimensionReg(args.reg1, 1),
    keras.layers.Dropout(0.5),

    Conv2DNew(args.conv2, 3, weights_shared=True, activation='relu', kernel_initializer='he_uniform'),#, padding='same'),
#    keras.layers.Conv2D(args.conv2, 3, activation='relu', kernel_initializer='he_uniform', padding='same'),
    keras.layers.MaxPool2D(2),
    DimensionReg(args.reg2, 1),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(units=args.dense1, activation='relu'),
    DimensionReg(args.reg3, 1),
    keras.layers.Dense(units=num_classes, activation='softmax'),
])
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if args.weight_share is False:
    for layer in model.layers:
        if isinstance(layer, Conv2DNew):
            layer.weights_shared = False
            layer._update_conv_function()

history = model.fit(x_train, y_train, batch_size=1000, epochs=500, validation_data=(x_test, y_test),
                    callbacks=[PlotAlpha(getOutputPath(args), x_train, batch_size=100)])
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
