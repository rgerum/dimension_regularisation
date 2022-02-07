import numpy as np
from tensorflow import keras

from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce

import argparse
parser = argparse.ArgumentParser(description='Settings to run dimensions')
parser.add_argument('--dataset', default="cifar10", type=str, help='the dataset name')
parser.add_argument('--pca_reduce', default=0, type=int, help='the v batch size')
parser.add_argument('--dense1', default=512, type=int, help='units in first hidden layer')
parser.add_argument('--reg1', default=0.1, type=float, help='units in first hidden layer')
parser.add_argument('--dense2', default=256, type=int, help='units in first hidden layer')
parser.add_argument('--reg2', default=0, type=float, help='units in first hidden layer')
parser.add_argument('--output', default='logs', help='the output directory')
args = parser.parse_args()

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, args.dataset).load_data()

if args.pca_reduce:
    PCAreduce(x_train, x_test, args.pca_reduce)

num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=x_train.shape[1:]),
    keras.layers.Flatten(),
    keras.layers.Lambda(lambda x: x/255),
    keras.layers.Dense(args.dense1, activation='relu'),
    DimensionReg(args.reg1, 1, "alpha_1"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(args.dense2, activation='relu'),
    DimensionReg(args.reg2, 1, "alpha_2"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=num_classes, activation='softmax'),
])
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=1000, epochs=500, validation_data=(x_test, y_test),
                    callbacks=[PlotAlpha(getOutputPath(args), x_train)])
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
