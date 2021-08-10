import numpy as np
from tensorflow import keras

from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce
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

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=x_train.shape[1:]),
    keras.layers.Lambda(lambda x: x/255),

    augmentation,

    keras.layers.Conv2D(p.conv1(32), 3, activation='relu', kernel_initializer='he_uniform', padding='same'),
    keras.layers.MaxPool2D(2),
    DimensionReg(p.reg1(0.), p.reg1value(0.)),

    keras.layers.Conv2D(p.conv2(64), 3, activation='relu', kernel_initializer='he_uniform', padding='same'),
    keras.layers.MaxPool2D(2),
    DimensionReg(p.reg2(0.), p.reg2value(0.)),

    Conv2DNew(p.conv3(128), 3, activation='relu', kernel_initializer='he_uniform', padding='same'),
    keras.layers.MaxPool2D(2),
    DimensionReg(p.reg3(0.), p.reg3value(0.)),

    keras.layers.Flatten(),
    keras.layers.Dense(units=p.dense1(256), activation='relu'),
    DimensionReg(p.reg4(0.), p.reg4value(0.)),
    keras.layers.Dense(units=p.dense2(128), activation='relu'),
    DimensionReg(p.reg5(0.), p.reg5value(0.)),
    keras.layers.Dense(units=num_classes, activation='softmax'),
])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if p.weight_share() is False:
    for layer in model.layers:
        if isinstance(layer, Conv2DNew):
            layer.weights_shared = False
            layer._update_conv_function()

history = model.fit(x_train, y_train, batch_size=100, epochs=500, validation_data=(x_test, y_test),
                    callbacks=[PlotAlpha(getOutputPath(p), x_train, batch_size=100)])
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
