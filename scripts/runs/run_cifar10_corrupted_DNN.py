import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from dimension_regularisation.dim_includes import getOutputPath
from dimension_regularisation.dimension_reg_layer import DimensionReg
from dimension_regularisation.callbacks import SaveHistory, SlurmJobSubmitterStatus
from dimension_regularisation.robustness import get_robustness_metrics
from dimension_regularisation.dim_includes import command_line_parameters as p


# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, p.dataset("cifar10")).load_data()


num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

cb = SaveHistory(getOutputPath(p), additional_logs_callback=get_robustness_metrics)
if cb.started() and 0:
    model, initial_epoch = cb.load()
else:
    initial_epoch = 0

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=x_train.shape[1:]),
        keras.layers.Lambda(lambda x: x/255),

        keras.layers.Flatten(),
        keras.layers.Dense(units=p.dense1(1000), activation='relu'),
        DimensionReg(p.reg1(1.), p.reg1value(0.6)),
        keras.layers.Dense(units=num_classes, activation='softmax'),
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

history = model.fit(x_train, y_train, batch_size=200, epochs=200, validation_data=(x_test, y_test),
                    initial_epoch=initial_epoch,
                    callbacks=[cb, SlurmJobSubmitterStatus()]
)




#loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
