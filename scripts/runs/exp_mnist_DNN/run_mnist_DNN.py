import numpy as np
from tensorflow import keras
import tensorflow as tf
print(tf.version.VERSION)

from dimension_regularisation.dim_includes import getOutputPath
from dimension_regularisation.dimension_reg_layer import DimensionReg
from dimension_regularisation.dimensions_reg_layer_gamma import DimensionRegGammaWeights, DimensionRegGammaWeightsPreComputedBase, CalcEigenVectors
from dimension_regularisation.callbacks import SaveHistory, SlurmJobSubmitterStatus
from dimension_regularisation.robustness import get_robustness_metrics
from dimension_regularisation.attack_tf import get_attack_metrics
from dimension_regularisation.dim_includes import command_line_parameters as p

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, p.dataset("mnist")).load_data()
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

# get the number of classes
num_classes = np.max(y_test)+1
# convert
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# restart from previous checkpoint if it exists
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

#get_robustness_metrics
cb = SaveHistory(getOutputPath(p), additional_logs_callback=[get_attack_metrics("mnist", np.arange(0, 0.2, 0.01))])
if cb.started() and 0:
    model, initial_epoch = cb.load()
else:
    initial_epoch = 0

    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=x_train.shape[1:]),

        keras.layers.Flatten(),
        keras.layers.Dense(units=p.dense1(2000), activation='tanh'),
        DimensionRegGammaWeights(p.reg1(1.), p.reg1value(1)),
        #DimensionRegGammaWeightsPreComputedBase(p.reg1(1.), p.reg1value(1.)),
        tf.keras.layers.Dense(units=num_classes, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

#print(get_attack_metrics("mnist", np.arange(0, 0.2, 0.01))(model))
#exit()
#int(2000*1.5)
print(x_train.shape, x_test.shape)
# earlyStopping
history = model.fit(x_train, y_train, batch_size=2500, epochs=50, validation_data=(x_test, y_test),
                    initial_epoch=initial_epoch,
                    callbacks=[cb, CalcEigenVectors(x_train)]
)
