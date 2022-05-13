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
from dimension_regularisation.dim_includes import command_line_parameters as p, PCAreduce2


def main(dataset="mnist", dense1=1000,
         reg_strength=1., reg_target=1.,
         gamma=False,
         iter=0,
         class_count=None,
         pca_dim=None,
         epochs=200,
         output="logs/tmp600__"):

    # set the seed depending on the iteration
    tf.random.set_seed(iter*1234+1234)
    np.random.seed(iter*1234+1234)

    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = getattr(keras.datasets, dataset).load_data()
    x_train = x_train.astype(np.float32)/255
    x_test = x_test.astype(np.float32)/255

    # for cifar10 we need to squeeze (Nx1) for mnist it is already (N)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    if len(x_train.shape) == 3:
        x_train = x_train[..., None]
        x_test = x_test[..., None]

    # optionally reduce number of classes
    if class_count is not None:
        index = (y_train <= class_count)
        index2 = (y_test <= class_count)

        x_train = x_train[index]
        y_train = y_train[index]
        x_test = x_test[index2]
        y_test = y_test[index2]

    if pca_dim is not None:
        x_train, x_test = PCAreduce2(x_train, x_test, pca_dim)

    # get the number of classes
    num_classes = np.max(y_test)+1
    # convert
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # restart from previous checkpoint if it exists
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    #get_robustness_metrics
    cb = SaveHistory(getOutputPath(main, locals()), additional_logs_callback=[get_attack_metrics((x_test, y_test), np.arange(0, 0.2, 0.01))])
    if cb.started() and 0:
        model, initial_epoch = cb.load()
    else:
        initial_epoch = 0

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=x_train.shape[1:]),

            keras.layers.Conv2D(16, 3, activation='tanh'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(32, 3, activation='tanh'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=dense1, activation='tanh'),
            #DimensionReg(reg_strength, reg_target),
            #DimensionRegGammaWeights(reg_strength, reg_target),
            DimensionReg(reg_strength, reg_target) if gamma is False else
            DimensionRegGammaWeightsPreComputedBase(reg_strength, reg_target),
            tf.keras.layers.Dense(units=num_classes, activation='softmax'),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

    print(x_train.shape, x_test.shape)
    # earlyStopping
    history = model.fit(x_train, y_train, batch_size=6000, epochs=epochs, validation_data=(x_test, y_test),
                        initial_epoch=initial_epoch,
                        callbacks=[cb] # CalcEigenVectors(x_train)
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
