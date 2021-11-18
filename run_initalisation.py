import numpy as np
from tensorflow import keras

from dimension_regularisation.dim_includes import DimensionReg, PlotAlpha, getOutputPath, PCAreduce, GetMeanStd, getAlpha
import tensorflow as tf

from dimension_regularisation.dim_includes import command_line_parameters as p
##

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = getattr(keras.datasets, p.dataset("cifar10")).load_data()

#if args.pca_reduce:
#    PCAreduce(x_train, x_test, args.pca_reduce)

num_classes = np.max(y_test)+1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train0 = x_train

##
x_train = x_train0-np.mean(x_train0)
x_train = x_train/np.std(x_train)

#x_train = x_train0/255


##

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=x_train.shape[1:]),
    keras.layers.Flatten(),
    #keras.layers.Lambda(lambda x: x/255-128),

    GetMeanStd("get_mean_std_0"),
    DimensionReg(p.reg0(0), 1, "alpha_0"),

    keras.layers.Dense(p.dense1(512), #activation='relu',
#                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
#                       kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
#                       bias_initializer=keras.initializers.RandomNormal(stddev=0.2, mean=0),
                       ),
    #Normal(),
#    keras.layers.Lambda(lambda x: x+1),
    GetMeanStd("get_mean_std_1"),
    keras.layers.Lambda(lambda x: x+1),
    keras.layers.ReLU(),
    #keras.layers.Lambda(tf.keras.activations.sigmoid),

    DimensionReg(p.reg1(0), 1, "alpha_1"),
    #keras.layers.Dropout(0.5),

    keras.layers.Dense(p.dense2(512), #activation='relu',
                       #kernel_initializer=keras.initializers.RandomNormal(stddev=0.033),
                       ),
    #Normal(),
    GetMeanStd("get_mean_std_2"),
    keras.layers.ReLU(),
    DimensionReg(p.reg2(0), 1, "alpha_2"),

    keras.layers.Dense(p.dense3(512),  # activation='relu',
                       #kernel_initializer=keras.initializers.RandomNormal(stddev=0.033),
                       ),
    # Normal(),
    GetMeanStd("get_mean_std_3"),
    keras.layers.ReLU(),
    DimensionReg(p.reg3(0), 1, "alpha_3"),

    keras.layers.Dense(p.dense4(512),  # activation='relu',
                       #kernel_initializer=keras.initializers.RandomNormal(stddev=0.033),
                       ),
    # Normal(),
    GetMeanStd("get_mean_std_4"),
    keras.layers.ReLU(),
    DimensionReg(p.reg4(0), 1, "alpha_4"),

    #    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=num_classes, activation='softmax'),
    GetMeanStd(),
    keras.layers.Softmax()
])

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

results = np.array(model.evaluate(x_train[:500], y_train[:500], batch_size=500, return_dict=True))[()]
print(results)
for i in range(0, 100):
    if f"alpha_{i}" not in results:
        continue
    print(i, results[f"get_mean_std_{i}_mean"], results[f"get_mean_std_{i}_std"], results[f"alpha_{i}"])


##

#exit()

history = model.fit(x_train, y_train, batch_size=200, epochs=500, validation_data=(x_test, y_test),
                    callbacks=[PlotAlpha(getOutputPath(p), x_train, batch_size=200)])
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
