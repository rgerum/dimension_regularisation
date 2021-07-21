import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import time


class PlotAlpha(keras.callbacks.Callback):
    def __init__(self, output, x_train, batch_size=1000):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "data.csv"
        self.output2 = output / "alpha.csv"
        self.x_train = x_train
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.data = []
        self.alpha_data = []

    def on_epoch_end(self, epoch, logs={}):
        logs["epoch"] = epoch
        logs["time"] = time.time()
        self.data.append(logs)
        eigen_values_list = []
        names = []
        for i in range(100):
            j = 0
            model2 = keras.models.Sequential()
            for layer in self.model.layers:
                if isinstance(layer, DimensionReg):
                    if j == i:
                        model2.add(DimensionRegOutput())
                        name = layer.metric_name
                        names.append(layer.metric_name)
                        break
                    else:
                        j += 1
                if isinstance(layer, keras.layers.Dropout):
                    rate = layer.rate
                    model2.add(keras.layers.Lambda(lambda x: tf.nn.dropout(x, rate=rate)))
                else:
                    model2.add(layer)
            else:
                break
            eigen_values = model2(self.x_train[:self.batch_size]).numpy()
            eigen_values_list.append(eigen_values)
            self.alpha_data.extend([dict(epoch=epoch, name=name, value=x) for x in eigen_values])
        while True:
            try:
                pd.DataFrame(self.data).to_csv(self.output, index=False)
            except FileNotFoundError as err:
                print(err)
                self.output.parent.mkdir(parents=True, exist_ok=True)
            else:
                break
        while True:
            try:
                pd.DataFrame(self.alpha_data).to_csv(self.output2, index=False)
            except FileNotFoundError as err:
                print(err)
                self.output2.parent.mkdir(parents=True, exist_ok=True)
            else:
                break


@tf.function
def getPCAVariance(data):
    normalized_data = data - tf.reduce_mean(data, axis=0)
    # Finding the Eigen Values and Vectors for the data
    eigen_values, eigen_vectors = tf.linalg.eigh(tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1))

    return eigen_values[::-1] / data.shape[0]


@tf.function
def linear_fit(x_data, y_data):
    x_mean = tf.reduce_mean(x_data)
    y_mean = tf.reduce_mean(y_data)
    b = tf.reduce_sum((x_data - x_mean) * (y_data - y_mean)) / tf.reduce_sum((x_data - x_mean) ** 2)
    a = y_mean - (b * x_mean)
    return a, b


@tf.function
def getAlpha(data):
    d = data#[..., 0]
    d = tf.reshape(d, (d.shape[0], -1))
    print(d.shape)
    eigen_values = getPCAVariance(d)

    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    y = tf.math.log(eigen_values)
    x = tf.math.log(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype))

    a, b = linear_fit(x[5:50], y[5:50])
    return -b


class DimensionReg(keras.layers.Layer):

    def __init__(self, strength=0.01, target_value=1, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg", "alpha")
        self.metric_name = metric_name

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name}

    def call(self, x):
        if x.shape[0] == None:
            return x

        alpha = getAlpha(x)
        self.add_metric(alpha, self.metric_name)
        self.add_loss(tf.math.abs(alpha-self.target_value)*self.strength)
        return x


class DimensionRegOutput(keras.layers.Layer):

    def call(self, x):
        if x.shape[0] == None:
            return x

        d = x#[..., 0]
        d = tf.reshape(d, (d.shape[0], -1))
        #print(d.shape)

        eigen_values = getPCAVariance(d)

        eigen_values = tf.nn.relu(eigen_values) + 1e-8
        return eigen_values


def PCAreduce(x_train, x_test, pca_dims):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dims)
    pca.fit(x_train.reshape(x_train.shape[0], -1))
    x_train = pca.transform(x_train.reshape(x_train.shape[0], -1)).reshape((x_train.shape[0], pca_dims))
    x_test = pca.transform(x_test.reshape(x_test.shape[0], -1)).reshape((x_test.shape[0], pca_dims))
    return x_train, x_test




def getGitHash():
    import subprocess
    try:
        short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        short_hash = str(short_hash, "utf-8").strip()
        return short_hash
    except subprocess.CalledProcessError:
        return ""

def getGitLongHash():
    import subprocess
    try:
        short_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        short_hash = str(short_hash, "utf-8").strip()
        return short_hash
    except subprocess.CalledProcessError:
        return ""


def getOutputPath(args):
    from datetime import datetime
    parts = [
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        getGitHash(),
    ]
    parts.extend([str(k) + "=" + str(v) for k, v in args._get_kwargs() if k != "output"])

    output = Path(args.output("logs")) / (" ".join(parts))
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    arguments = dict(datetime=parts[0], commit=parts[1], commitLong=getGitLongHash(), run_dir=os.getcwd())
    arguments.update(args._get_kwargs())
    with open(output / "arguments.yaml", "w") as fp:
        yaml.dump(arguments, fp)
    print("OUTPUT_PATH=\""+str(output)+"\"")
    return output


def get_parameter(parameter_name, default_value):
    import sys
    # parameter needs to have a default value
    if default_value is None:
        raise ValueError(f"No default value defined for {parameter_name}.")
    # try to find the value in the sys args
    for i, name in enumerate(sys.argv[:-1]):
        # did we find it? cast to desired value
        if name == "--"+parameter_name:
            if type(default_value) == bool:
                if sys.argv[i+1] == "False":
                    return False
            return type(default_value)(sys.argv[i+1])
    # if not return default
    return default_value

all_parameters = {}
def parameters(name, default_value):
    all_parameters[name] = get_parameter(name, default_value)
    return all_parameters[name]

class CommandLineParameters:
    all_parameters = {}
    def __init__(self):
        import sys
        print(" ".join(sys.argv))

    def parameters(self, name, default_value=None):
        if name in self.all_parameters:
            if default_value is not None:
                raise ValueError(f"Default value for {name} defined twice.")
            return self.all_parameters[name]
        self.all_parameters[name] = get_parameter(name, default_value)
        return self.all_parameters[name]

    def __getattr__(self, item):
        def func(x=None):
            return self.parameters(item, x)
        return func

    def __call__(self, item, x=None):
        return self.parameters(item, x)

    def _get_kwargs(self):
        return self.all_parameters.items()

command_line_parameters = CommandLineParameters()
