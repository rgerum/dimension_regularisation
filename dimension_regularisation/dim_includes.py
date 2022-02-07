import sys

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import time
from .robustness import robust_test


if 0:
    import matplotlib.pyplot as plt
    """
    data = []
    for file in Path("/home/richard/PycharmProjects/bird_data/docs/cropped").glob("*.png"):
        print(file)
        im = plt.imread(file)
        data.append(im.ravel())
    data = np.array(data)
    """
    data = np.load("../../caltech_birds/data (copy)/cropped_scaled.npy")
    data = data[:, :, 0, 0]
    #data = data.reshape([data.shape[0], -1])
    print(data.shape)
    print(data.shape, data.min(), data.max())
    print(getPCAVariance(data))

@tf.function
def getPCAVarianceWithEigenvetors(data):
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    # Finding the Eigen Values and Vectors for the data
    eigen_values, eigen_vectors = tf.linalg.eigh(tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1))

    return eigen_values[::-1] / data.shape[0], eigen_vectors

@tf.function
def getEstimatePCAVariance(data, eigen_vectors):
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)

    eigen_vectors_t = tf.transpose(eigen_vectors)
    eigen_values = tf.linalg.diag_part(tf.tensordot(eigen_vectors_t, tf.tensordot(sigma, eigen_vectors, 1), 1))

    return eigen_values[::-1] / data.shape[0]


if 0:
    data = x_train.reshape(x_train.shape[0], -1)
    import time
    t = time.time()
    eigen_values, eigen_vectors = getPCAVarianceWithEigenvetors(data)
    t2 = time.time()
    print(t2 - t)
    eigen_values2 = getEstimatePCAVariance(data, eigen_vectors)
    t2 = time.time()
    print(t2 - t)

    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)
    # Finding the Eigen Values and Vectors for the data
    eigen_values, eigen_vectors = tf.linalg.eigh(sigma)

    eigen_vectors_t = tf.transpose(eigen_vectors)
    eigen_values2 = tf.linalg.diag_part(tf.tensordot(eigen_vectors_t, tf.tensordot(sigma, eigen_vectors, 1), 1))
    exit()






@tf.function
def getAlphaXY(data, f=1):
    d = data#[..., 0]
    d = tf.reshape(d, (d.shape[0], -1))

    eigen_values = getPCAVariance(d)

    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    y = tf.math.log(eigen_values)
    x = tf.math.log(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype)*f)

    a, b = linear_fit(x[5:50], y[5:50])
    return x, y, a, b





class GetMeanStd(keras.layers.Layer):

    def __init__(self, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        if metric_name is None:
            metric_name = self.name#.replace("get_mean_std", "alpha")
        self.metric_name = metric_name

    def get_config(self):
        return {"metric_name": self.metric_name}

    def call(self, x):
        self.add_metric(tf.reduce_mean(x), self.metric_name+"_mean")
        self.add_metric(tf.math.reduce_std(x), self.metric_name+"_std")
        return x


from .dimension_reg_layer import get_pca_variance
class DimensionRegOutput(keras.layers.Layer):

    def call(self, x):
        if x.shape[0] == None:
            return x

        d = x#[..., 0]
        #d = d[::100, ::100, :5]
        d = tf.reshape(d, (d.shape[0], -1))
        #print(d.shape)

        eigen_values = get_pca_variance(d)

        eigen_values = tf.nn.relu(eigen_values) + 1e-8
        return eigen_values


def PCAreduce(x_train, x_test, pca_dims):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dims)
    pca.fit(x_train.reshape(x_train.shape[0], -1))
    x_train = pca.transform(x_train.reshape(x_train.shape[0], -1)).reshape((x_train.shape[0], pca_dims))
    x_test = pca.transform(x_test.reshape(x_test.shape[0], -1)).reshape((x_test.shape[0], pca_dims))
    return x_train, x_test




class GetAlphaLayer(keras.layers.Layer):

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
        print("layer", self.metric_name, x.shape)
        if x.shape[0] == None:
            return x
        print("x", x.shape, len(x.shape))
        x = tf.reshape(x, [x.shape[0], -1])
        if x.shape[1] > 1000:
            x = tf.gather(x, tf.random.uniform(shape=[1000], maxval=x.shape[1], dtype=tf.int32, seed=10), axis=1)

        #if len(x.shape) == 4:
        #    x = x[:, ::16, ::16, :]
        print("x", x.shape, len(x.shape))
        alpha = getAlpha(x)
        print("alpha", alpha)
        #self.add_metric(alpha, self.metric_name)
        #self.add_loss(tf.math.abs(alpha-self.target_value)*self.strength)
        return alpha


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


def getCommandLineArgs():
    index = 0
    data = {}
    while index < len(sys.argv):
        if sys.argv[index].startswith("--"):
            data[sys.argv[index][2:]] = sys.argv[index+1]
            index += 1
        index += 1
    return data


def getOutputPath(args):
    from datetime import datetime
    parts = [
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        getGitHash(),
    ]
    parts.extend([str(k) + "=" + str(v) for k, v in args._get_kwargs() if k != "output"])
    print("parts", parts)
    output = Path(args.filename_logs("logs/tmp3"))# / (" ".join(parts))
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    arguments = dict(datetime=parts[0], commit=parts[1], commitLong=getGitLongHash(), run_dir=os.getcwd())
    arguments.update(args._get_kwargs())
    print("arguments", arguments)

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
