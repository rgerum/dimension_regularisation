import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import time
from .robustness import robust_test


class PlotAlpha(keras.callbacks.Callback):
    def __init__(self, output, x_train, batch_size=1000, download_dir=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "data.csv"
        self.output2 = output / "alpha.csv"
        self.model_save = output / "model_save"
        self.state_output = output / "status.txt"
        self.x_train = x_train
        self.batch_size = batch_size

        self.download_dir = download_dir

        self.data = []
        self.alpha_data = []

    def started(self):
        if Path(self.output).exists():
            try:
                history = pd.read_csv(self.output)
                history.epoch.max()
                model = tf.keras.models.load_model("tmp_history")
                initial_epoch = int(history.epoch.max() + 1)
                data = [dict(history.iloc[i]) for i in range(len(history))]
                self.start_data = dict(model=model, initial_epoch=initial_epoch, data=data)
                return True
            except:
                return False
        return False

    def load(self):
        return self.start_data["model"], self.start_data["initial_epoch"]

    def on_epoch_end(self, epoch, logs={}):
        try:
            from slurm_job_submitter import set_job_status
            set_job_status(dict(epoch=epoch))
        except ModuleNotFoundError:
            pass

        for mode in ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"]:
            for i in range(1, 6):
                logs[f"accuracy_{mode}_{i}"] = robust_test(self.model, mode, i, self.download_dir)
        logs["epoch"] = epoch
        logs["time"] = time.time()
        self.data.append(logs)

        Path(self.model_save).mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_save)

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
        with Path(self.state_output).open("w") as fp:
            fp.write(f"{self.data[-1]['epoch']}\n")


    def on_train_end(self, logs=None):
        with Path(self.state_output).open("w") as fp:
            fp.write(f"{self.data[-1]['epoch']} done\n")


@tf.function
def getPCAVariance(data):
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    # Finding the Eigen Values and Vectors for the data
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)
    eigen_values, eigen_vectors = tf.linalg.eigh(sigma)

    if data.shape[0] is None:
        eigen_values_normed = eigen_values[::-1]
    else:
        eigen_values_normed = eigen_values[::-1] / data.shape[0]
    return eigen_values_normed

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

@tf.function
def getAlphaRegularizer(data, alpha=1.):
    lambdas = getPCAVariance(data)
    lambdas = tf.cast(lambdas, tf.float32)
    N = lambdas.shape[0]
    N = 100
    tau = 5
    lambdas = lambdas[tau:N]
    kappa = lambdas[0] * tf.math.pow(float(tau), alpha)
    gammas = kappa * tf.math.pow(tf.range(tau, N, dtype=tf.float32), -alpha)
    loss = 1/N * tf.reduce_sum((lambdas/gammas - 1) ** 2 + tf.nn.relu(lambdas/gammas - 1))
    return loss


class DimensionRegGammaWeights(keras.layers.Layer):

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
        x2 = flatten(x)
        if x2.shape[1] > 10000:
            x2 = tf.gather(x2, tf.random.uniform(shape=[10000], maxval=x2.shape[1], dtype=tf.int32, seed=10), axis=1)

        loss = getAlphaRegularizer(x2, self.target_value)*self.strength
        self.add_loss(loss)
        alpha = getAlpha(x2)
        self.add_metric(alpha, self.metric_name)
        self.add_metric(loss, self.metric_name+"_loss")
        return x

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
def linear_fit(x_data, y_data):
    x_mean = tf.reduce_mean(x_data)
    y_mean = tf.reduce_mean(y_data)
    b = tf.reduce_sum((x_data - x_mean) * (y_data - y_mean)) / tf.reduce_sum((x_data - x_mean) ** 2)
    a = y_mean - (b * x_mean)
    return a, b


@tf.function
def getAlpha(data, f=1):
    d = data#[..., 0]
    d = flatten(d)#tf.reshape(d, (d.shape[0], -1))

    eigen_values = getPCAVariance(d)

    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    y = tf.math.log(eigen_values)
    x = tf.math.log(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype)*f)

    a, b = linear_fit(x[5:50], y[5:50])
    return -b

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


@tf.function
def flatten(inputs):
    from tensorflow.python.framework import constant_op
    import functools
    import operator
    from tensorflow.python.ops import array_ops
    input_shape = inputs.shape
    non_batch_dims = input_shape[1:]
    last_dim = int(functools.reduce(operator.mul, non_batch_dims))
    flattened_shape = constant_op.constant([-1, last_dim])
    return array_ops.reshape(inputs, flattened_shape)


class DimensionReg(keras.layers.Layer):

    def __init__(self, strength=0.01, target_value=1, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg", "alpha")
        self.metric_name = metric_name
        self.calc_alpha = True

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name}

    def call(self, x):
        # flatten the non-batch dimensions
        x2 = flatten(x)
        # if the array is too big create a random sampled sub set
        if x2.shape[1] > 10000:
            x2 = tf.gather(x2, tf.random.uniform(shape=[10000], maxval=x2.shape[1], dtype=tf.int32, seed=10), axis=1)
        # get the alpha value
        if self.calc_alpha:
            alpha = getAlpha(x2)
        else:
            alpha = 0
        # record it as a metric
        self.add_metric(alpha, self.metric_name)
        # calculate the loss and add is a metric
        loss = tf.math.abs(alpha-self.target_value)*self.strength
        self.add_metric(loss, self.metric_name+"_loss")
        self.add_loss(loss)

        # return the unaltered x
        return x



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


class DimensionRegOutput(keras.layers.Layer):

    def call(self, x):
        if x.shape[0] == None:
            return x

        d = x#[..., 0]
        #d = d[::100, ::100, :5]
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


def hostname():
    import subprocess
    return subprocess.check_output(["hostname"]).strip()


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

    output = Path(args.output("logs/tmp3"))# / (" ".join(parts))
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
