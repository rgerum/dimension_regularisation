from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds


def hostname():
    import subprocess
    return subprocess.check_output(["hostname"]).strip().decode()


if hostname() == "richard-lassonde-linux":
    download_dir = Path(__file__).parent / ".." / "data" / "tensorflowdatasets"
else:
    download_dir = "/home/rgerum/scratch/tensorflowdatasets"


def get_robustness_metrics(model):
    logs = {}
    for mode in ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"]:
        for i in range(1, 6):
            logs[f"accuracy_{mode}_{i}"] = robust_test(model, mode, i, download_dir)
    return logs


@tf.function
def mask_to_categorical(data):
    data_label = tf.one_hot(tf.cast(data["label"], tf.int32), 10)
    data_label = tf.cast(data_label, tf.float32)
    return data["image"], data_label


def robust_test(model, corruption, level, download_dir=None):
    ds = tfds.load(f'cifar10_corrupted/{corruption}_{level}', download_and_prepare_kwargs={"download_dir": download_dir})
    ds = ds["test"]
    for layer in model.layers:
        layer.calc_alpha = False

    res = model.evaluate(ds.batch(200).map(mask_to_categorical))

    for layer in model.layers:
        layer.calc_alpha = True
    return res[1]
