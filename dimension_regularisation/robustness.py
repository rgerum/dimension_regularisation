import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def mask_to_categorical(data):
    data_label = tf.one_hot(tf.cast(data["label"], tf.int32), 10)
    data_label = tf.cast(data_label, tf.float32)
    return data["image"], data_label


def robust_test(model, corruption, level, download_dir=None):
    ds = tfds.load(f'cifar10_corrupted/{corruption}_{level}', download_and_prepare_kwargs={"download_dir": download_dir})
    ds = ds["test"]
    res = model.evaluate(ds.batch(200).map(mask_to_categorical))
    return res[1]
