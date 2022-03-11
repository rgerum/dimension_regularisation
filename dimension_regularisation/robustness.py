from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

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

def robust_fgsm(model, strengths):
    from dimension_regularisation.dimension_reg_layer import DimensionReg
    from dimension_regularisation.dimensions_reg_layer_gamma import DimensionRegGammaWeights
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    model2 = tf.keras.models.Sequential([
        layer for layer in model.layers if not isinstance(layer, (tf.keras.layers.Lambda, DimensionRegGammaWeights, DimensionReg))
        ]
    )
    model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.build(x_train.shape)
    model.build(x_train.shape)
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def get_pertubation(input_image, input_label):
        im = tf.cast(input_image/255, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(im)
            prediction = model2(im)
            loss = loss_object(tf.one_hot(input_label, 10), prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, im)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        perturbations = signed_grad
        return perturbations.numpy()
    perturbations = get_pertubation(x_test, y_test[:, 0])
    alphas = []
    for strength in strengths:
        adversarial_image = (x_test + strength * perturbations*255).astype(np.uint8)
        alpha = model.evaluate(adversarial_image, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
        alphas.append(alpha)
    return alphas


if __name__ == "__main__":
    import tensorflow as tf
    from scripts.net_helpers import read_data
    data1 = read_data(
        r"../results/cedar_logs_expcifar4/iter-1_gamma-False_reg1-{reg1}_reg1value-1.0/",
        file_name="data.csv")
    for filename, d in data1.groupby("filename"):
        print(filename, d.reg1[0])
        model = tf.keras.models.load_model(filename.replace("data.csv", "model_save"))
        strengths = np.arange(0, 0.21, 0.01)
        import matplotlib.pyplot as plt
        plt.plot(strengths, robust_fgsm(model, strengths), label=d.reg1[0])
    plt.legend()