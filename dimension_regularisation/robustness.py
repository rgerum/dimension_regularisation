from pathlib import Path

import pandas as pd
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

##
def robust_fgsm(model):
    from dimension_regularisation.dimension_reg_layer import DimensionReg
    from dimension_regularisation.dimensions_reg_layer_gamma import DimensionRegGammaWeights
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = x_test.astype(np.float32)/255
    x_train = x_train.astype(np.float32)/255
    y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

    #y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)
    model2 = tf.keras.models.Sequential([
        layer for layer in model.layers if not (layer.name.startswith("lambda") or
                                                layer.name.startswith("dimension_reg") or
                                                layer.name.startswith("dimension_reg_gamma_weights"))
    ]
    )
    model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.build(x_train.shape)
    model.build(x_train.shape)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    accuracy_object = tf.keras.metrics.Accuracy()

    def get_perturbation(input_image, input_label):
        im = tf.cast(input_image, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(im)
            prediction = model2(im)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, im)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        perturbations = signed_grad
        return perturbations.numpy()

    def run(strengths):
        perturbations = get_perturbation(x_test, y_test_categorical)
        alphas = []
        for strength in strengths:
            x_adv = x_test + strength * perturbations

            accuracy_object.reset_states()
            accuracy_object.update_state(y_test, tf.math.argmax(model2(x_adv, training=False), 1))
            alpha = accuracy_object.result().numpy()

            alphas.append(alpha)

        return alphas
    return run

if 0:
    strengths = np.arange(0, 0.21, 0.01)
    fgsm = robust_fgsm(model)
    acc = fgsm(strengths)
    print(acc)
    import matplotlib.pyplot as plt
    plt.plot(strengths, acc)

    im = plt.imread("robustness.png")
    plt.imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")

    pgd = robust_pgd(model)
    t = time.time()
    acc2 = pgd(strengths)

    plt.plot(strengths, acc2)
if 0:
    for i0 in range(2):
        for i, s in enumerate([0, 0.01, 0.05, 0.1]):
            plt.subplot(3, 4, i+1+8*i0)
            adversarial_image = np.clip(x_test + s * perturbations * 255, 0, 255).astype(np.uint8)
            text = f"strength {s:.2f}\n"
            res = np.round(model(adversarial_image[:10])[i0].numpy(), 2)
            for j in range(10):
                text += f" {res[j]:.2f}"
                if j % 3 == 2:
                    text+= "\n"

            plt.title(text)
            plt.imshow(adversarial_image[i0])
    alphas = []
    for strength in strengths:
        adversarial_image = (x_test + strength * perturbations*255).astype(np.uint8)
        alpha = model.evaluate(adversarial_image, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
        alphas.append(alpha)
    #return alphas
##

@tf.function
def projected_gradient_descent(model, x, y, loss_fn, accuracy_object, num_steps, step_size, step_norm, eps, eps_norm,
                               clamp=(0, 1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x = tf.cast(x, tf.float32)
    eps = tf.cast(eps, tf.float32)
    x_adv = x
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(_x_adv)

            prediction = model(_x_adv)
            loss = loss_fn(prediction, y_target if targeted else y)
        gradient = tape.gradient(loss, _x_adv)

        if 1:
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = tf.sign(gradient) * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = gradient * step_size
                normed = tf.norm(tf.reshape(gradient, (_x_adv.shape[0], -1)), step_norm, axis=-1)
                normed2 = tf.reshape(normed, (-1, 1, 1, 1))
                gradients = gradients / normed2

            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = tf.where(x_adv > x + eps, x + eps, x_adv)
            x_adv = tf.where(x_adv < x - eps, x - eps, x_adv)
            pass
        else:
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = tf.norm(tf.reshape(delta, (delta.shape[0], -1)), ord=1) <= eps

            scaling_factor = tf.norm(tf.reshape(delta, (delta.shape[0], -1)), ord=1)
            scaling_factor = tf.where(mask, eps, scaling_factor)
            #scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / tf.reshape(scaling_factor, (-1, 1, 1, 1))

            x_adv = x + delta

        x_adv = tf.clip_by_value(x_adv, *clamp)

        if 1:
            accuracy_object.reset_states()
            accuracy_object.update_state(tf.math.argmax(y, 1), tf.math.argmax(model(x_adv, training=False), 1))
            tf.print(tf.reduce_max(tf.abs(x_adv - x)), tf.reduce_mean(tf.abs(x_adv - x)), accuracy_object.result())

    accuracy_object.reset_states()
    accuracy_object.update_state(tf.math.argmax(y, 1), tf.math.argmax(model(x_adv, training=False), 1))
    accuracy = accuracy_object.result()

    return x_adv, accuracy

def robust_pgd(model):
    from dimension_regularisation.dimension_reg_layer import DimensionReg
    from dimension_regularisation.dimensions_reg_layer_gamma import DimensionRegGammaWeights
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    x_test = x_test.astype(np.float32) / 255
    x_train = x_train.astype(np.float32) / 255
    y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

    model2 = tf.keras.models.Sequential([
        layer for layer in model.layers if not (layer.name.startswith("lambda") or
                                                layer.name.startswith("dimension_reg") or
                                                layer.name.startswith("dimension_reg_gamma_weights"))
        ]
    )

    model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.build(x_train.shape)
    model2.trainable = False
    model.build(x_train.shape)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    accuracy_object = tf.keras.metrics.Accuracy()

    def run(strengths):
        acc = []
        for s in strengths:
            x_adv, accuracy = projected_gradient_descent(model2, x_test, y_test_categorical,
                                                         loss_object, accuracy_object, 100, 0.01, 2, s, "inf")
            acc.append(accuracy)
        return [a.numpy() for a in acc]
    return run

if 0:
    model2 = tf.keras.models.Sequential([
        layer for layer in model.layers if not (layer.name.startswith("lambda") or layer.name.startswith("dimension_reg") or layer.name.startswith("dimension_reg_gamma_weights"))
        ]
    )
    model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.build(x_train.shape)
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    strengths = [0, 0.01, 0.03, 0.1, 0.3]
    import time

    fgsm = robust_fgsm(model)
    t = time.time()
    print(fgsm(strengths))
    print(time.time() - t)

    pgd = robust_pgd(model)
    t = time.time()
    print(pgd(strengths))
    print(time.time() - t)
if 0:
    x_adv = (x_adv*255).numpy().astype(np.uint8)
    print(np.max(x_adv - x_test))
    model2.evaluate(x_adv, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
    model2.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
    model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
    model2.evaluate(x_test/255, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
    model2.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]

    robust_fgsm(model, [0.01])

##
def robust_pgdxxx(model, strengths):
    from dimension_regularisation.dimension_reg_layer import DimensionReg
    from dimension_regularisation.dimensions_reg_layer_gamma import DimensionRegGammaWeights
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    model2 = tf.keras.models.Sequential([
        layer for layer in model.layers if not (layer.name.startswith("lambda") or
                                                layer.name.startswith("dimension_reg") or
                                                layer.name.startswith("dimension_reg_gamma_weights"))
        ]
    )
    input_label = tf.keras.utils.to_categorical(y_test, 10)
    model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.build(x_train.shape)
    model2.trainable = False
    model.build(x_train.shape)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    accuracy_object = tf.keras.metrics.Accuracy()

    @tf.function
    def iteration(x, y, delta, strength, eta):
        # get the input
        im = x + delta
        # record the gradient of the loss
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(im)
            prediction = model2(im)
            loss = loss_object(y, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, im)
        # update the delta
        delta = delta + eta * gradient
        # project it to ensure the limit of the maximum norm
        if 0:
            norm = tf.reduce_max(tf.abs(delta), axis=(1, 2, 3))[:, None, None, None]
            delta = tf.where(norm > strength, delta * strength / norm, delta)
        delta = tf.where(delta > strength, strength, delta)
        delta = tf.where(delta < -strength, -strength, delta)
        # print the norm
        #tf.print(tf.reduce_max(tf.abs(delta)))
        return delta

    @tf.function
    def get_accuracy(x, y):
        accuracy_object.reset_states()
        accuracy_object.update_state(tf.math.argmax(y, 1), tf.math.argmax(model(x, training=False), 1))
        return accuracy_object.result()

    @tf.function
    def get_loss(x, y):
        return loss_object(y, model(x, training=False))

   # @tf.function
    def fit(x, y, epochs, strength, eta):
        im0 = tf.cast(x / 255, tf.float32)
        delta = np.random.uniform(-strength, strength, size=im0.shape).astype(np.float32)

        for i in range(epochs):
            delta = iteration(im0, y, delta, strength, eta)

            acc = get_accuracy((im0 + delta) * 255, y)
            tf.print("max", tf.reduce_max(tf.abs(delta)), "mean", tf.reduce_mean(tf.abs(delta)), "acc", acc)
        acc = get_accuracy((im0 + delta) * 255, y)
        return delta, acc

        t = time.time()
        d, alpha = fit(x_test, input_label, 100, 0.01, 0.1)
        print("time", time.time() - t, "s")

        input_image = x_test[:100]; input_label = y_test[:100, 0]
        delta = np.zeros(input_image.shape, np.float32)
        im0 = tf.cast(input_image / 255, tf.float32)
        input_label = tf.keras.utils.to_categorical(input_label, 10)
        import time
        for i in range(10):
            t = time.time()
            delta = iteration(model2, im0, input_label, delta, strength, eta)
            print("time", time.time()-t, "s")

            adversarial_image = (input_image + d * 255)#.astype(np.uint8)
            alpha2 = model.evaluate(adversarial_image, input_label, batch_size=10000, verbose=False)[1]
            print(alpha, alpha0)
        return delta

    alphas = []
    for strength in strengths:
        perturbations = get_perturbation(x_test[:10], y_test[:10, 0], strength, 100)

        #pred = model.predict(x_test[:10])
        #pred2 = model.predict(adversarial_image[:10])

        adversarial_image = (x_test + perturbations*255).astype(np.uint8)
        alpha = model.evaluate(adversarial_image, tf.keras.utils.to_categorical(y_test, 10), batch_size=10000)[1]
        alphas.append(alpha)
    return alphas

if 0:
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(x_test[1])
    plt.subplot(122)
    plt.imshow(adversarial_image[1])

if __name__ == "__main__":
    import tensorflow as tf
    from scripts.net_helpers import read_data
    model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500/model_save")
    modelB = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b/model_save")
    modelB2 = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed/model_save")
    modelB2_ = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed_noreg/model_save")
    modelB210 = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10/model_save")
    modelB210tanh = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10_tanh/model_save")
    modelB2100 = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed100/model_save")
    dataB2 = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed/data.csv")
    dataB2_ = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed_noreg/data.csv")
    dataB210 = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10/data.csv")
    dataB210tanh = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10_tanh/data.csv")
    dataB2100 = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed100/data.csv")
    data_valilla = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/logs/data.csv")

    data = pd.read_csv("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_vanilla_tanh_linear/data.csv")

    model_valilla_tanh = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_vanilla_tanh_linear/model_save")

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=2000, activation='tanh'),
        #DimensionRegGammaWeights(p.reg1(1.), p.reg1value(1)),
        #DimensionRegGammaWeightsPreComputedBase(p.reg1(10.), p.reg1value(1)),
        tf.keras.layers.Dense(units=10),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.weights[0].assign(np.load("/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/_batch_modifier._architecture.sequential.0.weight.npy").T)
    model.weights[1].assign(np.load("/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/_batch_modifier._architecture.sequential.0.bias.npy").T)
    model.weights[2].assign(np.load("/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/_batch_modifier._architecture.sequential.2.weight.npy").T)
    model.weights[3].assign(np.load("/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/_batch_modifier._architecture.sequential.2.bias.npy").T)

    fig, axs = plt.subplots(2, 4, sharey="row", sharex=True)
    def plot(ax, dataB210, title=""):
        plt.sca(ax[0])
        plt.title(title)
        plt.plot(dataB210.epoch, dataB210.accuracy)
        plt.plot(dataB210.epoch, dataB210.val_accuracy)
        plt.ylabel("accuracy")
        plt.grid()
        plt.sca(ax[1])
        plt.plot(dataB210.epoch, dataB210.alpha_pre_computed_base)
        plt.plot(dataB210.epoch, dataB210.val_alpha_pre_computed_base)
        plt.axhline(1, color="k", ls="--")
        plt.grid()
        plt.xlabel("epochs")
        plt.ylabel("alpha")

    plot(axs[:, 0], dataB2_, "reg0")
    plot(axs[:, 1], dataB2, "reg1")
    plot(axs[:, 2], dataB210, "reg10")
    plot(axs[:, 3], dataB210tanh, "reg100")

    from dimension_regularisation.robustness import robust_fgsm, robust_pgd
    strengths = np.arange(0, 0.21, 0.01)
    fgsm = robust_fgsm(model)
    fgsmB = robust_fgsm(modelB)
    fgsmB2 = robust_fgsm(modelB2)
    fgsmB2_ = robust_fgsm(modelB2_)
    fgsmB210 = robust_fgsm(modelB210)
    fgsmB210tanh = robust_fgsm(modelB210tanh)
    fgsmB2100 = robust_fgsm(modelB2100)
    acc = fgsm(strengths)
    accB = fgsmB(strengths)
    accB2 = fgsmB2(strengths)
    accB2_ = fgsmB2_(strengths)
    accB210 = fgsmB210(strengths)
    accB210tanh = fgsmB210tanh(strengths)
    accB2100 = fgsmB2100(strengths)

    acc_vanilla_tanh = robust_fgsm(model_valilla_tanh)(strengths)
    print(acc)
    import matplotlib.pyplot as plt
    plt.subplot(121)
    #plt.plot(strengths, acc, "-o")
    #plt.plot(strengths, accB, "-o")
    plt.plot(strengths, acc, "-o", label="imported")
    plt.plot(strengths, accB2_, "-o", label="0")
    plt.plot(strengths, accB2, "-o", label="1")

    plt.plot(strengths, accB210, "-o", label="10")
    plt.plot(strengths, accB210tanh, "-o", label="10tanh")
    plt.plot(strengths, accB2100, "-o", label="100")

    plt.plot(strengths, acc_vanilla_tanh, "-o", label="100")

    im = plt.imread("/home/richard/PycharmProjects/dimension_regularisation/dimension_regularisation/robustness.png")
    plt.imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
    plt.legend()
    plt.subplot(122)

    #pgdB2_ = robust_pgd(model)
    pgd = robust_pgd(model)
    pgdB = robust_pgd(modelB)
    pgdB2 = robust_pgd(modelB2)
    pgdB2_ = robust_pgd(modelB2_)
    pgdB210 = robust_pgd(modelB210)
    pgdB2100 = robust_pgd(modelB2100)
    pgd_acc = pgd(strengths)
    pgd_accB = pgdB(strengths)
    pgd_accB2 = pgdB2(strengths)
    pgd_accB2_ = pgdB2_(strengths)
    pgd_accB210 = pgdB210(strengths)
    pgd_accB2100 = pgdB2100(strengths)

    plt.plot(strengths, pgd_accB2_, "-o", label="0")
    plt.plot(strengths, pgd_accB2, "-o", label="1")

    plt.plot(strengths, pgd_accB210, "-o", label="10")
    plt.plot(strengths, pgd_accB2100, "-o", label="100")
    im = plt.imread("/home/richard/PycharmProjects/dimension_regularisation/dimension_regularisation/robustness2.png")
    plt.imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
    plt.legend()

    if 0:
        data1 = read_data(
            r"../results/cedar_logs_expcifar3/iter-1_reg1-0_reg1value-1.0/",
            file_name="data.csv")
        print(data1.iloc[0])
        model = tf.keras.models.load_model(data1.iloc[0].filename.replace("data.csv", "model_save"))
    if 0:
        data1 = read_data(
            r"../results/cedar_logs_expcifar4/iter-1_gamma-False_reg1-{reg1}_reg1value-1.0/",
            file_name="data.csv")
        for filename, d in data1.groupby("filename"):
            print(filename, d.reg1[0])
            model = tf.keras.models.load_model(filename.replace("data.csv", "model_save"))
            strengths = np.arange(0, 0.21, 0.01)
            import matplotlib.pyplot as plt
            plt.plot(strengths, robust_fgsm(model, strengths), label=d.reg1[0])

        plt.legend(title="regularisation strength")
        plt.xlabel("adversarial strength")
        plt.ylabel("accuracy")
