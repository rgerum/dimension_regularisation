from dimension_regularisation.attack_tf import fgsm, pgd, get_tf_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import sys
sys.path.append("/home/richard/PycharmProjects/power_law_original_code/neurips_experiments")
from attack_torch import fgsm_original, load_network_torch, pgd_original, pgd as pgd_torch, get_train_data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_test = tf.cast(x_test/255., tf.float32)
#y_test = tf.keras.utils.to_categorical(y_test)

strengths = np.arange(0, 0.2, 0.01)

x, y = get_train_data()
x = x[:1000]
y = y[:1000]

if 0:
    strengths = np.arange(0, 0.2, 0.05)
    model_torch = load_network_torch('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=5.0')
    #acc = fgsm(model, x_test, y_test, strengths)
    #acc2 = pgd(model, x_test, y_test, strengths)
    x, y = get_train_data()
    x = x[:1000]
    y = y[:1000]
    print("fgsm_original")
    acc = fgsm_original(model_torch, x, y, strengths)
    plt.plot(strengths, acc)
    print('pgd_original')
    acc2 = pgd_original(model_torch, x, y, strengths)
    plt.plot(strengths, acc2)
    print("pgd")
    acc3 = pgd(model_torch, x, y, strengths)
    plt.plot(strengths, acc3)
    plt.show()
    exit()

fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_vanilla_tanh_linear/model_save")
model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_vanilla_tanh/model_save")

acc = fgsm(model, x_test, y_test, strengths)
acc2 = pgd(model, x_test, y_test, strengths)

axes[0, 0].plot(strengths, acc, color="k")
axes[0, 1].plot(strengths, acc2, color="k")

model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10/model_save")

acc = fgsm(model, x_test, y_test, strengths)
acc2 = pgd(model, x_test, y_test, strengths)

axes[0, 0].plot(strengths, acc)
axes[0, 1].plot(strengths, acc2)


im = plt.imread("/home/richard/PycharmProjects/dimension_regularisation/dimension_regularisation/robustness.png")
axes[0, 0].imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
axes[1, 0].imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
axes[2, 0].imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
im = plt.imread("/home/richard/PycharmProjects/dimension_regularisation/dimension_regularisation/robustness2.png")
axes[0, 1].imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
axes[1, 1].imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")
axes[2, 1].imshow(im, extent=[0, 0.2, 0, 1], aspect="auto")

axes[0, 0].set_ylabel("trained tf\nevaluated tf")
axes[1, 0].set_ylabel("trained torch\nevaluated tf")
axes[2, 0].set_ylabel("trained torch\nevaluated torch")

for ax in axes.flatten():
    ax.grid()

kw, d = torch.load('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/vanilla_activation=tanh_epochs=50')[0]
model = get_tf_model(d)

acc = fgsm(model, x_test, y_test, strengths)
acc2 = pgd(model, x_test, y_test, strengths)
#acc2 = fgsm_original(model_torch, x_test, y_test, strengths)

axes[1, 0].plot(strengths, acc, color="k")
axes[1, 1].plot(strengths, acc2, color="k")

model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10/model_save")
kw, d = torch.load('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=1.0')[0]
model = get_tf_model(d)

acc = fgsm(model, x_test, y_test, strengths)
acc2 = pgd(model, x_test, y_test, strengths)

axes[1, 0].plot(strengths, acc)
axes[1, 1].plot(strengths, acc2)

model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10/model_save")
kw, d = torch.load('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=2.0')[0]
model = get_tf_model(d)

acc = fgsm(model, x_test, y_test, strengths)
acc2 = pgd(model, x_test, y_test, strengths)

axes[1, 0].plot(strengths, acc)
axes[1, 1].plot(strengths, acc2)

model = tf.keras.models.load_model("/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/logs/tmp500b_precomputed10/model_save")
kw, d = torch.load('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=5.0')[0]
model = get_tf_model(d)

acc = fgsm(model, x_test, y_test, strengths)
acc2 = pgd(model, x_test, y_test, strengths)

axes[1, 0].plot(strengths, acc)
axes[1, 1].plot(strengths, acc2)

###
if 1:

    model_torch = load_network_torch('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/vanilla_activation=tanh_epochs=50')
    #acc = fgsm(model, x_test, y_test, strengths)
    #acc2 = pgd(model, x_test, y_test, strengths)
    acc = fgsm_original(model_torch, x, y, strengths)
    acc2 = pgd_original(model_torch, x, y, strengths)

    axes[2, 0].plot(strengths, acc, color="k")
    axes[2, 1].plot(strengths, acc2, color="k")

    model_torch = load_network_torch(
        '/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=1.0')
    # acc = fgsm(model, x_test, y_test, strengths)
    # acc2 = pgd(model, x_test, y_test, strengths)
    acc = fgsm_original(model_torch, x, y, strengths)
    acc2 = pgd_original(model_torch, x, y, strengths)

    axes[2, 0].plot(strengths, acc)
    axes[2, 1].plot(strengths, acc2)


    model_torch = load_network_torch('/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=2.0')
    #acc = fgsm(model, x_test, y_test, strengths)
    #acc2 = pgd(model, x_test, y_test, strengths)
    acc = fgsm_original(model_torch, x, y, strengths)
    acc2 = pgd_original(model_torch, x, y, strengths)

    axes[2, 0].plot(strengths, acc)
    axes[2, 1].plot(strengths, acc2)

    model_torch = load_network_torch(
        '/home/richard/PycharmProjects/power_law_original_code/neurips_experiments/experiment_1/tau=10_activation=tanh_epochs=50_alpha=1.0_beta=5.0')
    # acc = fgsm(model, x_test, y_test, strengths)
    # acc2 = pgd(model, x_test, y_test, strengths)
    acc = fgsm_original(model_torch, x, y, strengths)
    acc2 = pgd_original(model_torch, x, y, strengths)

    axes[2, 0].plot(strengths, acc)
    axes[2, 1].plot(strengths, acc2)

plt.show()
