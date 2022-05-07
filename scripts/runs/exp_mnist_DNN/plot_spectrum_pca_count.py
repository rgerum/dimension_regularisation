import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from dimension_regularisation.pca_variance import linear_fit

fig, axes = plt.subplots(4, 5)
axes_all = axes.flatten()
for j, classes in enumerate([None]):#enumerate(np.arange(2, 11, 1)):
    axes = axes_all#[j*10:(j+1)*10]
    plt.sca(axes[-1])
    plt.axhline(1, lw=0.8, color="k")
    for index, pca in enumerate(np.arange(2, 11, 1)):
    #for index, pca in enumerate([5]):
        for iter in [0, 1]:
            alpha = []
            epochs = []
            for i in range(1000):
                plt.sca(axes[index])
                #plt.title(f"cifar10 pca {pca}")
                try:
                    y = np.load(f"logs_class_count/long_run_iter-{iter}_class_count-{pca}_mnist_pca-{pca}/epoch_{i}.npy")
                except FileNotFoundError:
                    continue
                x = np.log(np.arange(1, y.shape[0] + 1, 1.0, y.dtype))
                min_x = 0; max_x = 10

                max_x_value = 3
                if pca is None:
                    max_x_value = 5
                else:
                    max_x_value = np.log(pca + 0.5)
                max_x = np.argmax( (y < -15) | (x > max_x_value))
                print(max_x, np.argmax(y < -15), np.argmax(x > max_x_value))

                a, b = linear_fit(x[min_x:max_x], y[min_x:max_x])

                mse = np.mean((b * x[min_x:max_x] + a - y[min_x:max_x]) ** 2)
                # return the negative of the slope
                if i % 250 == 0 and iter == 0:
                    l, = plt.plot(x, y, "o", alpha=0.5)
                    plt.plot(x, x*b+a, color=l.get_color())
                print(-b, mse)
                epochs.append(i)
                alpha.append(-b)
            if len(epochs) == 0:
                continue
            plt.sca(axes[-2])
            plt.plot(epochs, alpha, "-")#, color=l.get_color())
            plt.text(epochs[-1], alpha[-1], pca)#, color=l.get_color())
            plt.ylim(0, 3)
            plt.sca(axes[-1])
            plt.grid(True)
            plt.plot(pca if pca is not None else 30, alpha[-1], "o")#, color=l.get_color())
            plt.ylim(0, 3)
            plt.grid(True)
    #plt.subplot(121)
#plt.axvline(x[min_x])
#plt.axvline(x[max_x])
plt.show()