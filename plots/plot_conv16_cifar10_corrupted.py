import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from net_helpers import read_data
import pylustrator
#pylustrator.start()

output_add = "_cifar10_gamma"
data1 = read_data(r"../cedar_logs_expcifar2/iter-{iter}_reg1-{reg1}_reg1value-{reg1value}/", file_name="data.csv")
print(data1.filename.unique())
if 0:
    for name, d0 in data1.groupby("reg1"):
        for i in range(1, 6):
            plt.subplot(1, 6, i)
            d = d0.groupby("epoch")[f"accuracy_brightness_{i}"].agg(["mean", "sem"])
            p, = plt.plot(d.index, d["mean"], label=name)
            plt.fill_between(d.index, d["mean"] - d["sem"], d["mean"] + d["sem"], color=p.get_color(), alpha=0.5)
            #plt.plot(d.epoch, d.accuracy_brightness_1)
elif 0:
    index = 0
    for name, d0 in data1.groupby("reg1"):
        d2 = []
        d2_std = []
        for i in range(0, 6):
            #plt.subplot(1, 6, i)
            if i == 0:
                d = d0.groupby("epoch")[f"val_accuracy"].agg(["mean", "sem"])
            else:
                d = d0.groupby("epoch")[f"accuracy_brightness_{i}"].agg(["mean", "sem"])
            d2.append(np.max(d["mean"]))
            d2_std.append(d["sem"][np.argmax(d["mean"])])
        plt.errorbar(np.arange(0, 6)+index*0.02, d2, d2_std, label=name, capsize=5)
        index += 1
else:
    index = 0
    ax = None
    data11 = data1
    corrupt = ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"][1]
    for index0, (name, data1) in enumerate(data11.groupby("reg1value")):
        for i in range(0, 6):
            if ax is None:
                ax = plt.subplot(5, 6, i + 1 + index0 * 6)
            else:
                ax = plt.subplot(5, 6, i + 1 + index0*6, sharex=ax, sharey=ax)
            for name, d0 in data1.groupby("reg1"):
                if i == 0:
                    plt.title("val_acc")
                    d = d0.groupby("epoch")[f"val_accuracy"].agg(["mean", "sem"])
                else:
                    plt.title(f"bright_{i}")
                    d = d0.groupby("epoch")[f"accuracy_{corrupt}_{i}"].agg(["mean", "sem"])
                maxi = np.argmax(d["mean"])
                print(name, d["mean"][maxi], d["sem"][maxi])
                plt.bar(name, d["mean"][maxi], yerr=d["sem"][maxi], label=name, capsize=5)
            index += 1
            plt.xlabel("reg")
            plt.ylabel("accuracy")
plt.legend()
#plt.ylim(bottom=0.5)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
#% end: automatic generated code from pylustrator
plt.show()

