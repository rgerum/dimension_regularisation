import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from net_helpers import read_data
import pylustrator
pylustrator.start()

data = read_data(r"../cedar_logs/* 7087590 */data.csv", file_name="data.csv")

def fmt(x):
    return "0" if x==0 else f"$10^{{{int(np.log10(x))}}}$"

data = data[data.reg2 == 0]
print(data.columns)
plt.subplot(141)
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["accuracy"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=name)
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)
plt.legend()
plt.ylabel("accuracy")
plt.xlabel("epoch")

plt.subplot(142)
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["val_accuracy"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=fmt(name))
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)
plt.legend()
plt.ylabel("val_accuracy")
plt.xlabel("epoch")

plt.subplot(143)
d = data.groupby(["datetime", "reg1"]).max().groupby(["reg1"])["val_accuracy"].agg(["mean", "sem"])
for x, y, yerr in zip([fmt(x) for x in d.index], d["mean"], d['sem']):
    print(x, y, yerr)
    plt.errorbar([x], [y], [yerr], capsize=5, zorder=10)
plt.errorbar([fmt(x) for x in d.index], d["mean"], d['sem'], color="k", capsize=5, barsabove=True, zorder=5)
plt.grid()
plt.gca().grid()
plt.subplot(144)
d1 = data.groupby(["datetime", "reg1"]).max().groupby(["reg1"])["accuracy"].agg(["mean", "sem"])
d2 = data.groupby(["datetime", "reg1"]).max().groupby(["reg1"])["val_accuracy"].agg(["mean", "sem"])
plt.plot([fmt(x) for x in d1.index], d2["mean"]/d1["mean"], color="k")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_ylim(0.3911857662495933, 0.5340535182386119)
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].set_position([0.111088, 0.634017, 0.313390, 0.310393])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend().set_visible(False)
plt.figure(1).axes[1].set_ylim(0.3911857662495933, 0.5340535182386119)
plt.figure(1).axes[1].grid(True)
plt.figure(1).axes[1].set_position([0.512867, 0.634017, 0.313390, 0.310393])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_legend()._set_loc((1.069869, -0.357022))
plt.figure(1).axes[2].set_ylim(0.3911857662495933, 0.5340535182386119)
plt.figure(1).axes[2].grid(True)
plt.figure(1).axes[2].set_position([0.111088, 0.197336, 0.313390, 0.324585])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("val_accuracy")
plt.figure(1).axes[3].grid(True)
plt.figure(1).axes[3].set_position([0.512867, 0.197336, 0.313390, 0.324585])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("val acc/acc")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.show()

