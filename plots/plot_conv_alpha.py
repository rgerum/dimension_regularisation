import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from net_helpers import read_data
import pylustrator
pylustrator.start()

data = read_data(r"../cedar_logs/conv/", file_name="alpha.csv")

def fmt(x):
    return "0" if x==0 else f"$10^{{{int(np.log10(x))}}}$"

print(data.columns)
def plot(value, data):
    data = data[data.reg1 == value]
    data = data[data.reg2 == 0]

    data = data[data.name == "alpha_1"]
    agg = data.groupby("epoch")["value"].agg(["mean", "sem"])
    plt.plot(agg.index, agg["mean"])
    plt.fill_between(agg.index, agg["mean"]-agg["sem"], agg["mean"]+agg["sem"], alpha=0.5)

plt.subplot(121)
plot(0, data)
plot(0.001, data)
plot(0.01, data)
plot(0.1, data)
plot(1, data)

def plot(value, data):
    data = data[data.reg1 == value]
    data = data[data.reg2 == 0]

    data = data[data.name == "alpha_2"]
    agg = data.groupby("epoch")["value"].agg(["mean", "sem"])
    plt.plot(agg.index, agg["mean"])
    plt.fill_between(agg.index, agg["mean"]-agg["sem"], agg["mean"]+agg["sem"], alpha=0.5)

plt.subplot(122)
plot(0, data)
plot(0.001, data)
plot(0.01, data)
plot(0.1, data)
plot(1, data)
plt.show()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(20.820000/2.54, 11.230000/2.54, forward=True)
plt.figure(1).axes[0].set_ylim(0.3911857662495933, 0.7340535182386119)
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].set_position([0.383242, 0.637759, 0.232879, 0.310832])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend().set_visible(False)
plt.figure(1).axes[0].texts[0].set_position([-161.475846, -0.084627])
plt.figure(1).axes[1].set_ylim(0.3911857662495933, 0.7340535182386119)
plt.figure(1).axes[1].grid(True)
plt.figure(1).axes[1].legend(handlelength=1.2999999999999998, handletextpad=0.19999999999999998, columnspacing=0.8, ncol=2, fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[1].set_position([0.707320, 0.637759, 0.232879, 0.310832])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_legend()._set_loc((1.069869, -0.357022))
plt.figure(1).axes[1].get_legend()._set_loc((0.516779, 0.092971))
plt.figure(1).axes[2].set_ylim(0.3911857662495933, 0.7340535182386119)
plt.figure(1).axes[2].grid(True)
plt.figure(1).axes[2].set_position([0.383242, 0.200461, 0.232879, 0.325043])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("val_accuracy")
plt.figure(1).axes[3].grid(True)
plt.figure(1).axes[3].set_position([0.707320, 0.200461, 0.232879, 0.325043])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("val acc/acc")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.show()

