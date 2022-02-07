import matplotlib.pyplot as plt
import numpy as np

from scripts.net_helpers import read_data
import pylustrator
pylustrator.start()

data = read_data(r"../cedar_logs/conv_noweightshare/*/data.csv", file_name="data.csv")

data = data[data.reg2 == 0]
data = data[(data.reg1value == 0.6) | (data.reg1 == 0)]
data = data[data.weight_share == True]

# only finished experiments
data = data[data.filename.isin(data[data.epoch == data.epoch.max()].filename)]

def fmt(x):
    return "0" if x==0 else f"$10^{{{int(np.log10(x))}}}$"

print(data.columns)
print(len(data))
plt.subplot(141)
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["accuracy"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=name)
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)
plt.legend()
plt.ylabel("accuracy")
plt.xlabel("epoch")

plt.figtext(0, 0, """
model = Sequential([
    InputLayer((32, 32, 3)),
    Lambda(lambda x: x/255),

    Conv2D(32, 3, activation='relu',
       kernel_initializer='he_uniform',
       weight_share=True),
    MaxPool2D(2),
    DimensionReg(x, 0.6), # alpha 1
    Dropout(0.5),

    Conv2D(32, 3, activation='relu',
       kernel_initializer='he_uniform',
       weight_share=True),
    MaxPool2D(2),
    DimensionReg(0, 1), # alpha 2
    Dropout(0.5),

    Flatten(),
    Dense(256, activation='relu'),
    DimensionReg(0, 1), # alpha 3
    Dense(10, activation='softmax'),
])
""")

plt.figtext(0, 0, "Dense Network")

plt.subplot(142)
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["val_accuracy"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=fmt(name))
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)
plt.legend()
plt.ylabel("val_accuracy")
plt.xlabel("epoch")

plt.subplot(143)
d = data.groupby(["datetime", "reg1"]).max().groupby(["reg1"])["val_accuracy"].agg(["mean", "sem", "count"])
for x, y, yerr, c in zip([fmt(x) for x in d.index], d["mean"], d['sem'], d['count']):
    print(x, y, yerr)
    plt.errorbar([x], [y], [yerr], capsize=5, zorder=10)
    plt.text(x, y+yerr, f"n={c}", ha="center", va="bottom")
plt.errorbar([fmt(x) for x in d.index], d["mean"], d['sem'], color="k", capsize=5, barsabove=True, zorder=5)
plt.grid()
plt.gca().grid()
plt.subplot(144)
d1 = data[data.epoch == data.epoch.max()].groupby(["reg1"])["accuracy"].agg(["mean", "sem"])
d2 = data[data.epoch == data.epoch.max()].groupby(["reg1"])["val_accuracy"].agg(["mean", "sem"])
plt.plot([fmt(x) for x in d1.index], d2["mean"]/d1["mean"], color="k")


#data = read_data(r"../cedar_logs/conv/", file_name="alpha.csv")

def fmt(x):
    return "0" if x==0 else f"$10^{{{int(np.log10(x))}}}$"

def plot_lines(group_name, value):
    for name, d in data.groupby(group_name):
        d = d.groupby("epoch")[value].agg(["mean", "sem"])
        p, = plt.plot(d.index, d["mean"], label=fmt(name))
        plt.fill_between(d.index, d["mean"] - d["sem"], d["mean"] + d["sem"], color=p.get_color(), alpha=0.5)

print(data.columns)

plt.axes(label="alpha1")
plot_lines("reg1", "alpha")
plt.axhline(0.6, color="r", lw=0.8)

plt.axes(label="alpha2")
plot_lines("reg1", "alpha_1")

plt.axes(label="alpha3")
plot_lines("reg1", "alpha_2")

plt.axes(label="val_alpha1")
plot_lines("reg1", "val_alpha")
plt.axhline(0.6, color="r", lw=0.8)

plt.axes(label="val_alpha2")
plot_lines("reg1", "val_alpha_1")

plt.axes(label="val_alpha3")
plot_lines("reg1", "val_alpha_2")


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).set_size_inches(24.070000/2.54, 20.470000/2.54, forward=True)
plt.figure(1).ax_dict["alpha1"].set_xlim(-4.5, 94.5)
plt.figure(1).ax_dict["alpha1"].set_ylim(0.5, 1.5)
plt.figure(1).ax_dict["alpha1"].set_xticks([0.0, 25.0, 50.0, 75.0])
plt.figure(1).ax_dict["alpha1"].set_xticklabels(["", "", "", ""], fontsize=10.0, fontweight="normal", color="#000000ff", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).ax_dict["alpha1"].grid(True)
plt.figure(1).ax_dict["alpha1"].set_position([0.326470, 0.308059, 0.194763, 0.169668])
plt.figure(1).ax_dict["alpha1"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alpha1"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alpha1"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["alpha1"].get_yaxis().get_label().set_text("alpha 1")
plt.figure(1).ax_dict["alpha2"].set_ylim(0.5, 1.5)
plt.figure(1).ax_dict["alpha2"].set_xticklabels([])
plt.figure(1).ax_dict["alpha2"].set_yticklabels([])
plt.figure(1).ax_dict["alpha2"].grid(True)
plt.figure(1).ax_dict["alpha2"].set_position([0.559856, 0.308059, 0.194763, 0.169668])
plt.figure(1).ax_dict["alpha2"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alpha2"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alpha2"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["alpha2"].get_yaxis().get_label().set_text("alpha 2")
plt.figure(1).ax_dict["alpha3"].set_ylim(0.5, 1.5)
plt.figure(1).ax_dict["alpha3"].set_yticks([1.0, 1.5, 2.0])
plt.figure(1).ax_dict["alpha3"].set_xticklabels([])
plt.figure(1).ax_dict["alpha3"].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="right")
plt.figure(1).ax_dict["alpha3"].grid(True)
plt.figure(1).ax_dict["alpha3"].set_position([0.793242, 0.308059, 0.194763, 0.169668])
plt.figure(1).ax_dict["alpha3"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alpha3"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alpha3"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["alpha3"].get_yaxis().get_label().set_text("alpha 3")
plt.figure(1).ax_dict["val_alpha1"].set_xlim(-4.5, 94.5)
plt.figure(1).ax_dict["val_alpha1"].set_ylim(0.5, 1.5)
plt.figure(1).ax_dict["val_alpha1"].set_xticks([0.0, 25.0, 50.0, 75.0])
plt.figure(1).ax_dict["val_alpha1"].set_xticklabels(["0", "25", "50", "75"], fontsize=10, fontweight="normal", color="#000000ff", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).ax_dict["val_alpha1"].grid(True)
plt.figure(1).ax_dict["val_alpha1"].set_position([0.326470, 0.108809, 0.194763, 0.169668])
plt.figure(1).ax_dict["val_alpha1"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["val_alpha1"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["val_alpha1"].get_xaxis().get_label().set_text("epoch")
plt.figure(1).ax_dict["val_alpha1"].get_yaxis().get_label().set_text("val alpha 1")
plt.figure(1).ax_dict["val_alpha2"].set_ylim(0.5, 1.5)
plt.figure(1).ax_dict["val_alpha2"].set_yticklabels([])
plt.figure(1).ax_dict["val_alpha2"].grid(True)
plt.figure(1).ax_dict["val_alpha2"].set_position([0.559856, 0.108809, 0.194763, 0.169668])
plt.figure(1).ax_dict["val_alpha2"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["val_alpha2"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["val_alpha2"].get_xaxis().get_label().set_text("epoch")
plt.figure(1).ax_dict["val_alpha2"].get_yaxis().get_label().set_text(" val alpha 2")
plt.figure(1).ax_dict["val_alpha3"].set_ylim(0.5, 1.5)
plt.figure(1).ax_dict["val_alpha3"].set_yticks([1.0, 1.5, 2.0])
plt.figure(1).ax_dict["val_alpha3"].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="right")
plt.figure(1).ax_dict["val_alpha3"].grid(True)
plt.figure(1).ax_dict["val_alpha3"].set_position([0.793242, 0.108809, 0.194763, 0.169668])
plt.figure(1).ax_dict["val_alpha3"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["val_alpha3"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["val_alpha3"].get_xaxis().get_label().set_text("epoch")
plt.figure(1).ax_dict["val_alpha3"].get_yaxis().get_label().set_text("val alpha 3")
plt.figure(1).axes[0].set_ylim(0.0, 1.0)
plt.figure(1).axes[0].set_xticklabels([])
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].set_position([0.701667, 0.799312, 0.194763, 0.169668])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend().set_visible(False)
plt.figure(1).axes[0].get_xaxis().get_label().set_text('')
plt.figure(1).axes[1].set_ylim(0.0, 1.0)
plt.figure(1).axes[1].grid(True)
plt.figure(1).axes[1].legend(handlelength=1.2999999999999998, handletextpad=0.19999999999999998, columnspacing=0.8, ncol=5, fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[1].set_position([0.701667, 0.589374, 0.194763, 0.169668])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_legend()._set_loc((-0.206196, -0.583490))
plt.figure(1).axes[1].get_yaxis().get_label().set_text("val accuracy")
plt.figure(1).axes[2].set_ylim(0.7, 0.78)
plt.figure(1).axes[2].grid(True)
plt.figure(1).axes[2].set_position([0.401031, 0.797718, 0.190393, 0.171262])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("val_accuracy")
plt.figure(1).axes[3].set_ylim(-0.026359598250212057, 1.0)
plt.figure(1).axes[3].grid(True)
plt.figure(1).axes[3].set_position([0.401031, 0.587780, 0.190393, 0.171262])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("val acc/acc")
plt.figure(1).texts[0].set_position([0.022175, 0.498137])
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.show()

