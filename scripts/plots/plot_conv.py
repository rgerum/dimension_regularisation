import matplotlib.pyplot as plt
import numpy as np

from scripts.net_helpers import read_data
import pylustrator
pylustrator.start()

data = read_data(r"../cedar_logs/conv/*/data.csv", file_name="data.csv")

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

plt.text(0, 0, """
model = Sequential([
    InputLayer((32, 32, 3)),
    Lambda(lambda x: x/255),

    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),
    DimensionReg(0, 1, "alpha_1"),
    Dropout(0.5),

    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),
    DimensionReg(0, 1, "alpha_2"),
    Dropout(0.5),

    Flatten(),
    Dense(10, activation='softmax'),
])
""")

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


#data = read_data(r"../cedar_logs/conv/", file_name="alpha.csv")

def fmt(x):
    return "0" if x==0 else f"$10^{{{int(np.log10(x))}}}$"

print(data.columns)

plt.axes(label="alpha1")
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["alpha_1"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=fmt(name))
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)


plt.axes(label="alpha2")
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["alpha_2"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=fmt(name))
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)


plt.axes(label="val_alpha1")
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["val_alpha_1"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=fmt(name))
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)


plt.axes(label="val_alpha2")
for name, d in data.groupby("reg1"):
    d = d.groupby("epoch")["val_alpha_2"].agg(["mean", "sem"])
    p, = plt.plot(d.index, d["mean"], label=fmt(name))
    plt.fill_between(d.index, d["mean"]-d["sem"], d["mean"]+d["sem"], color=p.get_color(), alpha=0.5)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).set_size_inches(20.780000/2.54, 18.490000/2.54, forward=True)
plt.figure(1).ax_dict["alpha1"].set_xlim(-4.5, 94.5)
plt.figure(1).ax_dict["alpha1"].set_ylim(0.5659017998038169, 1.3634335777894526)
plt.figure(1).ax_dict["alpha1"].set_xticks([0.0, 25.0, 50.0, 75.0])
plt.figure(1).ax_dict["alpha1"].set_xticklabels(["", "", "", ""], fontsize=10.0, fontweight="normal", color="#000000ff", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).ax_dict["alpha1"].grid(True)
plt.figure(1).ax_dict["alpha1"].set_position([0.431059, 0.340540, 0.230634, 0.191706])
plt.figure(1).ax_dict["alpha1"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alpha1"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alpha1"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["alpha1"].get_yaxis().get_label().set_text("alpha 1")
plt.figure(1).ax_dict["alpha2"].set_ylim(0.5659017998038169, 1.3634335777894526)
plt.figure(1).ax_dict["alpha2"].set_xticklabels([])
plt.figure(1).ax_dict["alpha2"].set_yticklabels([])
plt.figure(1).ax_dict["alpha2"].grid(True)
plt.figure(1).ax_dict["alpha2"].set_position([0.706517, 0.340540, 0.230634, 0.191706])
plt.figure(1).ax_dict["alpha2"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alpha2"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alpha2"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["alpha2"].get_yaxis().get_label().set_text("alpha 2")
plt.figure(1).ax_dict["val_alpha1"].set_xlim(-4.5, 94.5)
plt.figure(1).ax_dict["val_alpha1"].set_xticks([0.0, 25.0, 50.0, 75.0])
plt.figure(1).ax_dict["val_alpha1"].set_xticklabels(["0", "25", "50", "75"], fontsize=10, fontweight="normal", color="#000000ff", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).ax_dict["val_alpha1"].grid(True)
plt.figure(1).ax_dict["val_alpha1"].set_position([0.431059, 0.115409, 0.230634, 0.191706])
plt.figure(1).ax_dict["val_alpha1"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["val_alpha1"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["val_alpha1"].get_xaxis().get_label().set_text("epoch")
plt.figure(1).ax_dict["val_alpha1"].get_yaxis().get_label().set_text("val alpha 1")
plt.figure(1).ax_dict["val_alpha2"].set_yticklabels([])
plt.figure(1).ax_dict["val_alpha2"].grid(True)
plt.figure(1).ax_dict["val_alpha2"].set_position([0.706517, 0.115409, 0.230634, 0.191706])
plt.figure(1).ax_dict["val_alpha2"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["val_alpha2"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["val_alpha2"].get_xaxis().get_label().set_text("epoch")
plt.figure(1).ax_dict["val_alpha2"].get_yaxis().get_label().set_text(" val alpha 2")
plt.figure(1).axes[0].set_ylim(0.0, 1.0)
plt.figure(1).axes[0].set_xticklabels([])
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].set_position([0.118927, 0.340540, 0.230634, 0.191706])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend().set_visible(False)
plt.figure(1).axes[0].texts[0].set_position([-47.814757, 1.170957])
plt.figure(1).axes[0].get_xaxis().get_label().set_text('')
plt.figure(1).axes[1].set_ylim(0.0, 1.0)
plt.figure(1).axes[1].grid(True)
plt.figure(1).axes[1].legend(handlelength=1.2999999999999998, handletextpad=0.19999999999999998, columnspacing=0.8, ncol=5, fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[1].set_position([0.118927, 0.115409, 0.230634, 0.191706])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_legend()._set_loc((1.370093, 2.214668))
plt.figure(1).axes[1].get_legend()._set_loc((1.306486, 2.279244))
plt.figure(1).axes[1].get_yaxis().get_label().set_text("val accuracy")
plt.figure(1).axes[2].set_ylim(0.3911857662495933, 1.0)
plt.figure(1).axes[2].grid(True)
plt.figure(1).axes[2].set_position([0.402630, 0.738592, 0.225459, 0.193507])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("val_accuracy")
plt.figure(1).axes[3].set_ylim(1.0, 1.06)
plt.figure(1).axes[3].grid(True)
plt.figure(1).axes[3].set_position([0.706517, 0.738592, 0.225459, 0.193507])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].get_xaxis().get_label().set_text("regularisation strength")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("val acc/acc")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.show()

