import matplotlib.pyplot as plt
import pandas as pd
from scripts.net_helpers import read_data
import numpy as np
import pylustrator
from dimension_regularisation.pandas_grid_plot import grid_iterator, plot_color_grad
from matplotlib import rc
import matplotlib

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Ubuntu Condensed']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Tahoma']#, 'Roboto Conddensed', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
pylustrator.start()
from scripts.net_helpers import format_glob
import numpy as np

if 1:
    data1 = read_data(r"../../results/expcifar5_DNN_extened_range/iter-{iter:d}_gamma-{gamma}_reg1-{reg1:f}_reg1value-{reg1value:f}/", file_name="data.csv")
    output_path = "_alpha_"
    data1 = data1.query("gamma == 'False'")
#print([(c, data1[c].dtype) for c in data1.columns])
#exit()
if 0:
    #data1 = read_data(r"/home/richard/PycharmProjects/dimension_regularisation/scripts/runs/exp_mnist_DNN/logs/tmp600", file_name="data.csv")
    data1 = read_data(r"../../results/expcifar5_DNN/iter-0_gamma-True_reg1-1_reg1value-1.0/", file_name="data.csv")
    data1["gamma"] = 'True'
    data1["iter"] = '1'
    data1["reg1"] = 1
    data1["reg1value"] = 1
    if 0:
        data1["alpha_pre_computed_base"] = data1["alpha"]
        data1["val_alpha_pre_computed_base"] = data1["val_alpha"]
print(data1.filename.unique())
print(data1.iter.unique())

data1["reg_strength"] = data1["reg1"]
data1["reg_target"] = data1["reg1value"]
#if "alpha_pre_computed_base" in data1.columns:
#    data1["alpha"] = data1["alpha_pre_computed_base"]
#if "val_alpha_pre_computed_base" in data1.columns:
#    data1["val_alpha"] = data1["val_alpha_pre_computed_base"]

if 0:
    count = len(data1.filename.unique())
    rows = int(np.ceil(np.sqrt(count)))
    cols = rows
    #fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    for i, (file, d) in enumerate(data1.groupby("filename")):
        ax = axes.ravel()[i]
        ax.plot(d.epoch, d.accuracy)
        ax.plot(d.epoch, d["attack_FGSM_0.010"])
        ax.grid(True)
if 0:
    plt.figure(0, (14, 10))
    def plot(x, y, **kwargs):
        m = d.groupby(x.name).mean()
        s = d.groupby(x.name).sem()
        c, = plt.plot(m.index, m[y.name], **kwargs)
        plt.fill_between(m.index, m[y.name]-s[y.name], m[y.name]+s[y.name], alpha=0.5)
    #data1 = data1.query("iter == '0'")
    for indices, names, d in grid_iterator(data1, "reg_strength", "reg_target"):

        plot(d.epoch, d.accuracy, label="acc")
        plot(d.epoch, d.val_accuracy, label="val_acc")
        plot(d.epoch, d["attack_FGSM_0.020"], label="FGSM_0.02")
        plot(d.epoch, d["attack_FGSM_0.040"], label="FGSM_0.04")
        plot(d.epoch, d["attack_FGSM_0.060"], label="FGSM_0.06")
        plot(d.epoch, d["attack_FGSM_0.080"], label="FGSM_0.08")
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(__file__[:-3] + output_path + "accuracy_vs_epoch.png")
    plt.show()
if 0:
    plt.figure(1, (14, 10))
    def plot(x, y, **kwargs):
        m = d.groupby(x.name).mean()
        s = d.groupby(x.name).sem()
        c, = plt.plot(m.index, m[y.name], **kwargs)
        plt.fill_between(m.index, m[y.name]-s[y.name], m[y.name]+s[y.name], alpha=0.5)
    #data1 = data1.query("iter == '0'")
    for indices, names, d in grid_iterator(data1, "reg_strength", "reg_target"):
        plot(d.epoch, d["alpha"], label="acc")
        plot(d.epoch, d["val_alpha"], label="val_acc")
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("alpha")
        plt.axhline(names[2], color="k")
    plt.legend()
    plt.savefig(__file__[:-3] + output_path + "alpha_vs_epoch.png")
    plt.show()
if 0:
    plt.figure(2, (14, 10))
    def plot(x, y, **kwargs):
        #c, = plt.plot([], [], **kwargs)

        m = d.groupby("epoch").mean()
        s = d.groupby("epoch").sem()

        plot_color_grad(m[x.name], m[y.name], m.index, yerr=s[y.name], **kwargs)
        return
        for bin in range(0, 50, 10):
            m = d.query(f"{bin} <= epoch <= {bin+10}").groupby("epoch").mean()
            s = d.query(f"{bin} <= epoch <= {bin+10}").groupby("epoch").sem()
            plt.plot(m[x.name], m[y.name], color=c.get_color(), alpha=1-bin/40, **kwargs)
            plt.fill_between(m[x.name], m[y.name]-s[y.name], m[y.name]+s[y.name], color=c.get_color(), alpha=0.5*(1-bin/40))
    #data1 = data1.query("iter == '0'")
    for indices, names, d in grid_iterator(data1, "reg_strength", "reg_target"):#, sharex=False, sharey=False):
    #if 1:
        #d = data1
        #names = [None, 1, 1]

        plot(d.alpha, d.accuracy, label="acc")
        plot(d.alpha, d.val_accuracy, label="val_acc")
        plot(d.alpha, d["attack_FGSM_0.020"], label="FGSM_0.02")
        plot(d.alpha, d["attack_FGSM_0.040"], label="FGSM_0.04")
        plot(d.alpha, d["attack_FGSM_0.060"], label="FGSM_0.06")
        plot(d.alpha, d["attack_FGSM_0.080"], label="FGSM_0.08")
        plt.grid(True)
        print(names)
        plt.axvline(float(names[2]), color="k")
        plt.xlabel("alpha")
        plt.ylabel("accuracy")
        #plt.ylim(0, 1)
        #plt.xlim(0.6, 1.4)
        #plt.xticks([0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3])
    plt.legend()
    plt.savefig(__file__[:-3] + output_path + "accuracy_vs_alpha.png")
    plt.show()

def get_max(data, groupby="filename", col="val_accuracy"):
    new_data = []
    for filename, d in data.groupby(groupby):
        new_data.append(d.iloc[np.argmax(d[col])])
    return pd.DataFrame(new_data)

def get_corruption_to_level(data: pd.DataFrame, types=["brightness"]):
    new_data = []
    for i, row in data.iterrows():
        row = row.to_dict()
        for type in types:
            #new_data.append({**row, **dict(strength=0, corrupt=type, value=row["val_accuracy"])})
            for t in [col for col in data.columns if col.startswith(type)]:
                i = float(t.split("_")[-1])
                new_data.append({**row, **dict(strength=i, corrupt=type, value=row[t])})
    return pd.DataFrame(new_data)


data0 = data1
#data1 = get_max(data1)
#data1 = get_max(data1, "epoch")
data1 = data1.query("epoch == 49")
data1 = get_corruption_to_level(data1, ["attack_FGSM", "attack_PGD"])
#data1 = get_corruption_to_level(data1, ["attack_FGSM"])

#plt.figure(4, (14, 10))#, constrained_layout=True)
fig, axes = plt.subplots(3, 2, sharex="row", sharey="row")

data1["alpha"] = np.round(data1["alpha"], 1)

data1 = data1.query("reg_strength > 0.09 or reg_strength < 0.00000001")
index = (data1.strength == 0) & (data1.corrupt == "attack_FGSM")

data1.loc[index, "strength"] = 0.05
data1.loc[index, "corrupt"] = "None"

data1 = data1.query("strength == 0.05")


for index, (cor, dd) in enumerate(data1.groupby("corrupt")):
    if index >= 1:
        plt.sca(axes[1, index-1])
    else:
        plt.sca(axes[0, index])
    for i2, (strength, d) in enumerate(dd.groupby("reg_strength")):
        def plot(x, y, **kwargs):
            m = d.groupby(x.name).mean()
            s = d.groupby(x.name).sem()
            c, = plt.plot(m.index, m[y.name], **kwargs, zorder=10-i2)
            plt.fill_between(m.index, m[y.name] - s[y.name], m[y.name] + s[y.name], alpha=0.5, zorder=10-i2)

        #plt.plot(d.alpha, d.value, "o", label=strength, ms=2)
        plot(d.alpha, d.value, label=strength, ms=2)
        plt.grid()
        plt.xlabel("alpha")
        plt.ylabel("accuracy")
        plt.grid(True)
        plt.title(cor)
plt.legend(title="reg_target")
plt.ylim(0, 1)


for index, (cor, dd) in enumerate(data0.query("reg_strength > 0.09").groupby("reg_strength")):
    print("index", index, cor)
    def plot(x, y, **kwargs):
        m = d.groupby(x.name).mean()
        s = d.groupby(x.name).sem()
        c, = plt.plot(m.index, m[y.name], **kwargs)
        plt.fill_between(m.index, m[y.name]-s[y.name], m[y.name]+s[y.name], alpha=0.5)

    plt.sca(axes[2, index])
    print(dd.columns)
    N = len(dd.groupby("reg1value"))
    if index == 0:
        cmap = pylustrator.lab_colormap.LabColormap(["gray", "C1"], N)
    else:
        cmap = pylustrator.lab_colormap.LabColormap(["gray", "C2"], N)
    for i2, (strength, d) in enumerate(dd.groupby("reg1value")):
        plot(d.epoch, d["alpha"], label="acc", color=cmap(i2))
        #plot(d.epoch, d["val_alpha"], label="val_acc")
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("alpha")
    plt.title(cor)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(13.990000/2.54, 8.950000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.37999999999999995, 5.22)
plt.figure(1).axes[0].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[0].set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.112188, 0.578704, 0.218342, 0.340085])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].title.set_fontsize(10)
plt.figure(1).axes[0].title.set_text("no attack")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.449291, 1.063617])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[1].set_position([1.023559, 0.499289, 0.355019, 0.296493])
plt.figure(1).axes[2].set_xlim(0.37999999999999995, 5.22)
plt.figure(1).axes[2].set_ylim(0.0, 1.0)
plt.figure(1).axes[2].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[2].set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[2].set_position([0.519714, 0.578704, 0.218342, 0.340085])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].title.set_fontsize(10)
plt.figure(1).axes[2].title.set_text("FGSM")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.352252, 1.063617])
plt.figure(1).axes[2].texts[0].set_text("b")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[3].set_xlim(0.37999999999999995, 5.22)
plt.figure(1).axes[3].set_ylim(0.0, 1.0)
plt.figure(1).axes[3].set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[3].set_xticklabels(["1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="center")
plt.figure(1).axes[3].legend(handlelength=1.2999999999999998, handletextpad=0.4, title="strength", fontsize=10.0, title_fontsize=10.0)
plt.figure(1).axes[3].set_position([0.763531, 0.578704, 0.218342, 0.340085])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].title.set_fontsize(10)
plt.figure(1).axes[3].title.set_text("PGD")
plt.figure(1).axes[3].get_legend()._set_loc((-2.117189, 0.389776))
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([-0.141299, 1.063617])
plt.figure(1).axes[3].texts[0].set_text("c")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("")
plt.figure(1).axes[4].set_ylim(0.0, 5.222513057554038)
plt.figure(1).axes[4].set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[4].set_yticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[4].set_position([0.126611, 0.133400, 0.355019, 0.296493])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].title.set_text("")
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_position([-0.138320, 1.011562])
plt.figure(1).axes[4].texts[0].set_text("d")
plt.figure(1).axes[4].texts[0].set_weight("bold")
plt.figure(1).axes[5].set_ylim(0.0, 5.222513057554038)
plt.figure(1).axes[5].set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(1).axes[5].set_yticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Roboto Condensed", horizontalalignment="right")
plt.figure(1).axes[5].set_position([0.544343, 0.133400, 0.355019, 0.296493])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].title.set_text("")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.070856, 1.011562])
plt.figure(1).axes[5].texts[0].set_text("e")
plt.figure(1).axes[5].texts[0].set_weight("bold")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".pdf")
plt.show()

