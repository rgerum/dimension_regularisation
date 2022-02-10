import matplotlib.pyplot as plt
import pandas as pd
from scripts.net_helpers import read_data
import numpy as np
import pylustrator
from dimension_regularisation.pandas_grid_plot import grid_iterator
#pylustrator.start()

output_add = "_cifar10_gamma"
data1 = read_data(r"../../results/cedar_logs_expcifar3/iter-{iter}_reg1-{reg1}_reg1value-{reg1value}/", file_name="data.csv")
print(data1.filename.unique())
data1["reg1"] = pd.to_numeric(data1["reg1"])
data1["reg1value"] = pd.to_numeric(data1["reg1value"])

class RowColIterator:
    def __init__(self, data, col_name):
        self.col_name = col_name
        self.data = data

    def __call__(self, data):
        if self.col_name is not None:
            for i1, (name1, d1) in enumerate(data.groupby(self.col_name)):
                yield i1, (name1, d1)
        else:
            yield 0, (None, data)

    def __len__(self):
        if self.col_name is not None:
            return len(self.data[self.col_name].unique())
        return 1

def iterate(data, generators, indices=(), names=()):
    gen = generators[0](data)
    if len(generators) == 1:
        for i1, (name1, d1) in gen:
            yield indices + (i1,), names + (name1,), d1
    else:
        for i1, (name1, d1) in gen:
            yield from iterate(d1, generators[1:], indices + (i1,), names + (name1,))


def grid_iterator(data, rows=None, cols=None, *additional, figures=None, sharex=True, sharey=True, despine=True):
    iterator_list = [
        RowColIterator(data, figures),
        RowColIterator(data, rows),
        RowColIterator(data, cols),
    ]
    for col in additional:
        iterator_list.append(RowColIterator(data, col))

    rows = len(iterator_list[1])
    additional = len(iterator_list[2])
    set_ylabel = plt.ylabel
    set_xlabel = plt.xlabel
    ax = None
    for indices, names, d in iterate(data, iterator_list[:3]):
        i0 = indices[0]
        plt.figure(i0)
        if names[0]:
            plt.gcf().suptitle(f"{figures} {names[0]}")
        i1 = indices[1]
        i2 = indices[2]
        name1 = names[1]
        name2 = names[2]

        ax = plt.subplot(rows, additional, 1 + i1 * additional + i2, sharex=ax if sharex else None, sharey=ax if sharey else None)
        if i1 == 0 and name2 is not None:
            if i2 == int(additional // 2):
                plt.title(f"{cols}\n{name2}")
            else:
                plt.title(name2)

        if i1 != rows-1 and sharex:
            plt.xlabel = lambda x: set_xlabel("")
            plt.setp(plt.gca().get_xticklabels(), visible=False)
        else:
            plt.xlabel = set_xlabel

        if i2 != 0 and sharey:
            plt.ylabel = lambda x: set_ylabel("")
            plt.setp(plt.gca().get_yticklabels(), visible=False)
        else:
            plt.ylabel = set_ylabel

        if i2 == 0 and name1 is not None:
            if i1 == int(rows//2):
                plt.ylabel = lambda x: set_ylabel(f"{rows}\n{name1}\n{x}")
            else:
                plt.ylabel = lambda x: set_ylabel(f"{name1}\n{x}")
            plt.ylabel("")

        if despine:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if len(iterator_list) > 2:
            for indices2, names2, d2 in iterate(d, iterator_list[3:]):
                yield indices + indices2, names + names2, d2
        else:
            yield indices, names, d


if 0:
    print(data1["reg1"].unique())
    #print(data1["reg1", "reg1value"].unique())
    data1 = data1.query("reg1 > 0.9")
    print(data1["reg1value"].unique())
    data1 = data1.query("reg1value == 0.6")
    plt.plot(data1.epoch, data1.alpha, "o")
    plt.show()
    exit()
elif 0:
    for name1, name2, d in grid_iterator(data1, "reg1value", "reg1"):
        for xx, d00 in d.groupby("iter"):
            d = d00.groupby("epoch")[f"alpha"].agg(["mean", "sem"])
            p, = plt.plot(d.index, d["mean"], label=name2)
        plt.grid()
        # plt.fill_between(d.index, d["mean"] - d["sem"], d["mean"] + d["sem"], color=p.get_color(), alpha=0.5)
        # plt.plot(d.epoch, d.accuracy_brightness_1)
        plt.axhline(float(name1))
elif 0:
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
                new_data.append({**row, **dict(strength=0, corrupt=type, value=row["val_accuracy"])})
                for i in range(1, 6):
                    new_data.append({**row, **dict(strength=i, corrupt=type, value=row[f"accuracy_{type}_{i}"])})
        return pd.DataFrame(new_data)

    data1 = get_max(data1)
    data1 = get_corruption_to_level(data1, ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"])


    def do_fit(x, y):
        ax = plt.gca()
        plt.sca(last_ax)
        p = np.polyfit(x, y, 2)
        maxi = -p[1] / (2 * p[0])
        print(p, maxi)
        pp = np.poly1d(p)
        xx = np.arange(np.min(x), np.max(x), 0.01)
        plt.plot(xx, pp(xx), "k")
        plt.axvline(maxi, color="k", lw=0.8, linestyle="dashed")
        plt.sca(ax)

    corrupt = ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"][0]
    last_ax = None
    for indices, names, d in grid_iterator(data1, "corrupt", "strength", "reg1value"):
        if indices[3] == 0:
            if last_ax is not None:
                do_fit(x, y)
            x = []
            y = []
            last_ax = plt.gca()
        print(indices, names)
        plt.plot(d.alpha, d.value, "o", label=names[-1])
        for reg1, d0 in d.groupby("reg1"):
            plt.plot(d0.alpha, d0.value, "--k")
        x.extend(d.alpha)
        y.extend(d.value)
        plt.grid()
        plt.xlabel("alpha")
        plt.ylabel("accuracy")
    print(d[["reg1", "iter"]])
    do_fit(x, y)
    plt.legend()
    plt.show()
    exit()
    index = 0
    ax = None
    data11 = data1
    corrupt = ["brightness", "contrast", "defocus_blur", "elastic", "gaussian_noise"][0]
    for index0, (name, data1) in enumerate(data11.groupby("reg1value")):
        for i in range(0, 6):
            if ax is None:
                ax = plt.subplot(5, 6, i + 1 + index0 * 6)
            else:
                ax = plt.subplot(5, 6, i + 1 + index0*6, sharex=ax, sharey=ax)
            for name, d0 in data1.groupby("reg1"):
                name = str(name)
                if i == 0:
                    plt.title("val_acc")
                    d = d0.groupby("epoch")[f"val_accuracy"].agg(["mean", "sem"])
                else:
                    plt.title(f"bright_{i}")
                    d = d0.groupby("epoch")[f"accuracy_{corrupt}_{i}"].agg(["mean", "sem"])
                maxi = np.argmax(d["mean"])
                print(name, d["mean"][maxi], d["sem"][maxi])
                if 0:
                    plt.bar(name, d["mean"][maxi], yerr=d["sem"][maxi], label=name, capsize=5)
                else:
                    plt.errorbar([name], d["mean"][maxi], yerr=d["sem"][maxi], label=name, capsize=5)
            index += 1
            plt.xlabel("reg")
            plt.ylabel("accuracy")
    pylustrator.helper_functions.axes_to_grid()
plt.legend()
#plt.ylim(bottom=0.5)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
#% end: automatic generated code from pylustrator
plt.show()

