import pandas as pd
import matplotlib.pyplot as plt


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

    row_count = len(iterator_list[1])
    additional = len(iterator_list[2])
    set_ylabel = plt.ylabel
    set_xlabel = plt.xlabel
    ax = None
    ax_list = []
    for indices, names, d in iterate(data, iterator_list[:3]):
        i0 = indices[0]
        plt.figure(i0)
        if names[0]:
            plt.gcf().suptitle(f"{figures} {names[0]}")
        i1 = indices[1]
        i2 = indices[2]
        name1 = names[1]
        name2 = names[2]

        ax = plt.subplot(row_count, additional, 1 + i1 * additional + i2, sharex=ax if sharex else None, sharey=ax if sharey else None)
        ax_list.append([(i0, i1, i2), (names[0], name1, name2), ax])
        if i1 == 0 and name2 is not None:
            if i2 == int(additional // 2):
                plt.text(0.5, 1, f"$\\mathbf{{{cols}}}$\n$\\mathbf{{{name2}}}$", transform=ax.transAxes, ha="center", va="bottom")
            else:
                plt.text(0.5, 1, f"$\\mathbf{{{name2}}}$", transform=ax.transAxes, ha="center", va="bottom")

        if despine:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if len(iterator_list[3:]):
            for indices2, names2, d2 in iterate(d, iterator_list[3:]):
                yield indices + indices2, names + names2, d2
        else:
            yield indices, names, d

    # iterate over all axes to set the additional x and y labels
    for (i0, i1, i2), (name0, name1, name2), ax in ax_list:
        plt.sca(ax)
        if i1 != row_count - 1 and sharex:
            ax.set_xlabel("")
            plt.setp(plt.gca().get_xticklabels(), visible=False)

        if i2 != 0 and sharey:
            ax.set_ylabel("")
            plt.setp(plt.gca().get_yticklabels(), visible=False)
            if len(ax.get_xticklabels()):
                ax.get_xticklabels()[0].set_visible(False)

        if i2 == 0 and name1 is not None:
            if i1 == int(row_count // 2):
                ax.set_ylabel(f"$\\mathbf{{{rows}}}$\n$\\mathbf{{{name1}}}$\n{ax.get_ylabel()}")
            else:
                ax.set_ylabel(f"$\\mathbf{{{name1}}}$\n{ax.get_ylabel()}")