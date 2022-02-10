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


def grid_iterator(data, col1=None, col2=None, *cols, sharex=True, sharey=True, despine=True):
    iterator_list = [
        RowColIterator(data, col1),
        RowColIterator(data, col2),
    ]
    for col in cols:
        iterator_list.append(RowColIterator(data, col))

    rows = len(iterator_list[0])
    cols = len(iterator_list[1])
    set_ylabel = plt.ylabel
    set_xlabel = plt.xlabel
    ax = None
    for indices, names, d in iterate(data, iterator_list[:2]):
        i1 = indices[0]
        i2 = indices[1]
        name1 = names[0]
        name2 = names[1]

        ax = plt.subplot(rows, cols, 1+i1*cols+i2, sharex=ax if sharex else None, sharey=ax if sharey else None)
        if i1 == 0 and name2 is not None:
            if i2 == int(cols // 2):
                plt.title(f"{col2}\n{name2}")
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
                plt.ylabel = lambda x: set_ylabel(f"{col1}\n{name1}\n{x}")
            else:
                plt.ylabel = lambda x: set_ylabel(f"{name1}\n{x}")
            plt.ylabel("")

        if despine:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if len(iterator_list) > 2:
            for indices2, names2, d2 in iterate(d, iterator_list[2:]):
                yield indices + indices2, names + names2, d2
        else:
            yield indices, names, d
