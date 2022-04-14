import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class RowColIterator:
    def __init__(self, data, col_name):
        self.col_name = col_name
        self.data = data

    def __call__(self, data):
        if self.col_name is not None:
            if isinstance(self.col_name, str):
                i1_ = 0
                max_count = len(data.groupby(self.col_name))
                for i1, (name1, d1) in enumerate(data.groupby(self.col_name)):
                    yield (dict(index=i1, value=name1, title=self.col_name, max_count=max_count),
                               i1_), (name1, d1)
                    i1_ += 1
            else:
                i1_ = 0
                max_count1 = len(data.groupby(self.col_name[0]))
                for i1, (name1, d1) in enumerate(data.groupby(self.col_name[0])):
                    max_count2 = len(d1.groupby(self.col_name[1]))
                    for i1b, (name1b, d1b) in enumerate(d1.groupby(self.col_name[1])):
                        yield (dict(index=i1, value=name1, title=self.col_name[0], max_count=max_count1),
                               dict(index=i1b, value=name1b, title=self.col_name[1], max_count=max_count2),
                               i1_), ((name1, name1b), d1b)
                        i1_ += 1
        else:
            yield 0, (None, data)

    def __len__(self):
        if self.col_name is not None:
            if isinstance(self.col_name, str):
                return len(self.data[self.col_name].unique())
            else:
                return sum(len(d[self.col_name[1]].unique()) for name, d in self.data.groupby(self.col_name[0]))
        return 1

    def get_lengths(self):
        if isinstance(self.col_name, str):
            return 1
        return len(self.data.groupby(self.col_name[0]))


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
    col_count = len(iterator_list[2])
    set_ylabel = plt.ylabel
    set_xlabel = plt.xlabel
    ax = None
    ax_list = []

    for indices, names, d in iterate(data, iterator_list[:3]):
        i0 = indices[0]
        if figures is not None:
            plt.figure(i0)
        if names[0]:
            plt.gcf().suptitle(f"{figures} {names[0]}")
        i1 = indices[1]
        i2 = indices[2]
        name1 = names[1]
        name2 = names[2]

        i1_sum = i1
        if isinstance(i1, tuple):
            i1_sum = i1[-1]
            i1 = i1[:-1]

        i2_sum = i2
        if isinstance(i2, tuple):
            i2_sum = i2[-1]
            i2 = i2[:-1]

        ax = plt.subplot(row_count, col_count, 1 + i1_sum * col_count + i2_sum, sharex=ax if sharex else None, sharey=ax if sharey else None)
        ax_list.append([(i0, i1, i2), (names[0], name1, name2), ax])


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
        #plt.sca(ax)

        if i1 != row_count - 1 and sharex:
            ax.set_xlabel("")
            plt.setp(plt.gca().get_xticklabels(), visible=False)

        if i2 != 0 and sharey:
            ax.set_ylabel("")
            plt.setp(plt.gca().get_yticklabels(), visible=False)
            #if len(ax.get_xticklabels()):
            #    ax.get_xticklabels()[0].set_visible(False)

        indices_zero = np.cumsum([i1_["index"] != 0 for i1_ in i1][::-1])
        text = ""

        for en, i2_ in enumerate(i2[::-1]):
            if not indices_zero[en]:
                if i2_["index"] == int(i2_["max_count"] // 2):
                    text = f"$\\mathbf{{{i2_['title']}}}$\n$\\mathbf{{{i2_['value']}}}$\n" + text
                else:
                    text = f"\n$\\mathbf{{{i2_['value']}}}$\n" + text

        plt.text(0.5, 1, text, transform=ax.transAxes, ha="center", va="bottom")

        #if i2 == 0 and name1 is not None:
        #for i2_ in i2:
        indices_zero = np.cumsum([i2_["index"] != 0 for i2_ in i2][::-1])
        text = ""

        for en, i1_ in enumerate(i1[::-1]):
            if not indices_zero[en]:
                if i1_["index"] == int(i1_["max_count"] // 2):
                    text = f"$\\mathbf{{{i1_['title']}}}$\n$\\mathbf{{{i1_['value']}}}$\n" + text
                else:
                    text = f"\n$\\mathbf{{{i1_['value']}}}$\n" + text
        text += f"{ax.get_ylabel()}"
        ax.set_ylabel(text)


def grid_iterator(data, rows=None, cols=None, *additional, figures=None, sharex=True, sharey=True, despine=True):
    iterator_list = [
        RowColIterator(data, figures),
        RowColIterator(data, rows),
        RowColIterator(data, cols),
    ]
    for col in additional:
        iterator_list.append(RowColIterator(data, col))

    row_count = len(iterator_list[1])
    col_count = len(iterator_list[2])

    fig = plt.gcf()
    subfigs = fig.subfigures(iterator_list[1].get_lengths(), iterator_list[2].get_lengths(), wspace=0.07)
    subfigs = np.array(subfigs).ravel()
    i0 = 0
    new_subfigs = []
    print(f"{iterator_list[1].get_lengths()=}, {iterator_list[2].get_lengths()=}")
    for i in range(iterator_list[1].get_lengths()):
        new_subfigs.append([])
        for j in range(iterator_list[2].get_lengths()):
            new_subfigs[-1].append([subfigs[i0], None, None])
            i0 += 1
    subfigs = np.array(new_subfigs)
    print(f"{subfigs=}")

    set_ylabel = plt.ylabel
    set_xlabel = plt.xlabel
    ax = None
    ax_list = []
    for indices, names, d in iterate(data, iterator_list[:3]):
        i0 = indices[0]
        if figures is not None:
            plt.figure(i0)
        if names[0]:
            plt.gcf().suptitle(f"{figures} {names[0]}")
        i1 = indices[1]
        i2 = indices[2]
        name1 = names[1]
        name2 = names[2]

        i1_sum = i1
        if isinstance(i1, tuple):
            if len(i1) > 2:
                fig_index1 = i1[0]["index"]
                ax_index1 = i1[1]["index"]
                ax_max1 = i1[1]["max_count"]
            else:
                fig_index1 = 0
                ax_index1 = i1[0]["index"]
                ax_max1 = i1[0]["max_count"]
            i1_sum = i1[-1]
            i1 = i1[:-1]

        i2_sum = i2
        if isinstance(i2, tuple):
            if len(i2) > 2:
                fig_index2 = i2[0]["index"]
                ax_index2 = i2[1]["index"]
                ax_max2 = i2[1]["max_count"]
            else:
                fig_index2 = 0
                ax_index2 = i2[0]["index"]
                ax_max2 = i2[0]["max_count"]
            i2_sum = i2[-1]
            i2 = i2[:-1]

        sub_fig = subfigs[fig_index1, fig_index2]
        if sub_fig[1] is None:
            gs = sub_fig[0].add_gridspec(ax_max1, ax_max2)
            sub_fig[1] = gs.subplots(sharex=sharex, sharey=sharey)
            try:  # if it is just one axes
                len(sub_fig[1])
            except TypeError:
                sub_fig[1] = np.array([[sub_fig[1]]])
            print(f"{len(sub_fig[1])=}")
            sub_fig[2] = gs

        #ax = plt.subplot(row_count, col_count, 1 + i1_sum * col_count + i2_sum, sharex=ax if sharex else None, sharey=ax if sharey else None)
        ax = sub_fig[1][ax_index1, ax_index2]
        ax.indices = [i1, i2]
        fig.sca(ax)
        ax_list.append([(i0, i1, i2), (names[0], name1, name2), ax])


        if despine:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if len(iterator_list[3:]):
            for indices2, names2, d2 in iterate(d, iterator_list[3:]):
                yield indices + indices2, names + names2, d2
        else:
            yield indices, names, d

    # Hide x labels and tick labels for all but bottom plot.
    from matplotlib.patches import ConnectionPatch
    for sub in subfigs.reshape(-1, 3):
        axes = sub[1]
        gs = sub[2]
        for ax in axes.ravel():
            #ax.set_xlabel("")
            ax.label_outer()
            con = ConnectionPatch(
                xyA=(0, 0), coordsA=ax.transAxes,
                xyB=(1, 1), coordsB=ax.transAxes,
                arrowstyle="-", shrinkB=0)
            fig.add_artist(con)

        row = sub[0].add_subplot(gs[:, :])
        # the '\n' is important

        # hide subplot
        row.set_frame_on(False)
        row.set_xticks([])
        row.set_yticks([])
        #row.axis('off')

        if len(axes[0, 0].indices[0]) == 2:
            for ax in axes[:, 0]:
                ax.set_ylabel(f'$\\mathbf{{{ax.indices[0][1]["value"]}}}$'+"\n"+str(ax.get_ylabel()))
            row.set_ylabel(f'$\\mathbf{{{ax.indices[0][0]["value"]}}}$'+"\n"+str(ax.indices[0][1]["title"])+"\n", fontweight='semibold')
            fig.supylabel(ax.indices[0][0]["title"], fontweight='semibold')
        else:
            for ax in axes[:, 0]:
                ax.set_ylabel(f'$\\mathbf{{{ax.indices[0][0]["value"]}}}$' + "\n" + str(ax.get_ylabel()))
            row.set_ylabel(str(ax.indices[0][0]["title"])+"\n\n\n\n", fontweight='semibold')

            #sub[0].supylabel(str(ax.indices[0][0]["title"]))
            pass

        if len(axes[0, 0].indices[1]) == 2:
            for ax in axes[0, :]:
                ax.set_title(str(ax.indices[1][1]["value"]), fontweight='semibold')
            row.set_title(str(ax.indices[1][0]["value"])+"\n"+str(ax.indices[1][1]["title"])+"\n", fontweight='semibold')
            fig.suptitle(ax.indices[1][0]["title"], fontweight='semibold')
        else:
            for ax in axes[0, :]:
                ax.set_title(str(ax.indices[1][0]["value"]), fontweight='semibold')
            row.set_title(str(ax.indices[1][0]["title"])+"\n", fontweight='semibold')

        if 0:
            import matplotlib.transforms as transforms

            con = ConnectionPatch(
                xyA=(0, 1), coordsA=axes[0, 0].transAxes + transforms.ScaledTranslation(-10*3/72, 0, fig.dpi_scale_trans),
                xyB=(0, 0), coordsB=axes[-1, 0].transAxes + transforms.ScaledTranslation(-10*3/72, 0, fig.dpi_scale_trans),
                arrowstyle="-", shrinkB=0)
            sub[0].add_artist(con)
            #sub[0].suptitle("x")
    #plt.tight_layout()
    print(subfigs.shape)
    ax = subfigs[-1, -1][1][-1, -1]
    ax.figure.sca(ax)


def plot_color_grad(x, y, c, color1=None, color2=None, yerr=None, label=None, N=10):
    import matplotlib as mpl
    import numpy as np
    line1, = plt.plot([], [], color=color1, label=label)
    color1 = np.asarray(mpl.colors.to_rgba(line1.get_color()))
    if color2 is None:
        color2 = np.asarray(mpl.colors.to_rgba("w"))*0.9 + color1*0.1
    else:
        color2 = np.asarray(mpl.colors.to_rgba(color2))
    indices = np.argsort(c)
    if len(indices) < N:
        N = len(indices)
    for i in range(N):
        i1, i2 = int(len(indices)/N*i), int(len(indices)/N*(i+1))
        ind = indices[i1:i2+1]
        f = i/(N-1)
        color = color1*(1-f) + color2*f
        plt.plot(x[ind], y[ind], color=color)
        if yerr is not None:
            plt.fill_between(x[ind], y[ind]-yerr[ind], y[ind]+yerr[ind], color=color, alpha=0.5)


if __name__ == "__main__":
    if 1:
        x = np.arange(0, 12, 0.01)
        y = np.sin(x*10)
        #plt.plot(x, y)
        plot_color_grad(x, y, x, yerr=y*0.1)
        plot_color_grad(x, y+2, x)
        plot_color_grad(x[::-1], y[::-1]+2.5, -x[::-1])
        plt.show()
        exit()
    import numpy as np
    import pylustrator
    pylustrator.start()
    data = []
    for ii in range(2):
      for jj in range(2):
        for i in range(3):
            for j in range(3):
                data.append(dict(jj=jj, ii=ii, i=i, j=j, x=i+10*ii))
    data = pd.DataFrame(data)
    if 1:
        plt.figure(1, constrained_layout=True)
        for indices, names, d in grid_iterator(data, ["ii", "i"], "j"):
            plt.plot(np.arange(3)*d.iloc[0].j, np.arange(3)*d.iloc[0].x)
            plt.xlabel("xx")
            plt.ylabel("yy")
        #plt.tight_layout()
    if 0:
        plt.figure(3, constrained_layout=True)
        for indices, names, d in grid_iterator(data, "j", ["ii", "i"]):
            plt.plot(np.arange(3)*d.iloc[0].j, np.arange(3)*d.iloc[0].x)
            plt.xlabel("xx")
            plt.ylabel("yy")
        #plt.tight_layout()
    if 0:
        plt.figure(3, constrained_layout=True)
        for indices, names, d in grid_iterator(data, ["jj", "j"], ["ii", "i"]):
            plt.plot(np.arange(3)*d.iloc[0].j, np.arange(3)*d.iloc[0].x)
            plt.xlabel("xx")
            plt.ylabel("yy")
        #plt.tight_layout()
    if 0:
        plt.figure(2)
        for indices, names, d in grid_iterator(data, "i", "j","ii"):
            plt.plot(np.arange(3)*d.iloc[0].j, np.arange(3)*d.iloc[0].x)
            plt.xlabel("xx")
            plt.ylabel("yy")
    plt.show()