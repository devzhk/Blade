import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch


def add_zoom_inset(field,
                   ax,                 # axes to add inset to
                   rect_xy=(80, 60),   # (x0, y0) of patch
                   rect_xy2=(40, 40),  # (x0, y0) of patch2
                   rect_wh=(32, 32),   # (w,  h)
                   zoom=3,
                   vmin=-10, vmax=10,
                   cmap='Spectral'):

    x0, y0   = rect_xy
    x2, y2   = rect_xy2
    w,  h    = rect_wh
    H, W     = field.shape
    norm     = plt.Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------ main
    # fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(field, cmap=cmap, norm=norm)
    ax.set_xticks([]); ax.set_yticks([])

    edge_color = 'dimgrey'
    # rectangle on full field
    rect = Rectangle((x0, y0), w, h,
                     linewidth=1.5, edgecolor=edge_color, facecolor='none')
    ax.add_patch(rect)

    rect2 = Rectangle((x2, y2), w, h,
                     linewidth=1.5, edgecolor=edge_color, facecolor='none')
    ax.add_patch(rect2)

    # ------------------------------------------------------------ inset axes
    ax_ins = inset_axes(ax,
                        width = w*zoom/W,
                        height= h*zoom/H,
                        bbox_to_anchor=(1.05, - 0.57, 1, 1),  # x0,y0,w,h in ax coords
                        bbox_transform=ax.transAxes,
                        loc='upper left',
                        borderpad=0)

    # show entire field in inset, then set view to patch
    ax_ins.imshow(field, cmap=cmap, norm=norm, interpolation='lanczos')
    ax_ins.set_xlim(x0, x0 + w)
    ax_ins.set_ylim(y0 + h, y0)   # flip y so orientation matches
    ax_ins.set_xticks([]); ax_ins.set_yticks([])
    ax_ins.set_clip_on(False)

    # -------------------------------------------------------- connector lines
    #  - from upper-left of rect to upper-left of inset
    #  - from lower-left of rect to lower-left of inset
    con1 = ConnectionPatch(xyA=(x0,       y0),        coordsA=ax.transData,
                           xyB=(0,        1),         coordsB=ax_ins.transAxes,
                           color=edge_color, linewidth=1)
    con2 = ConnectionPatch(xyA=(x0+h,   y0 + h),    coordsA=ax.transData,
                           xyB=(1,        0),         coordsB=ax_ins.transAxes,
                           color=edge_color, linewidth=1)
    ax.figure.add_artist(con1)
    ax.figure.add_artist(con2)
    # add another axes
    ax_ins2 = inset_axes(ax,
                        width = w*zoom/W,
                        height= h*zoom/H,
                        bbox_to_anchor=(1.05, - 0.07, 1, 1),  # x0,y0,w,h in ax coords
                        bbox_transform=ax.transAxes,
                        loc='upper left',
                        borderpad=0)
    # show entire field in inset, then set view to patch
    ax_ins2.imshow(field, cmap=cmap, norm=norm, interpolation='lanczos')
    ax_ins2.set_xlim(x2, x2 + w)
    ax_ins2.set_ylim(y2 + h, y2)   # flip y so orientation matches
    ax_ins2.set_xticks([]); ax_ins2.set_yticks([])
    ax_ins2.set_clip_on(False)
    # -------------------------------------------------------- connector lines
    #  - from upper-left of rect to upper-left of inset
    #  - from lower-left of rect to lower-left of inset
    con3 = ConnectionPatch(xyA=(x2+h,       y2),        coordsA=ax.transData,
                           xyB=(1,        1),      coordsB=ax_ins2.transAxes,
                            color=edge_color, linewidth=1)
    con4 = ConnectionPatch(xyA=(x2,   y2 + h),    coordsA=ax.transData,
                           xyB=(0,        0),      coordsB=ax_ins2.transAxes,
                           color=edge_color, linewidth=1)
    ax.figure.add_artist(con3)
    ax.figure.add_artist(con4)
    # --------------------------------------------------------------- title
    # plt.show()


def plot_samples_joint(sample_dict, save_path=None, show_legend=True, **kwargs):
    """
    Plot multiple sets of 2D samples using seaborn's jointplot.
    
    Args:
        sample_dict: dict, keys are labels and values are numpy arrays of shape (n_samples, 2)
        save_path: str, path to save the plot. If None, the plot is displayed.
        show_legend: bool, whether to show the legend. Default is False.
    """
    data = []
    for label, samples in sample_dict.items():
        for point in samples:
            data.append([point[0], point[1], label])
    
    df = pd.DataFrame(data, columns=['x', 'y', 'label'])
    plot = sns.jointplot(data=df, x='x', y='y', hue='label', alpha=0.5, **kwargs)
    
    if not show_legend:
        legend = plot.ax_joint.get_legend()
        if legend:
            legend.remove()
    
    if save_path is not None:
        plot.savefig(save_path, bbox_inches='tight', dpi=150)
    return plot