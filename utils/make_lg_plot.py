import matplotlib.pyplot as plt 
import os
import torch 
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec


def plot_samples_on_ax(ax,
                       sample_dict,
                       ax_marg_x=None,
                       ax_marg_y=None,
                       xmin=None, xmax=None,
                       ymin=None, ymax=None,
                       font_size=18,
                       **kwargs):
    post = sample_dict['posterior']
    samp = sample_dict['samples']
    
    sns.scatterplot(
        x=post[:, 0], y=post[:, 1],
        ax=ax,
        s=10,
        alpha=0.5,
        edgecolor='none',
        **kwargs
    )
    
    sns.scatterplot(
        x=samp[:, 0], y=samp[:, 1],
        ax=ax,
        s=10,
        alpha=0.5,
        edgecolor='none',
        **kwargs
    )
    
    if ax_marg_x is not None:
        sns.kdeplot(
            x=post[:, 0],
            ax=ax_marg_x,
            fill=False,
            alpha=0.4,
            color="C0",
            legend=False
        )
        sns.kdeplot(
            x=samp[:, 0],
            ax=ax_marg_x,
            linewidth=1.5,
            alpha=0.5,
            color="C1",
            legend=False
        )
        ax_marg_x.set_xlim(ax.get_xlim())
        ax_marg_x.set_xticks([])
        ax_marg_x.set_yticks([])

    if ax_marg_y is not None:
        sns.kdeplot(
            y=post[:, 1],
            ax=ax_marg_y,
            fill=False,
            alpha=0.4,
            color="C0",
            legend=False
        )
        sns.kdeplot(
            y=samp[:, 1],
            ax=ax_marg_y,
            linewidth=1.5,
            alpha=0.5,
            color="C1",
            legend=False
        )
        ax_marg_y.set_ylim(ax.get_ylim())
        ax_marg_y.set_xticks([])
        ax_marg_y.set_yticks([])

    ax.set_xticks([])
    ax.set_yticks([])

def plot_samples_joint(sample_dict, save_path=None, xmin=None, xmax=None, ymin=None, ymax=None, 
                       font_size=18, legend_font_size=18, **kwargs):
    data = []
    for label, samples in sample_dict.items():
        for point in samples:
            data.append([point[0], point[1], label])
    
    df = pd.DataFrame(data, columns=['x', 'y', 'label'])
    plot = sns.jointplot(data=df, x='x', y='y', hue='label', alpha=0.5, **kwargs)

    if xmin is not None and xmax is not None:
        plot.ax_joint.set_xlim(xmin, xmax)
        plot.ax_marg_x.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        plot.ax_joint.set_ylim(ymin, ymax)
        plot.ax_marg_y.set_ylim(ymin, ymax)
        
    plot.ax_joint.set_xlabel('')
    plot.ax_joint.set_ylabel('')

    plot.ax_joint.set_xticks([])
    plot.ax_joint.set_yticks([])

    plot.ax_marg_x.set_xticks([])
    plot.ax_marg_y.set_yticks([])

    legend = plot.ax_joint.get_legend()
    if legend:
        legend.remove()
        
    if save_path is not None:
        plot.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(plot.fig) 

    return plot


methods = ['Blade', 'DPG', 'SCG', 'EnKG', 'EKS']

g_pref = 'exps/gaussian/'
gmm_pref = 'exps/gmm/'
suffix = 'default/figs/results.pt'

data = {}

for m in methods:
    g_filename = f'{g_pref}{m}/{suffix}'
    g_dict = torch.load(g_filename, weights_only=False)
    g_dict['samples'] = g_dict[m]
    
    gmm_filename = f'{gmm_pref}{m}/{suffix}'
    gmm_dict = torch.load(gmm_filename, weights_only=False)
    gmm_dict['samples'] = gmm_dict[m]
    
    if m == 'Blade':
        data['Blade'] = {'gaussian': g_dict, 'gmm': gmm_dict}
    else:
        data[m] =  {'gaussian': g_dict, 'gmm': gmm_dict}
        
methods    = ['Blade','EnKG','SCG','DPG','EKS']
dist_types = ['gaussian','gmm']

fig = plt.figure(figsize=(20, 8))
outer = gridspec.GridSpec(2, 5, figure=fig, wspace=0.02, hspace=0.02)

for i, dist in enumerate(dist_types):
    for j, method in enumerate(methods):
        # create a 2×2 sub-Gridspec in this cell
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[i, j],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            wspace=0.0,
            hspace=0.0
        )
        ax_marg_x = fig.add_subplot(gs[0, 0])  # top row, left col
        ax_joint  = fig.add_subplot(gs[1, 0])  # bottom row, left col
        ax_marg_y = fig.add_subplot(gs[1, 1])  # bottom row, right col

        sample_dict = {
            'posterior': data[method][dist]['posterior'],
            'samples':   data[method][dist]['samples']
        }

        plot_samples_on_ax(
            ax=       ax_joint,
            ax_marg_x=ax_marg_x,
            ax_marg_y=ax_marg_y,
            sample_dict=sample_dict
        )
        ax_marg_x.set_ylabel('')
        ax_marg_y.set_xlabel('')
        for marg in (ax_marg_x, ax_marg_y):
            marg.set_facecolor("none")                  # no background
            for spine in marg.spines.values():
                spine.set_visible(False)
            
        if dist == 'gaussian':
            if method == 'Blade':
                ax_marg_x.set_title(f"{method} (Ours)",
                            fontsize=14,
                            pad=8)
            else:
                ax_marg_x.set_title(f"{method}",
                            fontsize=14,
                            pad=8)


plt.tight_layout()
fig.savefig("post_vs_samples_with_marginals.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)