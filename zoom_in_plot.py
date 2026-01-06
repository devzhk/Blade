#%%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%%
root_dir = 'exps/inference/navier-stokes-ds2'
algo_name = 'EKS-DM'
noise = 2
exp_name = 'noise2-best' # f'noise{noise}-best2-step25-50'
test_id = 3
member_ids = list(range(512))
result_dir = os.path.join(root_dir, algo_name, exp_name)
save_dir = os.path.join('figs', algo_name, f'results_{test_id}')
os.makedirs(save_dir, exist_ok=True)
# Load the data
results = torch.load(os.path.join(result_dir, f'result_{test_id}.pt'), map_location=torch.device('cpu'))
print(results['target'].shape)
print(results['observation'].shape)
norm = plt.Normalize(vmin=-10, vmax=10)

#%%
plot_obs = True


fig, axes = plt.subplots(1, 1, figsize=(5, 4), layout='constrained',
                         sharex=True, sharey=True)
# im0 = axes.imshow(results['target'][0, 0], cmap='Spectral', norm=norm)
# axes[0].set_title('Ground truth', fontsize=15)
# axes.set_xticks([]); axes.set_yticks([])

save_dir = os.path.join(save_dir, f'noise{noise}')
os.makedirs(save_dir, exist_ok=True)
if plot_obs:
    im1 = axes.imshow(results['observation'][0, 0], cmap='Spectral', norm=norm)
    axes.set_xticks([]); axes.set_yticks([])
    cax = inset_axes(axes, 
                    width="65%",      # 60 % of the figure width
                    height="2.5%",      # 3 % of the figure height
                    loc='lower center',
                    bbox_to_anchor=(0, -0.08, 1, 1),  # y = –0.08 puts it just below
                    bbox_transform=fig.transFigure,
                    borderpad=0)
    fig.colorbar(im1, cax=cax, orientation='horizontal')
    # transparent background
    fig.patch.set_alpha(0.0)
    fig.savefig(os.path.join(save_dir, f'{algo_name}-obs.png'), bbox_inches='tight', dpi=300)
else:
    im0 = axes.imshow(results['target'][0, 0], cmap='Spectral', norm=norm)
    axes.set_xticks([]); axes.set_yticks([])
    fig.patch.set_alpha(0.0)
    fig.savefig(os.path.join(save_dir, f'{algo_name}-data.png'), bbox_inches='tight', dpi=300)
#%%