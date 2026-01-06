#%%
import os
import torch
import numpy as np
import hydra
from hydra.utils import instantiate

from models.gmm import GMScore, true_posterior, EDMPrecond, \
    construct_grid, construct_gmm_dist
from utils.plots import plot_samples_joint
from inverse_problems.linear import LinearOperator
from eval import LinearGaussianEvaluator


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    exp_dir = os.path.join('exps', 'gaussian', config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    fig_dir = os.path.join(exp_dir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    # ------ construct prior   -----
    cov_scale = 2.0
    num_samples = 1024
    num_modes = 1
    # means = construct_n_star(num_modes, scale=11.3137)
    means = construct_grid(num_modes)
    print(f'Prior mean: {means}')
    covs = [cov_scale * np.eye(2) for _ in range(means.shape[0])]
    prior_dist = construct_gmm_dist(means, cov_scale) # default: uniform prior
    prior_samples = prior_dist.sample((num_samples,)).cpu().numpy()
    
    # torch.save(prior_samples, 'prior_samples.pt')

    score_model = GMScore(torch.from_numpy(means).to(device), cov_scale) # uniform prior by default
    model = EDMPrecond(score_model)
    # ------ construct forward model    -----
    H = np.array([[-0.1117, -1.1456]])  # 1 x 2
    noise_std = 1.5

    y = np.array([-6.5626])  # (1,)
    forward_op = LinearOperator(H, device=device, sigma_noise=noise_std)

    # compute the ground truth posterior
    prior_probs = prior_dist.mixture_distribution.probs.cpu().numpy()
    posterior_dist, posterior_means, posterior_covs = true_posterior(y, H, prior_probs, means, covs, noise_std)
    gt_samples = posterior_dist.sample((num_samples,)).cpu().numpy()
    
    # torch.save(gt_samples, 'gt_samples.pt')

    # visualize the prior and posterior samples
    sample_dict = {
        'prior': prior_samples,
        'posterior': gt_samples
    }
    plot_samples_joint(sample_dict, save_path=os.path.join(fig_dir, 'prior_posterior_samples.png'))

    # ------ instantiate algorithm -----
    algo = instantiate(config.algorithm.method, forward_op=forward_op, net=model)
    results = []
    observation = torch.from_numpy(y).to(device)
    if 'dpg' in config.algorithm.method._target_ or 'scg' in config.algorithm.method._target_:
        algo_samples = []
        for i in range(config.num_samples):
            sample = algo.inference(observation.to(forward_op.dtype), num_samples=1).detach()
            algo_samples.append(sample)
        algo_samples = torch.cat(algo_samples, dim=0)
    else:
        algo_samples = algo.inference(observation.to(forward_op.dtype), num_samples=config.num_samples).detach()
    results.append(algo_samples)

    results = torch.cat(results, dim=0).cpu().numpy()
    
    evaluator = LinearGaussianEvaluator(forward_op=forward_op, posterior_mean=torch.from_numpy(posterior_means[0]), 
                                        posterior_cov=torch.from_numpy(posterior_covs))
    metric_dict = evaluator(torch.from_numpy(results).to(torch.float64), torch.from_numpy(gt_samples).to(torch.float64), y, forward_op)
    print(metric_dict)

    sample_dict = {
        'posterior': gt_samples,
        f'{config.algorithm.name}': results,
    }
    # plot_samples_joint(sample_dict, xmin=-10, xmax=10, ymin=-2, ymax=9, save_path=os.path.join(fig_dir, 'algorithm_samples.png'))
    plot_samples_joint(sample_dict, save_path=os.path.join(fig_dir, 'algorithm_samples.png'))
    
    torch.save(sample_dict, os.path.join(fig_dir, 'results.pt'))


if __name__ == '__main__':
    main()