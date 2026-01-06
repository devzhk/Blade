import torch
from tqdm import tqdm
from .base import Algo
import numpy as np

import wandb

from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler



def get_cov_diag(z):
    particles = z.reshape(len(z), -1)
    diff = particles - particles.mean(dim=0, keepdim=True)
    diff_sq = diff ** 2
    diag = diff_sq.sum(dim=0) / len(particles)
    return diag


class EKS_DM(Algo):
    '''
    Ensemble Kalman sampler with diffusion prior, as described in https://arxiv.org/abs/1903.08866 with the prior term modified to use the diffusion prior.
    The mathematical formulation is given in Eq. (2.13) in the paper.
    To ensure the consistency with the large-particle limit, we added the finite-particle correction term from https://arxiv.org/abs/1908.10890. 
    '''
    def __init__(self, 
             net,
             forward_op,
             guidance_scale,
             num_steps,
             gamma=None,
             batch_size=64,
             ensemble_size=1024,
             clean_init=False,
             mode='correction',         # correction, diag, original
             diffusion_scheduler_config={},
             **kwargs):
        super(EKS_DM, self).__init__(net, forward_op, **kwargs)
        self.scale = guidance_scale
        self.N = num_steps

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.mode = mode
        self.ensemble_size = ensemble_size
        self.gamma = gamma if gamma is not None else forward_op.sigma_noise
        self.clean_init = clean_init

        self.diffusion_scheduler = Scheduler(**diffusion_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        
    @torch.no_grad()
    def inference(self,  observation, num_samples=1, **kwargs):
        observation = observation.to(self.dtype)
        device = self.forward_op.device
        
        num_batches = self.ensemble_size // self.batch_size
        if self.clean_init:
            x_initial = torch.randn(self.ensemble_size, *self.net.shape, device=device, dtype=self.dtype) * self.diffusion_scheduler.sigma_max
            sampler = DiffusionSampler(self.diffusion_scheduler)
            for b in range(num_batches):
                start = b * self.batch_size
                end = (b + 1) * self.batch_size
                x_initial[start : end] = sampler.sample(self.net, x_initial[start : end])
        else:
            x_initial = torch.randn(self.ensemble_size, *self.net.shape, device=device, dtype=self.dtype)
        print('Starting inference...')
        
        x_next = x_initial
        # observation noise level
        std_y = self.gamma if self.gamma > 0 else self.forward_op.sigma_noise
        std_y = std_y if std_y > 0 else 1.0
        total_time = 0.0

        # setup for score function
        sigma_min = self.diffusion_scheduler.sigma_min
        scale = self.diffusion_scheduler.scaling_steps[-1]
        sampler = DiffusionSampler(self.diffusion_scheduler)

        pbar = range(self.num_steps)
        for _ in pbar:
            x_diff = (x_next - x_next.mean(dim=0, keepdim=True)).reshape(self.ensemble_size, -1)
            cov = x_diff.T @ x_diff / len(x_diff) # (D, D)
            # compute score for each particle
            scores = torch.empty(self.ensemble_size, *self.net.shape, device=device, dtype=self.dtype)
            for b in range(num_batches):
                start = b * self.batch_size
                end = (b + 1) * self.batch_size
                x_batch = x_next[start:end]
                scores[start:end] = sampler.score(self.net, x_batch / scale, sigma_min) / scale
            prior_term = (scores.reshape(self.ensemble_size, -1) @ cov).reshape(self.ensemble_size, *self.net.shape)
            correction_term = x_diff.reshape(self.ensemble_size, *self.net.shape) * (x_diff.shape[-1] + 1) / self.ensemble_size

            neg_likelihood_term, loss = self.ek_update(self.forward_op, observation, std_y, x_next, x_next, return_loss=True)

            drift = - neg_likelihood_term + prior_term + correction_term

            lr = self.scale / torch.linalg.matrix_norm(drift.reshape(self.ensemble_size, -1))
            total_time += lr

            x_next += drift * lr
            if self.mode == 'correction':
                eps = torch.randn(self.ensemble_size, self.ensemble_size, device=device, dtype=self.dtype)
                noise = eps @ x_diff / np.sqrt(self.ensemble_size) * torch.sqrt(2 * lr)
            elif self.mode == 'original':
                eps = torch.randn_like(x_next).reshape(self.ensemble_size, -1)
                cov_sqrt = torch.linalg.cholesky(cov + 1e-3 * torch.eye(len(cov), device=device))
                noise = (eps @ cov_sqrt) * torch.sqrt(2 * lr)
            else:
                raise ValueError(f"only 'correction', 'original' modes are expected, but got {self.mode}")
            x_next += noise.reshape(self.ensemble_size, *self.net.shape)
            if wandb.run is not None:
                wandb.log({'data_misfit': loss.item(), 'step size': lr})
        print(f'time horizon: {total_time}')
        return x_next
            
    @torch.no_grad()
    def ek_update(self, forward_operator, y, std_y, x, x_clean, return_loss=False):
    
        N, *spatial = x.shape
        
        preds = forward_operator.forward(x_clean)        
        
        xs_diff = x - x.mean(dim=0, keepdim=True)
        pred_err = (preds - y)  # (N, *spatial)
        pred_diff = preds - preds.mean(dim=0, keepdim=True) # (N, *spatial)
            
        coef = (
            torch.matmul(
                pred_err.reshape(pred_err.shape[0], -1) / (std_y ** 2),
                pred_diff.reshape(pred_diff.shape[0], -1).T,
            )
            / N
        )   # (N, N)
                
        dx = (coef @ xs_diff.reshape(N, -1)).reshape(N, *spatial)
        if return_loss:
            loss = torch.linalg.norm(pred_err.reshape(pred_err.shape[0], -1), dim=1).mean()
            return dx, loss
        else:
            return dx