import torch
from tqdm import tqdm
from .base import Algo
from .enkg import ode_sampler
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


class EKS(Algo):
    '''
    The original Ensemble Kalman Sampler, as described in https://arxiv.org/abs/1903.08866.
    The mathematical formulation is given in Eq. (2.13) in the paper.
    To ensure the consistency with the large-particle limit, we added the finite-particle correction term from https://arxiv.org/abs/1908.10890. 
    '''
    def __init__(self, 
             net,
             forward_op,
             guidance_scale,
             num_steps,
             gamma=None,
             prior_std=1.0,
             prior_mean=0.0,
             batch_size=64,
             init_ensemble=1024,
             clean_init=False,
             mode='correction',         # correction, diag, original
             diffusion_scheduler_config={},
             **kwargs):
        super(EKS, self).__init__(net, forward_op, **kwargs)
        self.scale = guidance_scale
        self.N = num_steps
        self.prior_mean = prior_mean
        self.prior_std = prior_std

        self.num_l_steps = num_steps
        self.batch_size = batch_size
        self.mode = mode
        self.init_ensemble = init_ensemble
        self.gamma = gamma if gamma is not None else forward_op.sigma_noise
        # self.growth_rate = growth_rate
        # self.ensemble_schedule = create_step_scheduler(init_ensemble, growth_rate, scheduler_steps, num_steps)
        self.clean_init = clean_init

        self.diffusion_scheduler = Scheduler(**diffusion_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        
    @torch.no_grad()
    def inference(self,  observation, num_samples=1, **kwargs):
        observation = observation.to(self.dtype)
        device = self.forward_op.device
        
        if self.clean_init:
            x_initial = torch.randn(self.init_ensemble, *self.net.shape, device=device, dtype=self.dtype) * self.diffusion_scheduler.sigma_max
            sampler = DiffusionSampler(self.diffusion_scheduler)
            num_batches = len(x_initial) // self.batch_size
            for b in range(num_batches):
                start = b * self.batch_size
                end = (b + 1) * self.batch_size
                x_initial[start : end] = sampler.sample(self.net, x_initial[start : end])
        else:
            x_initial = torch.randn(self.init_ensemble, *self.net.shape, device=device, dtype=self.dtype) * self.prior_std + self.prior_mean
        print('Starting inference...')
        
        x = x_initial
        x = self.ll_step(observation, x)
        return x
    
    @torch.no_grad()
    def ll_step(self, y, particles):
        x = self.prior_mean
        rho = self.prior_std
        z_next = particles.clone()
        
        J, *spatial = particles.shape
        
        total_time = 0.
                        
        pbar = range(self.num_l_steps)
        for _ in pbar:
            
            z_diff = (z_next - z_next.mean(dim=0, keepdim=True)).reshape(J, -1)
            
            if self.mode == 'diag':
                cov_diag = get_cov_diag(z_next)
                dz_reg = ((x - z_next).reshape(J, -1) * cov_diag).reshape(J, *spatial) / (rho ** 2) 
            else:
                cov = z_diff.T @ z_diff / len(z_diff)
                dz_reg = ((x - z_next).reshape(J, -1) @ cov).reshape(J, *spatial) / (rho ** 2)
            
            if self.mode == 'correction':
                dz_reg = dz_reg + z_diff.reshape(J, *spatial) * (z_diff.shape[-1] + 1) / J
            
            std_y = self.gamma if self.gamma > 0 else self.forward_op.sigma_noise
            std_y = std_y if std_y > 0 else 1.0
            dz_ll, loss = self.ek_update(self.forward_op, y, std_y, 
                                         z_next, z_next, return_loss=True)
            
            lr = self.scale / torch.linalg.matrix_norm((- dz_ll + dz_reg).reshape(J, -1))
            total_time += lr
            
            z_next -= dz_ll * lr
            z_next += dz_reg * lr

            if self.mode == 'correction':
                eps = torch.randn(J, J, device=z_next.device, dtype=z_next.dtype)
                noise = eps @ z_diff / np.sqrt(J) * torch.sqrt(2 * lr)
            elif self.mode == 'diag':
                eps = torch.randn_like(z_next).reshape(J, -1)
                cov_sqrt = torch.sqrt(cov_diag)
                noise = (eps * cov_sqrt) * torch.sqrt(2 * lr)
            elif self.mode == 'original':
                eps = torch.randn_like(z_next).reshape(J, -1)
                cov_sqrt = torch.linalg.cholesky(cov + 1e-3 * torch.eye(len(cov), device=z_next.device))
                noise = (eps @ cov_sqrt) * torch.sqrt(2 * lr)
            else:
                raise ValueError(f"only 'correction', 'diag' and 'original' modes are expected, but got {self.mode}")

            z_next += noise.reshape(J, *spatial)
            if wandb.run is not None:
                wandb.log({'data_misfit': loss.item(), 'll_step_lr': lr})
        print(f'time horizon: {total_time}')
        return z_next
            
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