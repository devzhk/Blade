import torch
from tqdm import tqdm
from .base import Algo

import wandb
import numpy as np
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class FeynmanKac(Algo):
    '''
    Derivative-free variant of Feynman-Kac diffusion steering, as described in https://arxiv.org/abs/2501.06848
    General algorithm is given in Algorithm 1 in the paper. Reference code: https://github.com/zacharyhorvitz/Fk-Diffusion-Steering/blob/main/text_to_image/fkd_diffusers/fkd_class.py
    '''
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 num_particles=8,
                 potential_type='diff', 
                 temperature_inv=1.0, # 1.0 means no temperature
                 batch_size=128,
                 threshold=0.25):
        super(FeynmanKac, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.num_particles = num_particles
        self.potential_type = potential_type
        self.temperature_inv = temperature_inv # 1.0 means no temperature
        self.threshold = threshold
        self.batch_size= min(batch_size, num_particles)
        assert self.num_particles % self.batch_size == 0, 'Batch size should be divisible by number of particles.'

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=True):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        num_batches = self.num_particles // self.batch_size
        x_next = torch.randn(self.num_particles, *self.net.shape, device=device) * self.scheduler.sigma_max
        pbar = tqdm(range(num_steps))
        population_rewards = None
        for i in pbar:
            x_cur = x_next
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]

            denoised = torch.empty(self.num_particles, *self.net.shape, device=device)
            for k in range(num_batches):
                x_batch = x_cur[k*self.batch_size:(k+1)*self.batch_size]
                denoised[k*self.batch_size:(k+1)*self.batch_size] = self.net(x_batch / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            if i < int(num_steps * self.threshold):
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * torch.randn_like(x_cur)
            elif i > int(num_steps * self.threshold) and i < num_steps - 1:
                # step with resample
                rewards = - self.forward_op.loss(denoised.to(self.forward_op.dtype), observation)
                loss_scale = torch.mean(- rewards).item()
                if population_rewards is None:
                    population_rewards = rewards
                    log_weights = rewards
                elif self.potential_type == 'diff':
                    log_weights = rewards - population_rewards
                elif self.potential_type == 'add':
                    rewards = rewards + population_rewards
                    log_weights = rewards
                elif self.potential_type == 'max':
                    rewards = torch.max(rewards, population_rewards)
                    log_weights = rewards
                else:
                    raise ValueError(f'Invalid potential type: {self.potential_type}')

                log_weights_scaled = log_weights * self.temperature_inv
                log_weights_scaled = log_weights_scaled - log_weights_scaled.max()  # log-sum-exp trick for stability
                probs = torch.exp(log_weights_scaled)
                indices = torch.multinomial(probs, self.num_particles, replacement=True)
                resampled_x = x_cur[indices]
                resampled_score = score[indices]
                population_rewards = rewards[indices]
                x_next = resampled_x * scaling_factor + factor * resampled_score + np.sqrt(factor) * torch.randn_like(resampled_x)
                pbar.set_description(f'Iteration {i + 1}/{num_steps}. Data fitting loss: {np.sqrt(loss_scale)}')
                if wandb.run is not None:
                    wandb.log({'data_misfit': np.sqrt(loss_scale), 'step': i})
            else:
                x_next = denoised
        return x_next


