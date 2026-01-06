from tqdm import tqdm
import torch
import numpy as np
from utils.scheduler import Scheduler

class DiffusionSampler:
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, condition=None, SDE=False, verbose=False, temp=None):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, verbose)
        elif self.solver == 'euler_conditional':
            return self._euler_conditional(model, condition, x_start, SDE, verbose)
        elif self.solver == 'euler_cfg':
            return self._euler_cfg(model, condition, x_start, temp, SDE, verbose)
        else:
            raise NotImplementedError

    def score(self, model, x, sigma):
        """
            Computes the score function for the given model.

            Parameters:
                model (DiffusionModel): Diffusion model.
                x (torch.Tensor): Input tensor.
                sigma (float): Sigma value.

            Returns:
                torch.Tensor: The computed score.
        """
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (d - x) / sigma**2
    
    def score_conditional(self, model, x, condition, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma, condition)
        return (d - x) / sigma**2
    
    def _euler(self, model, x_start, SDE=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step]
            score = self.score(model, x / self.scheduler.scaling_steps[step], sigma) / self.scheduler.scaling_steps[step]
            if SDE:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x * scaling_factor + factor * score * 0.5 
        return x
    
    def _euler_conditional(self, model, condition, x_start, SDE=False, verbose=False):
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step]
            score = self.score_conditional(model, x / self.scheduler.scaling_steps[step], condition, sigma) / self.scheduler.scaling_steps[step]
            if SDE:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x * scaling_factor + factor * score * 0.5 
        return x
    
    def _euler_cfg(self, model, condition, x_start, temp=1.0, SDE=False, verbose=False):
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step]
            score_cond = self.score_conditional(model, x / self.scheduler.scaling_steps[step], condition, sigma) / self.scheduler.scaling_steps[step]
            score_uncond = self.score_conditional(model, x / self.scheduler.scaling_steps[step], torch.zeros_like(condition), sigma) / self.scheduler.scaling_steps[step]
            score = (1. - temp) * score_uncond + temp * score_cond
            if SDE:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x * scaling_factor + factor * score * 0.5 
        return x

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start
    
