import torch
from tqdm import tqdm
from .base import Algo

import wandb

from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class LocalizedEKS_DM(Algo):
    '''
    Localized Ensemble Kalman sampler with diffusion prior, adapted from https://arxiv.org/abs/1911.10832
    mathematical formulation is given in Eq. (3.73) in the paper with the prior term modified to use the diffusion prior.
    '''
    def __init__(self, 
             net,
             forward_op,
             guidance_scale,
             num_steps,
             gamma=None,
             localization_scale=1.0,
             batch_size=64,
             ensemble_size=1024,
             clean_init=False,
             mode='correction',         # correction, diag, original
             diffusion_scheduler_config={},
             **kwargs):
        super(LocalizedEKS_DM, self).__init__(net, forward_op, **kwargs)
        self.scale = guidance_scale
        self.N = num_steps

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.mode = mode
        self.ensemble_size = ensemble_size
        self.gamma = gamma if gamma is not None else forward_op.sigma_noise
        self.localization_scale = localization_scale
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
        sampler = DiffusionSampler(self.diffusion_scheduler)

        pbar = range(self.num_steps)
        for _ in pbar:
            # get localization weights and localised mean (without storing x_dev to save memory)
            x_flatten = x_next.reshape(self.ensemble_size, -1)
            localised_mean, localization_weights = self.get_localization_weights(x_flatten)
            
            # calculate prior term
            prior_term = self.get_prior_term(x_next, sampler, x_flatten, localised_mean, localization_weights, device=device)

            correction_term = self.get_correction_term(x_flatten, localised_mean, localization_weights)

            neg_likelihood_term, loss = self.ek_update(self.forward_op, observation, std_y, x_next, x_flatten, localised_mean, localization_weights, return_loss=True)

            drift = - neg_likelihood_term + prior_term + correction_term

            lr = self.scale / torch.linalg.matrix_norm(drift.reshape(self.ensemble_size, -1))
            total_time += lr

            x_next += drift.reshape(self.ensemble_size, *self.net.shape) * lr
            if self.mode == 'correction':
                noise = self.get_diffusion_noise_term(x_flatten, localised_mean, localization_weights, device) * torch.sqrt(2 * lr)
            else:
                raise ValueError(f"only 'correction' mode is expected, but got {self.mode}")
            x_next += noise.reshape(self.ensemble_size, *self.net.shape)
            if wandb.run is not None:
                wandb.log({'data_misfit': loss.item(), 'step size': lr})
        print(f'time horizon: {total_time}')
        return x_next

    def get_localization_weights(self, x_flatten):
        # x_flatten: (B, D_x)
        pairwise_dist = torch.cdist(x_flatten, x_flatten, p=2) ** 2 # (B, B)
        weights = torch.softmax(-pairwise_dist / (2 * self.localization_scale), dim=1)
        localised_mean = weights @ x_flatten # (B, D_x)
        return localised_mean, weights

    def get_diffusion_noise_term(self, x_flatten, localised_mean, localization_weights, device):
        # Compute noise without materializing full (B, B, D_x) tensor
        # x_flatten: (B, D_x), localised_mean: (B, D_x)
        eps = torch.randn(self.ensemble_size, self.ensemble_size, device=device, dtype=self.dtype)
        sqrt_weights = torch.sqrt(localization_weights)  # (B, B)
        
        # Compute: noise_term[i,k] = sum_j sqrt(w[i,j]) * (x[j,k] - localised_mean[i,k]) * eps[i,j]
        # = sum_j sqrt(w[i,j]) * eps[i,j] * x[j,k] - sum_j sqrt(w[i,j]) * eps[i,j] * localised_mean[i,k]
        weighted_eps = sqrt_weights * eps  # (B, B)
        noise_term = weighted_eps @ x_flatten - (weighted_eps.sum(dim=1, keepdim=True) * localised_mean)  # (B, D_x)
        
        return noise_term

    def get_prior_term(self, x_next, 
                       sampler, 
                       x_flatten,  # shape: (B, D_x)
                       localised_mean,  # shape: (B, D_x)
                       localization_weights,  # shape: (B, B)
                       device):
        # compute score for each particle
        sigma_min = self.diffusion_scheduler.sigma_min
        scale = self.diffusion_scheduler.scaling_steps[-1]
        scores = torch.empty(self.ensemble_size, *self.net.shape, device=device, dtype=self.dtype)
        for b in range(self.ensemble_size // self.batch_size):
            start = b * self.batch_size
            end = (b + 1) * self.batch_size
            x_batch = x_next[start:end]
            scores[start:end] = sampler.score(self.net, x_batch / scale, sigma_min) / scale
        
        scores_flatten = scores.reshape(self.ensemble_size, -1)  # (B, D_x)
        
        # Compute prior term without materializing (B, B, D_x) tensor
        # score_proj[i,j] = (x[j] - localised_mean[i]) · scores[i]
        # = x[j] · scores[i] - localised_mean[i] · scores[i]
        score_proj = scores_flatten @ x_flatten.T - (localised_mean * scores_flatten).sum(dim=1, keepdim=True)  # (B, B)
        
        # prior_term[i] = sum_j w[i,j] * (x[j] - localised_mean[i]) * score_proj[i,j]
        # = sum_j w[i,j] * score_proj[i,j] * x[j] - localised_mean[i] * sum_j w[i,j] * score_proj[i,j]
        weighted_score_proj = localization_weights * score_proj  # (B, B)
        prior_term = weighted_score_proj @ x_flatten - localised_mean * weighted_score_proj.sum(dim=1, keepdim=True)  # (B, D_x)
        
        return prior_term

    def get_correction_term(self, 
                            x_flatten,  # shape: (B, D_x)
                            localised_mean,  # shape: (B, D_x)
                            localization_weights):  # shape: (B, B)
        D_x = x_flatten.shape[-1]
        
        # Compute correction term without materializing large tensors
        # Process in chunks to save memory
        chunk_size = 128  # Process 128 particles at a time
        
        # Diagonal term: w[i,i] * (D_x + 1) * (x[i] - localised_mean[i])
        diag_weights = torch.diagonal(localization_weights)  # (B,)
        correction_term = diag_weights.unsqueeze(1) * (D_x + 1) * (x_flatten - localised_mean)  # (B, D_x)
        
        # Compute remaining terms in chunks to avoid large intermediates
        for chunk_start in range(0, self.ensemble_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.ensemble_size)
            chunk_indices = slice(chunk_start, chunk_end)
            
            # Compute dw for this chunk: dw[i,j,k] = w[i,j] * (x[j,k] - localised_mean[i,k]) / localization_scale
            # Shape: (chunk_size, B, D_x)
            x_dev_chunk = x_flatten.unsqueeze(0) - localised_mean[chunk_indices].unsqueeze(1)  # (chunk_size, B, D_x)
            dw_chunk = localization_weights[chunk_indices].unsqueeze(2) * x_dev_chunk / self.localization_scale  # (chunk_size, B, D_x)
            
            # Rewrite einsums to avoid large intermediates
            # term2[i,k] = sum_j sum_l x[j,k] * x[j,l] * dw[i,j,l]
            #            = sum_j x[j,k] * (sum_l x[j,l] * dw[i,j,l])
            #            = sum_j x[j,k] * (x[j] · dw[i,j])
            x_dot_dw = (x_flatten.unsqueeze(0) * dw_chunk).sum(dim=2)  # (chunk_size, B)
            term2_chunk = x_dot_dw @ x_flatten  # (chunk_size, D_x)
            
            # term3[i,k] = sum_j sum_l localised_mean[i,k] * x[j,l] * dw[i,j,l]
            #            = localised_mean[i,k] * sum_j sum_l x[j,l] * dw[i,j,l]
            #            = localised_mean[i,k] * sum_j (x[j] · dw[i,j])
            sum_x_dot_dw = x_dot_dw.sum(dim=1, keepdim=True)  # (chunk_size, 1)
            term3_chunk = localised_mean[chunk_indices] * sum_x_dot_dw  # (chunk_size, D_x)
            
            # term4[i,k] = sum_j sum_l x[j,k] * localised_mean[i,l] * dw[i,j,l]
            #            = sum_j x[j,k] * (localised_mean[i] · dw[i,j])
            lm_dot_dw = (localised_mean[chunk_indices].unsqueeze(1) * dw_chunk).sum(dim=2)  # (chunk_size, B)
            term4_chunk = lm_dot_dw @ x_flatten  # (chunk_size, D_x)
            
            correction_term[chunk_indices] += term2_chunk - term3_chunk - term4_chunk
        
        return correction_term

        
    @torch.no_grad()
    def ek_update(self, forward_operator, y, std_y, 
                  x_clean, 
                  x_flatten,  # shape: (B, D_x)
                  localised_mean,  # shape: (B, D_x)
                  localization_weights,  # shape:(B, B)
                  return_loss=False):
        preds = forward_operator.forward(x_clean)
        pred_flatten = preds.reshape(self.ensemble_size, -1)  # (B, D_y)
        localised_preds = localization_weights @ pred_flatten  # (B, D_y)
        
        pred_err = (preds - y).reshape(self.ensemble_size, -1)  # (B, D_y)
        
        # Compute update without materializing (B, B, D_x) tensor
        # pred_term[i,j] = pred_err[i] · (pred_flatten[j] - localised_preds[i])
        pred_term = pred_err @ pred_flatten.T - (pred_err * localised_preds).sum(dim=1, keepdim=True)  # (B, B)
        
        # x_updates[i] = sum_j w[i,j] * (x[j] - localised_mean[i]) * pred_term[i,j] / (std_y^2)
        weighted_pred_term = localization_weights * pred_term / (std_y ** 2)  # (B, B)
        x_updates = weighted_pred_term @ x_flatten - localised_mean * weighted_pred_term.sum(dim=1, keepdim=True)  # (B, D_x)
        
        if return_loss:
            loss = torch.linalg.norm(pred_err, dim=1).mean()
            return x_updates, loss
        else:
            return x_updates