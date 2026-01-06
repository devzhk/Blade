import torch
import torch.nn as nn
import numpy as np
import scipy


def construct_grid(num_modes=4, scale=16.0):
    means = []
    num_modes_per_dim = int(np.sqrt(num_modes))
    for i in range(num_modes_per_dim):
        for j in range(num_modes_per_dim):
            means.append([i * scale, j * scale])
    return np.array(means)


def construct_n_star(num_modes, scale=1.0):
    '''
    2d N roots
    '''
    thetas = np.linspace(0.25 * np.pi, 2.25 * np.pi, num_modes, endpoint=False)
    means = np.array([np.cos(thetas), np.sin(thetas)]).T * scale
    return means


def gaussian_logprob(x, mean, cov):
    '''
    Compute the Gaussian pdf for each sample in x and each mode in the Gaussian mixture
    Args:
        x: input, (N, d)
        mean: mean, (K, d)
        cov: cov, (K,)
    Returns:
        logprob without normalizing constant. (N, K)
    '''
    x = x.reshape(x.shape[0], 1, -1)  # (N, 1, d)
    mean = mean.reshape(1, mean.shape[0], -1)  # (1, K, d)
    var = (cov ** 2).reshape(1, -1, 1)  # (1, K, 1)
    logprob = - 0.5 * torch.sum((x - mean) ** 2 / var, dim=-1)  # (N, K)
    return logprob


class GScore(nn.Module):
    def __init__(self, mean, cov):
        super().__init__()
        '''
        Score model for Gaussian
        Args:
            - mean: (d,)
            - cov: a scalar or vector. If a scalar, the same cov is used for all dimensions
        '''
        self.mean = mean
        self.cov = torch.tensor([cov] * len(mean)).to(mean.device) if isinstance(cov, float) else cov
    
    def forward(self, t, x):
        '''
        Args: 
            - x: (N, d)
            - t: time, a scalar
        Return:
          - score of p_t(x): (N, d)
        '''
        cov_arr = torch.sqrt(self.cov + t ** 2)
        diff = self.mean - x
        score = diff / cov_arr ** 2
        return score


class GMScore(nn.Module):
    def __init__(self, mean, cov, prior=None):
        '''
        Score model for a Gaussian Mixture with K modes
        Args:
            - mean: (K, d)
            - cov: a list of float or a scalar, (K,) or scalar
            - prior: prior probability of each mode, if None, uniform prior is used
        '''
        super().__init__()
        self.device = mean.device if hasattr(mean, 'device') else 'cpu'
        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, device=self.device)
        self.shape = self.mean[0].shape
        self.cov = torch.tensor([cov] * len(mean), device=self.device) if isinstance(cov, float) else cov
        self.num_modes = len(mean)
        self.prior = prior if prior is not None else torch.ones(self.num_modes, device=self.device) / self.num_modes
    
    def forward(self, x, t):
        '''
        Args: 
            - x: (N, d)
            - t: time, a scalar
        Return:
          - score of p_t(x): (N, d)
        '''
        N, d = x.shape

        cov_arr = torch.sqrt(self.cov + t ** 2)          # (K,)

        # Compute prior_i * normalizing constant
        log_mix = torch.log(self.prior) - d * torch.log(cov_arr)  # (K,)
        
        log_gaussian = gaussian_logprob(x, self.mean, cov_arr) # (N, K)
        
        log_prob = log_gaussian + log_mix  # (N, K)

        les = torch.logsumexp(log_prob, dim=-1, keepdim=True) # (N, 1)
        weights = torch.exp(log_prob - les) # (N, K)

        diff = self.mean.reshape(1, self.num_modes, -1) - x.reshape(x.shape[0], 1, -1)  # (N, K, d)
        diff = diff / cov_arr.reshape(1, self.num_modes, 1) ** 2 # (N, K, d)
        
        # Compute the score
        score = torch.sum(weights.reshape(N, self.num_modes, 1) * diff, dim=1) # (N, d)
        return score


class EDMPrecond(nn.Module):
    def __init__(self, score_model):
        super().__init__()
        self.shape = score_model.shape
        self.score_model = score_model
    
    def forward(self, x, t):
        score = self.score_model(x, t)
        return x + t ** 2 * score
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def construct_gmm_dist(means, covs, probs=None, logits=None, dtype=torch.float32):
    if probs is not None:
        weights = torch.tensor(probs, dtype=dtype) if not isinstance(probs, torch.Tensor) else probs
        mix = torch.distributions.Categorical(probs=weights)
    elif logits is not None:
        logits = torch.tensor(logits, dtype=dtype) if not isinstance(logits, torch.Tensor) else logits
        mix = torch.distributions.Categorical(logits=logits)
    else:
        weights = torch.ones(len(means), dtype=dtype) / len(means)
        mix = torch.distributions.Categorical(probs=weights)
    
    means = torch.tensor(means, dtype=dtype) if not isinstance(means, torch.Tensor) else means.to(dtype)
    device = means.device
    if isinstance(covs, float):
        covs = torch.eye(means.shape[1], dtype=dtype, device=device).repeat(means.shape[0], 1, 1) * covs
    elif isinstance(covs, torch.Tensor):
        covs = covs.to(device).to(dtype)
    elif isinstance(covs, np.ndarray):
        covs = torch.from_numpy(covs).to(device).to(dtype)
    else:
        raise ValueError(f"covs must be a float, torch.Tensor, or numpy array")
    comp = torch.distributions.MultivariateNormal(means, covs)
    return torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)


class GaussianMixture(object):
    def __init__(self, mean, cov, prior=None):
        '''
        Args:
            num_samples: number of samples to generate
            mean: a list of means of the Gaussian mixture
            cov: a list of cov, or a single cov for all modes, or a scalar
            prior: prior probability of each mode, if None, uniform prior is used
        '''
        super().__init__()
        self.device = mean.device if hasattr(mean, 'device') else 'cpu'

        self.mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, device=self.device)
        self.shape = self.mean[0].shape
        if isinstance(cov, float):
            self.covs = [cov * torch.eye(mean.shape[1], device=self.device)] * len(mean)
            self.stds = [np.sqrt(cov) * torch.eye(mean.shape[1], device=self.device)] * len(mean)
        elif len(cov.shape) == 2:
            self.covs = [cov] * len(mean)
            self.stds = [np.sqrt(cov) * torch.eye(mean.shape[1], device=self.device)] * len(mean)
        else:
            raise ValueError(f"cov must be a float or a 2D array, but got {cov.shape}")
        self.num_modes = len(mean)
        self.prior = prior if prior is not None else np.ones(len(mean)) / len(mean)

    def generate(self, num_samples):
        '''
        Return (num_samples, dim)
        '''
        # generate samples from a Gaussian mixture given the prior
        # sample latent code
        latents = np.random.choice(self.num_modes, size=num_samples, p=self.prior)
        samples = []
        for i in range(self.num_modes):
            num_samples_per_mode = np.sum(latents == i)
            print(f'num_samples_per_mode: {num_samples_per_mode}')
            sample_mean = self.mean[i]
            sample_cov = self.stds[i]
            rand = torch.randn((num_samples_per_mode, sample_mean.shape[0]), device=self.device) @ sample_cov
            sample_per_mode = sample_mean + rand
            samples.append(sample_per_mode)

        samples = torch.cat(samples, dim=0)
        return samples

    def logp(self, samples):
        # Args: 
        #   - samples: (batch_size, d)
        # Return: 
        #   - log_prob: (batch_size,), log probability of each sample
        
        batch_size, d = samples.shape
        
        # Convert to tensor if needed
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, device=self.device)
        
        log_probs_per_mode = []
        
        for i in range(self.num_modes):
            mean_i = self.mean[i]  # (d,)
            cov_i = self.covs[i]   # (d, d) - assuming diagonal for simplicity
            
            # For diagonal covariance, extract diagonal elements
            if len(cov_i.shape) == 2:
                var_i = torch.diag(cov_i)  # (d,)
            else:
                var_i = cov_i  # already diagonal
            
            # Compute log probability for mode i
            diff = samples - mean_i.unsqueeze(0)  # (batch_size, d)
            log_prob_i = -0.5 * torch.sum(diff ** 2 / var_i, dim=1)  # (batch_size,)
            log_prob_i = log_prob_i - 0.5 * torch.sum(torch.log(2 * np.pi * var_i))  # add normalizing constant
            log_prob_i = log_prob_i + np.log(self.prior[i])  # add prior weight
            
            log_probs_per_mode.append(log_prob_i)
        
        # Stack and compute logsumexp
        log_probs_per_mode = torch.stack(log_probs_per_mode, dim=1)  # (batch_size, num_modes)
        log_prob = torch.logsumexp(log_probs_per_mode, dim=1)  # (batch_size,)
        
        return log_prob
    

# -------------------- Derive the ground truth posterior for the linear Gaussian mixture setting --------------------
# Consider the following linear inverse problem:
# y = Hx + e
# where e ~ N(0, noise_std^2 I)
# The prior on x is a Gaussian mixture model
# p(x) = sum_i w_i N(x; means[i], covs[i])

# The posterior is given by
# p(x | y) = sum_i v_i N(x; posterior_means[i], posterior_covs[i])
# where
# posterior_covar = inv(H^T * noise_std^-2 * H + covs[i]^-1)
# posterior_mean = posterior_covar @ (H^T * y * noise_std^-2 + covs[i]^-1 * means[i])
# v_i = w_i * N(y; H * means[i], H^T @ covs[i] @ H + noise_std^2 I) / sum_j w_j * N(y; H * means[j], H^T @ covs[j] @ H + noise_std^2 I)


# Returns:
#     - posterior_means: list of posterior means, K x n
#     - posterior_covs: list of posterior covariances, K x n x n
#     - posterior_weights: array of posterior weights of each mode, K


def true_posterior(y,           # observation,  (m,)
                   H,           # linear operator, m x n
                   prior,       # prior weight of each mode, K
                   means, covs, # prior means and covariances, K x n, K x n x n
                   noise_std):  # noise std, scalar
    '''

    '''
    m, n = H.shape
    K = means.shape[0]
    posterior_means = []
    posterior_covs = []
    log_weights = []
    for i in range(K):
        assert covs[i].shape == (n, n)
        cov_inv = H.T @ H / noise_std ** 2 + np.linalg.inv(covs[i]) # (n, n)
        posterior_cov = np.linalg.inv(cov_inv)
        posterior_mean = posterior_cov @ (H.T @ y / noise_std ** 2 + np.linalg.inv(covs[i]) @ means[i]) # (n,)
        posterior_means.append(posterior_mean)
        posterior_covs.append(posterior_cov)
        residual = y - H @ means[i] # (m,)

        weight_cov = H @ covs[i] @ H.T + noise_std ** 2 * np.eye(m)
        log_weight = - 0.5 * residual[None, :] @ np.linalg.inv(weight_cov) @ residual[:, None] - 0.5 * np.log(np.linalg.det(weight_cov))
        log_weights.append(log_weight.item())
    log_weights = np.array(log_weights) + np.log(prior)
    les = scipy.special.logsumexp(log_weights)
    posterior_weights = np.exp(log_weights - les)   # normalize the weights
    # construct the posterior mixture
    posterior_covs = np.stack(posterior_covs, axis=0)   # K x n x n
    posterior_means = np.stack(posterior_means, axis=0) # K x n
    print(f'posterior_weights: {posterior_weights}')
    print(f'posterior_means: {posterior_means}')
    print(f'posterior_covs: {posterior_covs}')
    posterior_dist = construct_gmm_dist(posterior_means, posterior_covs, probs=posterior_weights)
    # posterior_dist = GaussianMixture(mean=posterior_means, cov=posterior_covs, prior=posterior_weights)
    return posterior_dist, posterior_means, posterior_covs