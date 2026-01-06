'''
This file implements evaluator for each problem.
'''
import torch
import numpy as np
from abc import ABC, abstractmethod
import torch
import numpy as np
from piq import LPIPS, psnr, ssim
from collections import defaultdict
from training.loss import DynamicRangePSNRLoss, DynamicRangeSSIMLoss
import ot
from inverse_problems.base import BaseOperator


class Evaluator(ABC):
    def __init__(self, 
                 metric_list, 
                 forward_op=None, 
                 data_misfit=False):
        self.metric_list = metric_list
        self.forward_op = forward_op
        self.data_misfit = data_misfit
        if data_misfit:
            assert forward_op is not None, "forward_op must be provided for data misfit evaluation"
        
        self.device = forward_op.device if forward_op is not None else 'cpu'
        self.metric_state = {key: [] for key in metric_list.keys()}
        if data_misfit:
            self.metric_state['data misfit'] = []
        # each metric is a list of values

    def eval_data_misfit(self, pred, observation):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W) unnormalized
            - observation (torch.Tensor): (N, C, H, W)
        Returns:
            - data_misfit (torch.Tensor): (N,), data misfit
        '''
        data_misfit = self.forward_op.loss(pred, observation, unnormalize=False)
        return torch.sqrt(data_misfit)

    @abstractmethod
    def __call__(self, pred, target, observation=None, forward_op=None):
        ''''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
            - observation (torch.Tensor): (N, *observation.shape) or (*observation.shape)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        pass

    def compute(self):
        '''
        Returns:
            - metric_state (Dict): a dictionary of metric values
        '''
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state
    
    
def gaussian_kl_divergence(gt_mean: torch.Tensor, gt_cov: torch.Tensor, samples: torch.Tensor):
    '''
    Returns the KL divergence between two Gaussian distributions. For generated samples,
    the mean and covariance matrix are estimated from the data.
    '''
    
    gt_mean = gt_mean.to(torch.float64)
    gt_cov = gt_cov.to(torch.float64)
    
    if len(gt_cov.shape) == 3:
        gt_cov = gt_cov[0]
    
    sample_mean = torch.mean(samples, dim=0, keepdim=True)
    sample_diff = samples - sample_mean
    cov = sample_diff.T @ sample_diff / len(sample_diff)
    print(f'cov: {cov}')
    
    d = gt_mean.shape[0]
    cov_inv = torch.linalg.inv(cov)
    gt_det = torch.linalg.det(gt_cov)
    cov_det = torch.linalg.det(cov)
    
    d_kl = 0.5 * (torch.trace(cov_inv @ gt_cov) + (sample_mean - gt_mean) @ cov_inv @ (sample_mean - gt_mean).T - d + torch.log(cov_det / gt_det))
    
    return d_kl
    

def sliced_wd(samples1: torch.Tensor, samples2: torch.Tensor, n_projections: int = 500):
    wd = ot.sliced_wasserstein_distance(samples1.to(torch.float64), samples2.to(torch.float64), n_projections=n_projections)
    return wd 
    

class LinearGaussianEvaluator(Evaluator):
    def __init__(self, 
                 forward_op: BaseOperator, 
                 posterior_mean: torch.Tensor, 
                 posterior_cov: torch.Tensor, 
                 n_projections: int = 500):
        metric_list = {'d_kl': gaussian_kl_divergence, 'sliced_wd': None}
        super().__init__(metric_list, forward_op=forward_op)
        self.device = forward_op.device
        self.posterior_mean = posterior_mean
        self.posterior_cov = posterior_cov
        
    def __call__(self, pred, target, observation=None, forward_op=None):
        metric_dict = {}
        
        kl = gaussian_kl_divergence(self.posterior_mean, self.posterior_cov, pred)
        swd = sliced_wd(pred, target)
        
        metric_dict['d_kl'] = kl
        metric_dict['sliced_wd'] = swd 
        self.metric_state['d_kl'].append(kl)
        self.metric_state['sliced_wd'].append(swd)
        return metric_dict


class BlackHoleEvaluator(Evaluator):
    def __init__(self, 
                 forward_op=None):
        metric_list = {'cp_chi2': None, 'camp_chi2': None, 'psnr': None, 'blur_psnr (f=10)': None,
                       'blur_psnr (f=15)': None,
                       'blur_psnr (f=20)': None}
        super().__init__(metric_list, forward_op=forward_op)
        self.device = forward_op.device

    def __call__(self, pred, target, observation=None):
        metric_dict = {}
        pred, target, observation = pred.to(self.device), target.to(self.device), observation.to(self.device)

        # evaluation
        if pred.shape != target.shape:
            target = target.repeat(pred.shape[0], 1, 1, 1)
            observation = observation.repeat(pred.shape[0], 1, 1, 1)

        # chi-square
        chisq_cp, chisq_logcamp = self.forward_op.evaluate_chisq(pred, observation, True)

        # blurry PSNR
        blur_factors = [0, 10, 15, 20]
        blur_psnr = self.forward_op.evaluate_psnr(target, pred, blur_factors)
        blur_psnr = blur_psnr.max(dim=0)[0]

        metric_dict['cp_chi2'] = chisq_cp.min().item()
        metric_dict['camp_chi2'] = chisq_logcamp.min().item()
        metric_dict['psnr'] = blur_psnr[0].item()
        metric_dict['blur_psnr (f=10)'] = blur_psnr[1].item()
        metric_dict['blur_psnr (f=15)'] = blur_psnr[2].item()
        metric_dict['blur_psnr (f=20)'] = blur_psnr[3].item()

        self.metric_state['cp_chi2'].append(metric_dict['cp_chi2'])
        self.metric_state['camp_chi2'].append(metric_dict['camp_chi2'])
        self.metric_state['psnr'].append(metric_dict['psnr'])
        self.metric_state['blur_psnr (f=10)'].append(metric_dict['blur_psnr (f=10)'])
        self.metric_state['blur_psnr (f=15)'].append(metric_dict['blur_psnr (f=15)'])
        self.metric_state['blur_psnr (f=20)'].append(metric_dict['blur_psnr (f=20)'])
        return metric_dict
    

def relative_l2(pred: torch.Tensor, 
                target: torch.Tensor):
    ''''
    Args:
    -----
        pred: (N, M, ...)
        target: (N, ...)

    Returns:
    -------
        rel_l2: float, relative L2 error
    '''
    diff = pred - target.unsqueeze(1)

    sq_error = torch.square(diff).reshape(diff.shape[0], diff.shape[1], -1).sum(dim=-1)
    sq_ref = torch.square(target).reshape(target.shape[0], -1).sum(dim=-1, keepdim=True)

    rel_l2 = torch.sqrt(sq_error / sq_ref).mean()  # (N, M)
    return rel_l2


def crps(pred: torch.Tensor, 
         target: torch.Tensor):
    ''''
    Fast and memory-efficient implementation of 
    unbiased estimator of CPRS (Continuous Ranked Probability Score) for ensemble prediction, 
    averaged over dimensions and cases. 

    References:
    ---------
    Zamo, Michaël, and Philippe Naveau. "Estimation of the continuous ranked probability score with 
    limited information and applications to ensemble weather forecasts." Mathematical Geosciences 50.2 (2018): 209-234.

    Args:
    -------
        pred: (N, M, ...)
               N ensemble forecasts with M members each.
        target: (N, ...)
                N verifying observations.

    Returns:
    -------
         crps: float
    '''
    M= pred.shape[1]
    data_dim = len(pred.shape) - 2  # data dimensionality: D
    # basic shape checks
    if pred.shape[1] < 2:
        print("Warning: Cannot compute CRPS with only 1 ensemble member. Returning 0.0")
        return torch.tensor(0.0)

    # 1) sort along the ensemble axis (M) – O(N·D·M log M)
    xs, _ = torch.sort(pred, dim=1)

    # 2) mean absolute error term
    mae = (xs - target.unsqueeze(1)).abs().mean()

    beta0 = xs.mean()
    idx = torch.arange(M, dtype=pred.dtype, device=pred.device).view(1, M, *([1] * data_dim))  # (1, M, ...)
    beta1 = (idx * xs).mean() / (M - 1) 

    crps_score = mae + beta0 - 2.0 * beta1 
    return crps_score


def get_skill2(pred: torch.Tensor, 
               target: torch.Tensor):
    """
    Compute the squared spread and squared skill of an ensemble prediction.

    Args:
    ----
        pred: Ensemble predicted values. Shape (N, M, ...)
        target: Target values. Shape (N, ...)

    Returns:
    -------
        skill2: Squared skill of the ensemble predictions.
    """

    # compute spread^2
    M = pred.shape[1]
    if M > 1:
        s2 = torch.mean(torch.var(pred, dim=1, correction=1))

        # compute skill^2
        # error of ensemble mean
        ens_mean_error = torch.mean(pred, dim=1) - target
        skill2 = torch.mean(torch.square(ens_mean_error)) + s2 / M / (M - 1)  # skill^2
        return skill2
    else:
        print("Warning: Cannot compute spread-skill-ratio with only 1 ensemble member. Returning 100.0")
        return torch.tensor(100.0)


def get_spread2(pred: torch.Tensor, 
                target: torch.Tensor):
    """
    Compute the squared spread of an ensemble prediction.

    Args:
    ----
        pred: Ensemble predicted values. Shape (N, M, ...)
        target: Target values. Shape (N, ...)

    Returns:
    -------
        spread2: Squared spread of the ensemble predictions.
    """
    # compute spread^2
    M = pred.shape[1]
    if M > 1:
        s2 = torch.mean(torch.var(pred, dim=1, correction=1))
        spread2 = s2 * M / (M - 1)  # spread^2
        return spread2
    else:
        print("Warning: Cannot compute spread-skill-ratio with only 1 ensemble member. Returning 0.0")
        return torch.tensor(0.0)


class NavierStokes2d(Evaluator):
    def __init__(self, forward_op, 
                 ssr_weight=0.2, 
                 l2_weight=0.4, 
                 crps_weight=0.4):
        weight_sum = ssr_weight + l2_weight + crps_weight
        self.ssr_weight = ssr_weight / weight_sum
        self.l2_weight = l2_weight / weight_sum
        self.crps_weight = crps_weight / weight_sum
        print(f"ssr_weight: {self.ssr_weight}, l2_weight: {self.l2_weight}, crps_weight: {self.crps_weight}")
        metric_list = {'relative l2': relative_l2, 
                       'crps': crps, 
                       'spread2': get_spread2, 
                       'skill2': get_skill2}
        super(NavierStokes2d, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {}
        pred = pred.unsqueeze(0)
        target = target
        for metric_name, metric_func in self.metric_list.items():
            val = metric_func(pred, target).item()
            metric_dict[metric_name] = val
            self.metric_state[metric_name].append(val)
        return metric_dict

    def compute(self):
        metric_state = super().compute()
        spread_skill_ratio = np.sqrt(metric_state['spread2']) / np.sqrt(metric_state['skill2'])
        metric_state['spread-skill-ratio'] = spread_skill_ratio
        spread_skill_ratio_dist = np.abs(spread_skill_ratio - 1.0)
        metric_state['spread-skill-ratio-dist'] = spread_skill_ratio_dist
        if metric_state['relative l2'] > 0.9:
            metric_state['score'] = 5.0
        else: 
            metric_state['score'] = self.ssr_weight * spread_skill_ratio_dist \
                + self.l2_weight * metric_state['relative l2'] \
                + self.crps_weight * metric_state['crps']
        return metric_state


class Image(Evaluator):
    def __init__(self, forward_op=None):
        self.eval_batch = 32
        metric_list = {'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'lpips': LPIPS(replace_pooling=True, reduction='none')}
        super(Image, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if pred.shape != target.shape:
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name] += metric_dict[metric_name]
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name] += val
        return metric_dict

def fwi_norm(x):
    return (x - 1.5) / 3.0


class AcousticWave(Evaluator):
    def __init__(self, forward_op=None):
        metric_list = {'relative l2': relative_l2, 
                       'psnr': lambda x, y: psnr(fwi_norm(x).clip(0, 1), fwi_norm(y).clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(fwi_norm(x).clip(0, 1), fwi_norm(y).clip(0, 1), data_range=1.0, reduction='none')}
        super(AcousticWave, self).__init__(metric_list, forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {'data misfit': 0.0}
        for metric_name, metric_func in self.metric_list.items():
            if len(target.shape) == 3:
                val = metric_func(pred, target).item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
                self.metric_state[metric_name].append(val)
        
        data_misfit = self.eval_data_misfit(pred, observation).mean().item()
        metric_dict['data misfit']= data_misfit
        self.metric_state['data misfit'].append(data_misfit)
        return metric_dict
    
    
class MRI(Evaluator):
    def __init__(self, forward_op=None):
        dr_psnr_loss = DynamicRangePSNRLoss()
        dr_ssim_loss = DynamicRangeSSIMLoss()
        self.eval_batch = 32
        metric_list = {
            'psnr': lambda x, y: -dr_psnr_loss(x, y),
            'ssim': lambda x, y: 1-dr_ssim_loss(x, y)
        }
        super(MRI, self).__init__(metric_list, forward_op=forward_op)
        self.metric_state = defaultdict(list)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if len(pred) != len(target):
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name].append(metric_dict[metric_name])
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        if self.forward_op is not None and observation is not None:
            pred = pred.to(self.device)
            observation = observation.to(self.device)
            metric_dict['data misfit'] = torch.linalg.norm(self.forward_op.forward(pred) - observation).item()
            self.metric_state['data misfit'].append(metric_dict['data misfit'])
        return metric_dict



class InverseScatter(Evaluator):
    def __init__(self, forward_op=None):
        self.eval_batch = 32
        metric_list = {'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none')}
        super(InverseScatter, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            if pred.shape != target.shape:
                val = metric_func(pred, target.repeat(pred.shape[0],1,1,1)).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        return metric_dict
    
