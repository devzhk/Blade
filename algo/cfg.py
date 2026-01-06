import torch
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler

class CFGSampler(Algo):
    def __init__(self, net, forward_op, diffusion_scheduler_config={}, sde=False, solver='euler_conditional', temp=1.0, scale=10.):
        super(CFGSampler,self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.sde = sde
        self.solver = solver
        self.temp = temp
        self.scale = scale

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=True):
        
        observation = torch.cat([observation for _ in range(num_samples)], dim=0)
                
        device = self.forward_op.device
        diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config)
        xt = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * diffusion_scheduler.sigma_max
        sampler = DiffusionSampler(diffusion_scheduler, solver=self.solver)
        xt = sampler.sample(self.net, xt, condition=observation / self.scale, SDE=self.sde, verbose=False, temp=self.temp)
        return xt