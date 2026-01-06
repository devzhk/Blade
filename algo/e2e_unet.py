import torch
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler

class E2EUnet(Algo):
    def __init__(self, net, forward_op, scale=10.):
        super(E2EUnet,self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.scale = scale

    def inference(self, observation, num_samples=1, verbose=True):
        
        input = torch.cat([observation for _ in range(num_samples)], dim=0)
        output = self.net(input / self.scale)
        return output