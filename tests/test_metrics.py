import os
import torch
from eval import NavierStokes2d


class TestNavierStokesEval:
    evaluator = NavierStokes2d(forward_op=None)
    def test_call(self):
        N, M, D = 128, 256, 1024
        pred = torch.randn(N, M, D)
        target = torch.randn(N, D)

        for i in range(N):
            pred_i = pred[i:i+1]
            target_i = target[i:i+1]
            metric_dict = self.evaluator(pred_i, target_i)
            print(f"Metric dict for sample {i}: {metric_dict}")
        
        metric_state = self.evaluator.compute()
        print(f"Metric state: {metric_state}")
        
        
