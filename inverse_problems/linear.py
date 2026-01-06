import torch
from .base import BaseOperator


class LinearOperator(BaseOperator):
    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.A = A.to(self.device) if isinstance(A, torch.Tensor) else torch.from_numpy(A).to(self.dtype).to(self.device)
    
    def forward(self, x):
        '''
        Args:
            x: torch.Tensor, shape (batch_size, in_dim)
        Returns:
            y: torch.Tensor, shape (batch_size, out_dim)
        '''
        return x @ self.A.T