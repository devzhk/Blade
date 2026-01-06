import torch
from .base import BaseOperator


class NonlinearOperator(BaseOperator):
    """
    Nonlinear forward operator for 2D toy problems.
    Implements various nonlinear transformations from R^2 -> R^1.
    """
    
    def __init__(self, operator_type='exp_trig', **kwargs):
        """
        Args:
            operator_type: Type of nonlinear transformation
                - 'quadratic': x1^2 + x2^2
                - 'product': x1 * x2
                - 'mixed': sin(x1) + cos(x2)
                - 'exp_trig': exp(x1/4) * cos(x2) + sin(x2)
        """
        super().__init__(**kwargs)
        self.operator_type = operator_type
        
    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch_size, 2) - input in R^2
        Returns:
            y: torch.Tensor, shape (batch_size, 1) - output in R^1
        """
        if x.shape[-1] != 2:
            raise ValueError(f"Input must be 2D, got shape {x.shape}")
            
        x1, x2 = x[..., 0], x[..., 1]
        
        if self.operator_type == 'quadratic':
            # x1^2 + x2^2
            y = (x1-8.0) ** 2 + x2 ** 2
            
        elif self.operator_type == 'product':
            # x1 * x2
            y = x1 * x2
            
        elif self.operator_type == 'mixed':
            # sin(x1) + cos(x2)
            y = torch.log((x1 - x2 ** 2) ** 2 + (8.0-x2) ** 2 * 0.1) 
            
        elif self.operator_type == 'mixed2':
            # exp(x1/4) * cos(x2) + sin(x2)
            y = torch.log(10 * (x1 - x2 ** 2) ** 2 + (8.0-x2) ** 2 * 0.3)
            
        else:
            raise ValueError(f"Unknown operator_type: {self.operator_type}")
            
        return y.unsqueeze(-1)