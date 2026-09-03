import torch
import torch.nn as nn

class LossCalc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_current_init: torch.Tensor, x_next_pred: torch.Tensor, x_next_true: torch.Tensor, r_param: torch.Tensor):
        
        # 1. Supervised Data Loss: Fit observed transitions
        data_loss = self.mse(x_next_pred, x_next_true)
        
        # 2. Discrete Physics Residual: Algebraic recurrence relation
        # Physcis loss takes x_next_pred and x_current_pred, it should uses prediction data entirely

        physics_residual = x_next_pred - r_param * x_current_init * (1.0 - x_current_init)
        physics_loss = torch.mean(physics_residual ** 2)
        
        total_loss = data_loss +  physics_loss
        

        return total_loss