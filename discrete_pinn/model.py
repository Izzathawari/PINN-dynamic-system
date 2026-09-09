import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PINN (nn.Module):

    def __init__(self, N_INPUT, N_HIDDEN, N_OUTPUT,R_INIT):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN), nn.Tanh(),
            nn.Linear(N_HIDDEN, N_HIDDEN), nn.Tanh(),
            nn.Linear(N_HIDDEN, N_OUTPUT)
        )

        # Learnable physical parameter (e.g., initialized at 2.0; true value is 3.8)
        self.r = nn.Parameter(torch.tensor([R_INIT], dtype=torch.float32))

    def forward(self, x):
            return self.net(x)



class LossCalc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_current_true: torch.Tensor, x_next_pred: torch.Tensor, x_next_true: torch.Tensor, r_param: torch.Tensor):
        
        # 1. Supervised Data Loss: Fit observed transitions
        data_loss = self.mse(x_next_pred, x_next_true)
        
        # 2. Discrete Physics Residual: Algebraic recurrence relation
        # Physcis loss takes x_next_pred and x_current_pred, it should uses prediction data entirely
        # In loss calculation, 

        physics_residual = x_next_pred - r_param * x_current_true* (1.0 - x_current_true)
        physics_loss = torch.mean(physics_residual ** 2)
        
        total_loss = data_loss +  physics_loss
        

        return total_loss

