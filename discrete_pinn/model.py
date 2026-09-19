import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PINN (nn.Module):
    """
    Parameter
    ---------
    N_INPUT : Number of input neuron
    N_HIDDEN : Number of neuron in each layer
    N_OUTPUT : Number of output neuron

    Output
    ---------
    Scalar value

    """

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_input,n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_output)
        )

        
    def forward(self, x):
            return self.net(x)

    def get_physics_loss(self, current_state, next_state_pred):
        """Forces child classes to define their own physics equations."""
        raise NotImplementedError("Subclasses must implement this method!")


class LogisticPINN (PINN):
     def __init__(self, n_hidden, r_init=2.0):
        # 1 Input (x_n), 1 Output (x_n+1)
        super().__init__(n_input=1, n_hidden=n_hidden, n_output=1)
        self.r = nn.Parameter(torch.tensor([r_init], dtype=torch.float32))
        
     def get_physics_residual(self, current_state, next_state_pred):
        # x_{n+1} - r * x_n * (1 - x_n)
        return next_state_pred - self.r * current_state * (1.0 - current_state)


class HenonPINN(PINN):
    def __init__(self, n_hidden, a_init: float, b_init: float):
        # 2 Inputs (x_n, y_n), 2 Outputs (x_n+1, y_n+1)
        super().__init__(n_input=2, n_hidden=n_hidden, n_output=2)
        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init], dtype=torch.float32))
        
    def get_physics_residual(self, current_state, next_state_pred):
        # Slice the 2D state into x and y components
        x_n = current_state[:, 0:1]
        y_n = current_state[:, 1:2]
        
        x_next_pred = next_state_pred[:, 0:1]
        y_next_pred = next_state_pred[:, 1:2]
        
        # Eq 1: x_{n+1} = 1 - a * x_n^2 + y_n
        eq1_res = x_next_pred - (1.0 - self.a * (x_n ** 2) + y_n)
        
        # Eq 2: y_{n+1} = b * x_n
        eq2_res = y_next_pred - (self.b * x_n)
        
        # Return combined residuals
        return torch.cat([eq1_res, eq2_res], dim=1)

    
class LossCalc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, model:PINN,  x_next_pred: torch.Tensor, x_next_true: torch.Tensor,x_current = True, calc_physics=True):
            
        # 1. Supervised Data Loss: Fit observed transitions
        data_loss = self.mse(x_next_pred, x_next_true)
        
        # 2. Discrete Physics Residual: Algebraic recurrence relation
        # Physcis loss takes x_next_pred and x_current_pred, it should uses prediction data entirely
        if calc_physics:
            # The loss function doesn't need to know the formula, 
            # it just asks the model to compute its own residual!
            physics_residual = model.get_physics_residual(x_current, x_next_pred)
            physics_loss = torch.mean(physics_residual ** 2)
            
            return data_loss + physics_loss

        

        return data_loss

