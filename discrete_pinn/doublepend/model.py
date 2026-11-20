import numpy as np
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dp_lagrangian import DoublePendulumLagrangian

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
            nn.Linear(n_hidden, n_output))

        
    def forward(self, x):
            return self.net(x)


    
class LossCalc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, model:PINN,
                 state_pred: torch.Tensor,
                 state_true: torch.Tensor,
                 calc_physics=True):

        """
        Use data loss and physics loss

        """
        dp_eq = DoublePendulumLagrangian()
        #=================
        # theta_1, theta_2 : 角度重り１と２
        # w_1, w_2 : 角速度重り１と２
        # alpha_1, alpha_2 : 各加速度重り１と２
        
        data_loss = self.mse(state_pred, state_true)
        
        physics_loss = 

        

        return data_loss

