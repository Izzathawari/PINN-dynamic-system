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
            nn.Linear(n_input,n_hidden), nn.SiLU(),
            nn.Linear(n_hidden,n_hidden), nn.SiLU(),
            nn.Linear(n_hidden,n_hidden), nn.SiLU(),
            nn.Linear(n_hidden,n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_output))

        
    def forward(self, x):
            return self.net(x)


    
class LossCalc(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, 
                 t: torch.Tensor,
                 state_pred: torch.Tensor,
                 state_true: torch.Tensor,
                 pend_param = None,
                 ic_loss = None):

        """
        Use data loss and physics loss
        
        Parameters
        -----------------------------
        state_pred : 2D pytorch array
            Referring to angle of bob 1, and bob 2. (theta_1 and theta_2)
    

        Returns
        ----------------------------
        total_loss 
            Array of total loss of data and physics loss 

        """
        total_loss = 0

        #===============
        # Data Loss
        data_loss = self.mse(state_pred, state_true)
        total_loss = data_loss


        #==============
        # Physics Loss
        if pend_param is not None:

            g, m1, m2, t1_ic, t2_ic, w1_ic, w2_ic, L1, L2 = pend_param
            dp_eq = DoublePendulumLagrangian(*pend_param)
            #=================
            # theta_1, theta_2 : 角度重り１と２
            # w_1, w_2 : 角速度重り１と２
            # alpha_1, alpha_2 : 各加速度重り１と２

            theta_1 = state_pred[:, 0:1]
            theta_2 = state_pred[:, 1:2]

            # 1st derivatives w.r.t. time t -> angular velocities (w_1, w_2)
            w_1 = torch.autograd.grad(
                theta_1, t, grad_outputs=torch.ones_like(theta_1), create_graph=True)[0]
            w_2 = torch.autograd.grad(
                theta_2, t, grad_outputs=torch.ones_like(theta_2), create_graph=True)[0]

            # 2nd derivatives w.r.t. time t -> angular accelerations (alpha_1, alpha_2)
            alpha_1 = torch.autograd.grad(
                w_1, t, grad_outputs=torch.ones_like(w_1), create_graph=True)[0]
            alpha_2 = torch.autograd.grad(
                w_2, t, grad_outputs=torch.ones_like(w_2), create_graph=True)[0]

            # Theoretical Euler-Lagrange accelerations from physics equations
            g1_phys, g2_phys = dp_eq.accel_torch(theta_1, theta_2, w_1, w_2)
            
            physics_loss = self.mse(alpha_1, g1_phys) + self.mse(alpha_2, g2_phys)

            if ic_loss :

                t1_ic = torch.tensor(t1_ic, dtype=state_pred.dtype, device=state_pred.device)
                t2_ic = torch.tensor(t2_ic, dtype=state_pred.dtype, device=state_pred.device)
                w1_ic = torch.tensor(w1_ic, dtype=state_pred.dtype, device=state_pred.device)
                w2_ic = torch.tensor(w2_ic, dtype=state_pred.dtype, device=state_pred.device)

                # Squared error on initial angles and initial angular velocities at index 0 (t_0)
                ic_loss = (
                    (theta_1[0, 0] - t1_ic) ** 2
                    + (theta_2[0, 0] - t2_ic) ** 2
                    + (w_1[0, 0] - w1_ic) ** 2
                    + (w_2[0, 0] - w2_ic) ** 2
                )
                total_loss = total_loss +  ic_loss +  physics_loss

            else:
                total_loss = total_loss +  physics_loss
                 

            total_loss = data_loss + physics_loss
            

            return total_loss

        
        
        return data_loss


             

