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

