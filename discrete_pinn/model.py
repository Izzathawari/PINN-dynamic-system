import torch
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PINN ():

    def __init__(self, N_INPUT, N_HIDDEN, N_OUTPUT):
        super().__init__()
        # Use Tanh or GELU activations for smooth second derivatives!
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN), nn.Tanh(),
            nn.Linear(N_HIDDEN, N_HIDDEN), nn.Tanh(),
            nn.Linear(N_HIDDEN, N_OUTPUT)
        )
