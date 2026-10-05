import torch
import numpy as np

def create_delay_embedding(data, delay = 1):
    """
    Given 1D tensor x_series of shape (L, 1):
    Returns:
      x_input:  shape (L-2, 2) containing [x_n, x_{n-1}]
      x_target: shape (L-2, 1) containing x_{n+1}

    
    Delay Embedding Sequence Tally Layout
    -------------------------------------
    Index (k) | Feature 0 (X_n) | Feature 1 (X_{n-1}) | Target (X_{n+1}) | Tally Check
    ----------+-----------------+---------------------+------------------+----------------------
    0         | x_1             | x_0                 | x_2              | X_{n-1}[0] == x_0
    1         | x_2             | x_1                 | x_3              | X_{n-1}[1] == X_n[0]
    
    """
    if data.ndim == 2:
        x = data[:, 0]
    else:
        x = data
    
    x_curr = x[delay:]      # X_n
    x_prev = x[:-delay]     # X_{n-1}
    x_next = x[delay+1:]   # X_{n+1} (target)
    
    inputs = np.column_stack([x_curr[:-1], x_prev[:-1]])
    targets = x_next
    return inputs, targets