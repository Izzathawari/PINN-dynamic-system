import torch

def create_delay_embedding(x_series: torch.Tensor):
    """
    Given 1D tensor x_series of shape (L, 1):
    Returns:
      x_input:  shape (L-2, 2) containing [x_n, x_{n-1}]
      x_target: shape (L-2, 1) containing x_{n+1}
    """
    x_prev = x_series[0:-2]  # x_{n-1} (indices 0 to L-3)
    x_curr = x_series[1:-1]  # x_n     (indices 1 to L-2)
    x_next = x_series[2:]    # x_{n+1} (indices 2 to L-1)

    # Stack [x_n, x_{n-1}] along column dimension
    x_input = torch.cat([x_curr, x_prev], dim=1)  # shape (N, 2)
    x_target = x_next                             # shape (N, 1)

    return x_input, x_target