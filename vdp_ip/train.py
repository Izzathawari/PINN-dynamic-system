import torch
import time
import numpy as np
from tqdm import tqdm

# def initial_loss_function(params, timepoints, population):
#     y0 = [population[0]]
#     t = np.linspace(timepoints[0], timepoints[-1], num=len(timepoints))
#     solution = odseint(sim, y0, t, args=(params,), atol=1e-4, rtol=1e-4)  # Relax tolerances
#     residuals = population - solution[:, 0]
#     return np.mean(residuals**2)

# def optimize_parameters(timepoints, population):
#     params0 = np.array([1.0, 10.0])  # Initial guesses for alpha and K
#     result = minimize(
#         initial_loss_function,
#         params0,
#         args=(timepoints, population),
#         method='L-BFGS-B',
#         options={'disp': True, 'gtol': 1e-4}
#     )
#     return result.x

def train_model (model, timepoints, num_epochs, learning_rate,EPOCH_INTERVAL):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    mu_history = []

    for epoch in tqdm(range (num_epochs), desc="Training VDP Inverse Problem..."):

        optimizer.zero_grad()
        loss_data, loss_physics = model.compute_loss(t_data, x_data, t_physics)

        total_loss = loss_data + loss_physics
        total_loss.backward()
        optimizer.step()

        loss_value = total_loss.item()
        loss_history.append(loss_value)
        
        # Record the parameter values
        mu_history.append(model.mu.item())

        # if (epoch + 1) % EPOCH_INTERVAL == 0:
        #     # Optimize parameters periodically
        #     alpha_init, K_init = optimize_parameters(timepoints, population)
        #     model.alpha.data = torch.tensor(alpha_init, dtype=torch.float32)
        #     model.K.data = torch.tensor(K_init, dtype=torch.float32)

        if (epoch + 1) % EPOCH_INTERVAL == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.4f}, alpha: {model.alpha.item():.4f}, K: {model.K.item():.4f}")