import torch
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model (model, t_data,x_data, t_physics, num_epochs, epoch_interval, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    mu_history = []

    LAMBDA_1 = 1.0
    LAMBDA_2 = 1.0

    for epoch in tqdm(range (num_epochs), desc="Training VDP Inverse Problem..."):

        optimizer.zero_grad()
        loss_data, loss_physics = model.compute_loss(t_data, x_data, t_physics)

        total_loss = LAMBDA_1 * loss_data + LAMBDA_2 * loss_physics
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

        if (epoch + 1) % epoch_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.4f}, mu: {model.mu.item():.4f}")
        
    return loss_history, mu_history

def predict_model (model,t_test, t_true, x_true,v_true, t_train_data, x_train_data, mu_history,INITIAL_MU, TARGET_MU):

    t_test = t_test.clone().detach().requires_grad_(True)

    x_pred_tensor = model(t_test)
    
    # Calculate predicted velocity: v = dx/dt
    v_pred_tensor = torch.autograd.grad(
        x_pred_tensor, t_test, 
        grad_outputs=torch.ones_like(x_pred_tensor), 
        create_graph=False
    )[0]
    
    x_pred = x_pred_tensor.detach().numpy()
    v_pred = v_pred_tensor.detach().numpy()
    t_test = t_test.detach().numpy()

    t_train_data = t_train_data.cpu().numpy().flatten()
    x_train_data = x_train_data.cpu().numpy().flatten()
    
    #3. Plotting the Results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left Plot: Trajectory Tracking
    ax1.plot(t_true, x_true, 'b-', alpha=0.5, label='True Trajectory (RK4)', lw=2)
    ax1.plot(t_test, x_pred, 'r--', label='PINN Prediction', lw=2)
    ax1.scatter(t_train_data, x_train_data, color='black', zorder=5, label='Sparse Training Data')
    ax1.set_title(f"Trajectory Reconstruction (Discovered $\mu$={model.mu.item():.2f})")
    ax1.set_xlabel("Time ($t$)")
    ax1.set_ylabel("Position ($x$)")
    ax1.legend()
    ax1.grid(True)
    
    # Right Plot: Parameter Convergence
    ax2.plot(mu_history, color='tab:orange', lw=2, label="Estimated $\mu$")
    ax2.axhline(y=TARGET_MU, color='red', linestyle='--', label=f"True $\mu$ ({TARGET_MU})")
    ax2.set_title("System Identification: $\mu$ Convergence")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Value of $\mu$")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(x_true, v_true, 'b-', alpha=0.5, label='True Limit Cycle (RK4)', lw=2)
    ax3.plot(x_pred, v_pred, 'r--', label='PINN Limit Cycle', lw=2)
    # Highlight the origin crosshairs
    ax3.axhline(0, color='black', linestyle='-', lw=0.8, alpha=0.5)
    ax3.axvline(0, color='black', linestyle='-', lw=0.8, alpha=0.5)
    ax3.set_title("Phase Space Portrait Comparison")
    ax3.set_xlabel("Position ($x$)")
    ax3.set_ylabel("Velocity ($v = \dot{x}$)")
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)


    print("\n================ PINN METRICS EVALUATION ================")
    print(f"True $\mu$ = {TARGET_MU}")
    print(f"Predicted $\mu$ = {model.mu.item()}")
    print(f"Predicted $\mu$ error = {(abs(model.mu.item() - TARGET_MU)/TARGET_MU)*100}") 
    print("=========================================================")   
    plt.tight_layout()
    plt.savefig('prediction_result.png')
