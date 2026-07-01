import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Great for plotting clean heatmaps

# Re-use your existing building blocks!
from vdp_func import VDP
from pinn import PINNInverse
from train import train_model

def run_grid_search():
    # 1. Define your optimization parameters grid
    learning_rates = [1e-2, 1e-3, 1e-4]
    epoch_options = [2000, 4000, 6000]
    
    # Create an empty 2D grid to store final estimation errors for the heatmap
    # Shape will be (len(epochs), len(learning_rates))
    error_matrix = np.zeros((len(epoch_options), len(learning_rates)))

    # 2. Generate your baseline data once (True mu = 1.8)
    TRUE_MU = 1.8
    vdp = VDP(mu=TRUE_MU)
    t_timeline = np.linspace(0, 30, 1000)
    true_data = vdp.generate_trajectory([0.0, -1e-3], t_timeline)
    x_true = true_data[:, 0]

    # Sample sparse points for training data
    sparse_indices = np.linspace(0, len(t_timeline) - 1, 100).astype(int)
    t_data = torch.tensor(t_timeline[sparse_indices], dtype=torch.float32).view(-1, 1)
    x_data = torch.tensor(x_true[sparse_indices], dtype=torch.float32).view(-1, 1)
    t_physics = torch.linspace(0, 30, 400, dtype=torch.float32).view(-1, 1)

    print("🚀 Starting Grid Search Parameter Tuning...\n")

    # 3. Nest loops to try every single combination
    for i, num_epochs in enumerate(epoch_options):
        for j, lr in enumerate(learning_rates):
            print(f"Testing: Epochs={num_epochs} | Learning Rate={lr}")
            
            # Initialize a fresh model with a bad guess (mu=1.0) for every trial
            model = PINNInverse(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=64, mu_init=1.0)
            
            # Run training using the parameters for this specific cell
            train_model(
                model=model,
                t_data=t_data,
                x_data=x_data,
                t_physics=t_physics,
                num_epochs=num_epochs,
                epoch_interval=num_epochs, # Only print at the very end
                learning_rate=lr
            )
            
            # Calculate the final parameter estimation error
            final_mu_discovered = model.mu.item()
            absolute_error = abs(final_mu_discovered - TRUE_MU)
            
            # Save the result into our tracking matrix
            error_matrix[i, j] = absolute_error
            print(f"↳ Finished! Error: {absolute_error:.4f}\n")

    # 4. Generate the Heatmap Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        error_matrix, 
        annot=True, # Display raw error numbers inside the boxes
        fmt=".4f", 
        xticklabels=learning_rates, 
        yticklabels=epoch_options, 
        cmap="YlGnBu_r" # Reverse color map so dark blue = low error (excellent)
    )
    plt.title("PINN Hyperparameter Tuning: Absolute Error in Discovered $\mu$")
    plt.xlabel("Learning Rate")
    plt.ylabel("Number of Epochs")
    plt.tight_layout()
    plt.savefig("pinn_parameter_heatmap.png")

if __name__ == "__main__":
    run_grid_search()