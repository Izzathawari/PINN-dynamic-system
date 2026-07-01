from vdp_func import VDP
from train import train_model
from pinn import PINNInverse
import numpy as np
import torch
import matplotlib.pyplot as plt

def main ():

    ##-------------------Step 1----------------------------------##
    """
    Create observable data
    """
    INITIAL_MU = 1.8
    vanderpol = VDP(mu = INITIAL_MU)

    t_timedata = np.linspace(0, 30,1000)
    initial_condition = [0.5,0.0]
    print(f"-----Preparing Vander Pol Data via RK4-----")
    true_data = vanderpol.generate_trajectory(initial_condition, t_timedata)
    plot_data = vanderpol.plot_trajectory(initial_condition, t_timedata)
    x_true_positions = true_data[:,0]


    ##--------------------- Step 2 --------------------------------##

    """
    Feed into PINN
    """
    random_points = np.linspace(0, len(t_timedata)-1, 50).astype(int)

    t_data_np = t_timedata[random_points]
    x_data_np = x_true_positions[random_points]

    t_data = torch.tensor(t_data_np, dtype=torch.float32).view(-1,1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32).view(-1,1)

    # Generate 400 uniform grid points for internal physics calculus checking
    t_physics = torch.linspace(0, 30, 1000, dtype=torch.float32).view(-1, 1)

    ##--------------------- Step 3 --------------------------------##
    """
    Set Up and train PINN Experiment
    """
    
    NUM_EPOCH = 6000
    EPOCH_INTERVAL = 100
    LEARNING_RATE = 1e-3

    model = PINNInverse(N_INPUT=1, N_OUTPUT=1, 
                        N_HIDDEN=64, 
                        mu_init=INITIAL_MU)
    
    loss_history, mu_history = train_model(
                                    model=model, 
                                    t_data= t_data,
                                    x_data= x_data,
                                    t_physics= t_physics,
                                    num_epochs=NUM_EPOCH, 
                                    epoch_interval=EPOCH_INTERVAL, 
                                    learning_rate=LEARNING_RATE)


    ##--------------------- Step 4 --------------------------------##
    """
    Prediction
    """
    t_test_np = np.linspace(0, 30, 1000)
    t_test_tensor = torch.tensor(t_test_np, dtype=torch.float32).view(-1,1)
    
    model.eval()
    with torch.no_grad():
        x_pred_tensor = model(t_test_tensor)
    
    x_pred_np = x_pred_tensor.numpy().flatten()
    
    #3. Plotting the Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Plot: Trajectory Tracking
    ax1.plot(t_timedata, x_true_positions, 'b-', alpha=0.5, label='True Trajectory (RK4)', lw=2)
    ax1.plot(t_test_np, x_pred_np, 'r--', label='PINN Prediction', lw=2)
    ax1.scatter(t_data_np, x_data_np, color='black', zorder=5, label='Sparse Training Data')
    ax1.set_title(f"Trajectory Reconstruction (Discovered $\mu$={model.mu.item():.2f})")
    ax1.set_xlabel("Time ($t$)")
    ax1.set_ylabel("Position ($x$)")
    ax1.legend()
    ax1.grid(True)
    
    # Right Plot: Parameter Convergence
    ax2.plot(mu_history, color='tab:orange', lw=2, label="Estimated $\mu$")
    ax2.axhline(y=INITIAL_MU, color='red', linestyle='--', label=f"True $\mu$ ({INITIAL_MU})")
    ax2.set_title("System Identification: $\mu$ Convergence")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Value of $\mu$")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    



if __name__ == "__main__":
    main()
