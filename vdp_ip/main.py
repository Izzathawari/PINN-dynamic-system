from vdp_func import VDP
from train import train_model, predict_model
from pinn import PINNInverse
import numpy as np
import torch
import matplotlib.pyplot as plt

def main ():

    ##-------------------Step 1----------------------------------##
    """
    Create observable data
    """
    TARGET_MU = 1.8
    vanderpol = VDP(mu = TARGET_MU)

    t_timedata = np.linspace(0, 30,1000)
    initial_condition = [0.5,0.0]
    print(f"-----Preparing Vander Pol Data via RK4-----")
    true_data = vanderpol.generate_trajectory(initial_condition, t_timedata)
    plot_data = vanderpol.plot_trajectory(initial_condition, t_timedata)
    x_true_positions = true_data[:,0]
    v_true_positions = true_data[:1]


    ##--------------------- Step 2 --------------------------------##

    """
    Feed into PINN
    """
    random_points = np.linspace(0, len(t_timedata)-1, 50).astype(int)

    t_data_np = t_timedata[random_points]
    x_data_np = x_true_positions[random_points]

    t_train_data = torch.tensor(t_data_np, dtype=torch.float32).view(-1,1)
    x_train_data = torch.tensor(x_data_np, dtype=torch.float32).view(-1,1)

    # Generate 400 uniform grid points for internal physics calculus checking
    t_physics = torch.linspace(0, 30, 1000, dtype=torch.float32).view(-1, 1)

    ##--------------------- Step 3 --------------------------------##
    """
    Set Up and train PINN Experiment
    """
    
    NUM_EPOCH = 6000
    EPOCH_INTERVAL = 100
    LEARNING_RATE = 1e-3
    INITIAL_MU = 0.9

    model = PINNInverse(N_INPUT=1, N_OUTPUT=1, 
                        N_HIDDEN=64, 
                        mu_init=INITIAL_MU)
    
    loss_history, mu_history = train_model(
                                    model=model, 
                                    t_data= t_train_data,
                                    x_data= x_train_data,
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

    predict_model(model, t_test_tensor,t_timedata,x_true_positions,v_true_positions ,t_train_data, x_train_data,mu_history,INITIAL_MU)
    
    
if __name__ == "__main__":
    main()
