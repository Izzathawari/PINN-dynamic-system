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
    TARGET_MU = 0.1
    vanderpol = VDP(mu = TARGET_MU)

    t_timedata = np.linspace(0, 50,1000)
    initial_condition = [0.5,0.0]
    print(f"\n-----Preparing Vander Pol Data via RK4-----")
    true_data = vanderpol.generate_trajectory(initial_condition, t_timedata)
    plot_data = vanderpol.plot_trajectory(initial_condition, t_timedata)
    x_true_positions = true_data[:,0]
    v_true_positions = true_data[:,1]


    ##--------------------- Step 2 --------------------------------##

    """
    Feed into PINN
    """
    TRAINING_POINTS = 1000
    COLLOCATION_POINTS = 1000
    COLLOCATION_POINTS_START = 0
    COLLOCATION_POINTS_END = 40
    TRAIN_LENGTH = 40
    DELTA_T = 0.05

    t_full_np = np.arange(0, 50 + DELTA_T / 2, DELTA_T)

    t_train_np = t_full_np[t_full_np <= 40.0]
    t_test_np  = t_full_np[t_full_np >= 40.0]

    x_train_np = vanderpol.generate_trajectory(initial_condition, t_train_np)[:, 0]

    t_train_data = torch.tensor(t_train_np, dtype=torch.float32).view(-1, 1)
    x_train_data = torch.tensor(x_train_np, dtype=torch.float32).view(-1, 1)

    # Collocation points use the exact same train grid
    t_physics = t_train_data.clone()
    # Collocation points cover the FULL range (0s to 50s -> 1001 points)
    # t_physics = torch.tensor(t_full_np, dtype=torch.float32).view(-1, 1)

    t_test_tensor = torch.tensor(t_test_np, dtype=torch.float32).view(-1,1)

    ##--------------------- Step 3 --------------------------------##
    """
    Set Up and train PINN Experiment
    """
    
    NUM_EPOCH = 5000
    EPOCH_INTERVAL = 100
    LEARNING_RATE = 1e-2
    INITIAL_MU = 0.5

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
    PREDICT_LENGTH_START = 40
    PREDICT_LENGTH_END = 50


   

    predict_model(model, t_test_tensor,t_timedata,x_true_positions,v_true_positions ,t_train_data, x_train_data,mu_history,INITIAL_MU, TARGET_MU)
    
    
if __name__ == "__main__":
    main()
