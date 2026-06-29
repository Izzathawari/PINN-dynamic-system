from vdp_func import VDP
from pinn import PINNInverse
import numpy as np
import torch



def main ():

    ##-------------------Step 1----------------------------------##
    """
    Create observable data
    """
    mu = 1.8
    vanderpol = VDP(mu = mu)

    t_timedata = np.linspace(0, 30,1000)
    initial_condition = [0.0,-1e-3]
    print(f"-----Preparing Vander Pol Data via RK4-----")
    true_data = vanderpol.generate_trajectory(initial_condition, t_array)
    x_true_positions = true_data[:,0]


    ##--------------------- Step 2 --------------------------------##

    """
    Feed into PINN
    """
    random_points = np.arange(0, len(t_timedata), 100)

    t_data_np = t_timedata[random_points]
    x_data_np = x_true_positions[random_points]

    t_data = torch.tensor(t_data_np, dtype=torch.float32).view(-1,1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32).view(-1,1)

    # Generate 400 uniform grid points for internal physics calculus checking
    t_physics = torch.linspace(0, 30, 400, dtype=torch.float32).view(-1, 1)

    ##--------------------- Step 3 --------------------------------##
    """
    Set Up PINN Experiment
    """
    INITIAL_MU = 1.0
    model = PINNInverse(N_INPUT=1, N_OUTPUT=1, N_HIDDEN=64, mu_init=INITIAL_MU)

    ##--------------------- Step 4 --------------------------------##
    """
    
    """



if __name__ == "__main__":
    main()
