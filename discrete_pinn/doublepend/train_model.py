import math
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import PINN, LossCalc

def data_setup (num_steps, dt, csv_path= "dp_data.csv", force_recompute=False):
    """

    Arguments
    ------------
    num_steps
        The time step of the whole 
        simulation of Double Pendulum
    dt
        Small steps for Rungge Kutta calculation

    Returns
    ------------
    timestep
        The timestep uses for the whole simulation

    theta_1, theta_2
        The angle of bob 1 and bob 2
    
    pendulum_param
        Pendulum parameter consist of (g, m1, m2, t1, t2, w1, w2, L1, L2)

    
    """
    timestep = []
    angles = []


    # default simulation parameter values
    g = 9.8
    #m1 = float(input("Mass 1 : "))
    m1, m2 = 1, 1
    t1, t2 = 2.62, 2.62 # 0.9 = 51 deg, 0.2 = 11 deg, 1.5 = 86 deg
    w1 = w2 = 0
    L1 = L2 = 1
    pend_param  = (g, m1, m2, t1, t2, w1, w2, L1, L2)

    # if not force_recompute and os.path.exists(csv_path):
    #     df = pd.read_csv(csv_path)
    #     same_len = len(df) == num_steps
    #     same_dt = num_steps < 2 or np.isclose(df["t"].iloc[1] - df["t"].iloc[0], dt)
    #     same_ic = np.isclose(df["theta1"].iloc[0], t1) and np.isclose(df["theta2"].iloc[0], t2)

    #     if same_len and same_dt and same_ic:
    #         timestep = df[["t"]].to_numpy(dtype=np.float64)              # shape: (N, 1)
    #         angles = df[["theta1", "theta2"]].to_numpy(dtype=np.float64)  # shape: (N, 2)
    #         print(f"Loaded cached RK45 data from '{csv_path}'")
    #         return timestep, angles, pend_param

    # Compute RK45 trajectory and save via Pandas
    from dp_lagrangian import DoublePendulumLagrangian
    obj = DoublePendulumLagrangian(*pend_param)
    timestep, angles = obj.simulate(num_steps=num_steps, dt=dt)

    # df = pd.DataFrame({
    #     "t": timestep.flatten(),
    #     "theta1": angles[:, 0],
    #     "theta2": angles[:, 1],
    # })
    # df.to_csv(csv_path, index=False, float_format="%.17g")
    # print(f"Computed and saved RK45 data to '{csv_path}'")

    return timestep, angles, pend_param

def data_split(timestep,data, train_len_percent: float, val_len_percent: float):

    """
    
    Returns
    ----------
    train data
        (train_len_timestep,train_data) 
    
    validation data
        (validation_len_timestep, validation_data)
    
    test data
        (test_len_timestep, test_data)


    Arguments
    ----------
    data
        Data used in model
    
    train_len
        Training length
    
    val_len
        Validation length


    Raise
    ---------
    TypeError  
        If no arguments are input
    """

    timestep = torch.tensor(timestep,dtype=torch.float32)
    data = torch.tensor(data, dtype=torch.float32)

    n_total = len(data)
    train_end = int(round(train_len_percent * n_total))
    val_end = train_end + int(round(val_len_percent * n_total))

    return (
            (timestep[:train_end], data[:train_end]),
            (timestep[train_end:val_end], data[train_end:val_end]),
            (timestep[val_end:], data[val_end:]),
    )


def evaluate(model, criterion, eval_data_pair,  fulldata= False, timestep=False, save_path="eval_plot.png", plot=False):

    timestep, eval_data = eval_data_pair
    model.eval()
    with torch.no_grad():
        state_pred = model(timestep)
        loss = criterion(timestep,state_pred, eval_data, pend_param= None)

    t_np = timestep.detach().cpu().numpy().flatten()
    true_np = eval_data.detach().cpu().numpy()
    pred_np = state_pred.detach().cpu().numpy()

    true_np = wrap_to_pi(true_np)
    pred_np = wrap_to_pi(pred_np)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Theta 1
        axes[0].plot(t_np, true_np[:, 0], label="True $\\theta_1$ (RK4)", color="black", linewidth=1.8)
        axes[0].plot(t_np, pred_np[:, 0], label="Predicted $\\theta_1$ (PINN)", color="tab:red", linestyle="--", linewidth=1.8)
        axes[0].set_ylabel("$\\theta_1$ [rad]")
        axes[0].set_title("$\\theta_1$ , $\\theta_2$ Prediction vs True Data")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Theta 2
        axes[1].plot(t_np, true_np[:, 1], label="True $\\theta_2$ (RK4)", color="black", linewidth=1.8)
        axes[1].plot(t_np, pred_np[:, 1], label="Predicted $\\theta_2$ (PINN)", color="tab:blue", linestyle="--", linewidth=1.8)
        axes[1].set_xlabel("Time $t$ [s]")
        axes[1].set_ylabel("$\\theta_2$ [rad]")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
        

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close(fig)

    return loss.item()

def get_prediction (model, timestep, data):
    timestep = torch.tensor(timestep, dtype=torch.float32)
    y_true = np.array(data)  # shape: (N, 2) -> [theta_1_true, theta_2_true]

    model.eval()
    with torch.no_grad():
        y_pred = model(timestep).cpu().numpy()  # shape: (N, 2) -> [theta_1_pred, theta_2_pred]

    return {
        "t": np.array(timestep).flatten(),
        "theta1_true": y_true[:, 0],
        "theta2_true": y_true[:, 1],
        "theta1_pred": y_pred[:, 0],
        "theta2_pred": y_pred[:, 1],
    }

def train_model ():
    

    timestep, data, pend_param = data_setup(num_steps=401, dt=0.005)
    train_data, val_data, test_data = data_split(timestep, data, train_len_percent=0.6, val_len_percent=0.2 )

    t_train, y_train = train_data 
    t_train = t_train.clone().detach().requires_grad_(True)


    model = PINN(n_input=1, n_hidden=30, n_output=2)
    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 10000
    history = {"epoch": [], "train": [], "val": []}

    print(f"---Training Start----")
    for epoch in range(epochs):

        model.train()
        state_pred = model(t_train)

        # Compute combined loss
        total_loss = criterion(t_train,state_pred, y_train, pend_param, ic_loss=True)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if epoch % 10 == 0:

            val_loss = evaluate(model, criterion, val_data)

            history["train"].append(total_loss.item())
            history["val"].append(val_loss)
            history["epoch"].append(epoch)


    train_loss = evaluate(model, criterion, train_data,save_path="train_plot.png", plot=True)
    print(f"---------- \n TRAIN LOSS : {train_loss}")
    test_loss = evaluate(model, criterion, test_data, data,timestep,save_path="test.png",plot=True)
    print(f"{"-" * 10} \n TEST LOSS : {test_loss}")

    plt.plot(history["epoch"],history["train"],  color="black", linewidth=1.8)
    plt.plot(history["epoch"],history["val"], color="tab:red", linestyle="--", linewidth=1.8)
    plt.grid(True, alpha=0.3)
    plt.show()

    

    return model,history


def wrap_to_pi(angles):
    """Wraps angles into [-pi, pi] for plotting against Steger et al. Fig. 1."""
    return (angles + np.pi) % (2 * np.pi) - np.pi



        
if __name__  == "__main__":


    pass
    
