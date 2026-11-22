import math
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import PINN, LossCalc

def data_setup (num_steps, dt):
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

    # initialize the double pendulum
    from dp_lagrangian import DoublePendulumLagrangian
    obj = DoublePendulumLagrangian(g, m1, m2, t1, t2, w1, w2, L1, L2)

    timestep, angles = obj.simulate(num_steps=num_steps,dt=dt)
        

    return timestep, angles, (g, m1, m2, t1, t2, w1, w2, L1, L2)


def data_split(timestep,data, train_len: int, val_len: int):

    """
    
    Returns
    ----------
    train_data, validation_data, test_data


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

    return (
            (timestep[:train_len], data[:train_len]),
            (timestep[train_len:val_len], data[train_len:val_len]),
            (timestep[val_len:], data[val_len:]),
    )


def evaluate(model, criterion, eval_data_pair, save_path="eval_plot.png", plot=False):

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
    

    timestep, data, pend_param = data_setup(num_steps=201, dt=0.005)
    train_data, val_data, test_data = data_split(timestep, data, train_len=120, val_len=160 )

    t_train, y_train = train_data 
    t_train = t_train.clone().detach().requires_grad_(True)


    model = PINN(n_input=1, n_hidden=24, n_output=2)
    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    epochs = 8000
    history = {"epoch": [], "train": [], "val": []}

    print(f"---Training Start----")
    for epoch in range(epochs):

        model.train()
        state_pred = model(t_train)

        # Compute combined loss
        total_loss = criterion(t_train,state_pred, y_train, pend_param)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if epoch % 10 == 0:

            val_loss = evaluate(model, criterion, val_data)

            history["train"].append(total_loss.item())
            history["val"].append(val_loss)
            history["epoch"].append(epoch)


    test_loss = evaluate(model, criterion, test_data, plot=True)
    print(f"{"-" * 10} \n TEST LOSS : {test_loss}")

    

    return model,history


def wrap_to_pi(angles):
    """Wraps angles into [-pi, pi] for plotting against Steger et al. Fig. 1."""
    return (angles + np.pi) % (2 * np.pi) - np.pi
        
if __name__  == "__main__":


    timestep, angles, pend_param = data_setup(num_steps=201, dt=0.005)
    wrapped = wrap_to_pi(angles)

    plt.figure(figsize=(8, 4))
    plt.plot(timestep, wrapped[:, 0], color="blue", label="$\\theta_1$")
    plt.plot(timestep, wrapped[:, 1], color="blue", linestyle="--", label="$\\theta_2$")
    plt.ylim(-3.5, 3.5)
    plt.xlabel("Time $t$ [s]")
    plt.ylabel("$\\theta_1, \\theta_2$ [rad]")
    plt.title("Double Pendulum Reference Trajectory (RK45)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("rk45_reference.png", dpi=300)
    plt.show()
    
