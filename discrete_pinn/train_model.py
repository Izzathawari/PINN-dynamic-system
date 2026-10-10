import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from model import PINN, LossCalc, LogisticPINN, HenonPINN
from data_creation import MapFunction
from data_setup import create_delay_embedding



def evaluate(model, criterion, data):
    x_current, x_next = data
    model.eval()
    with torch.no_grad():
        x_next_pred = model(x_current)
        loss = criterion(model, x_next_pred, x_next, x_current= x_current, calc_physics = False)
    return loss.item(), x_next_pred


def plot_predictions(param_hist, pred_data,true_data, save_filename="output_dir/pinn_predictions.png"):

    if not (Path(save_filename)):
        print(f"Error: The path '{save_filename}' does not exist.")
        return

    # Convert PyTorch tensors to NumPy arrays for Matplotlib
    pred_data = pred_data.squeeze().detach().cpu().numpy()
    true_data = true_data.squeeze().detach().cpu().numpy()

    #--------------------------------#

    plt.figure(figsize=(10, 5))
    plt.title("PINN Predictions - Test Phase")
    plt.xlabel("Time Step")
    plt.ylabel(r"$State Variable$")

    if pred_data.ndim == 1:
        # Logistic Map: Plot a single X variable
        plt.plot(true_data, label="True Data", color="tab:blue", lw=2)
        plt.plot(pred_data, '--', label="NN Prediction", color="tab:orange", lw=2)
    else:
        # Hénon Map: Plot both X (column 0) and Y (column 1) separately
        plt.plot(true_data[:, 0], label="True X", color="tab:blue", lw=2)
        plt.plot(pred_data[:, 0], '--', label="Pred X", color="tab:cyan", lw=2)
        
        plt.plot(true_data[:, 1], label="True Y", color="tab:orange", lw=2)
        plt.plot(pred_data[:, 1], '--', label="Pred Y", color="tab:red", lw=2)

    
    plt.grid(True, linestyle="--", alpha=0.2)
    plt.legend(ncol=2)
    plt.tight_layout()

    output_path = Path(save_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)

    #--------------------------------#
    fig,ax = plt.subplots(2,1, figsize=(9, 6))
    ax[0].plot(param_hist["a"], label="Learned $a$", color="tab:blue", lw=2)
    ax[0].axhline(y=1.4, color="tab:blue", linestyle="--", alpha=0.7, label="True $a$ (1.4)")
    ax[0].set_title("PINN Parameter $a$ and $b$ Convergence")
    ax[0].set_ylabel("Parameter $a$")
    ax[0].grid(True, linestyle="--", alpha=0.3)

    ax[1].plot(param_hist["b"], label="Learned $b$", color="tab:blue", lw=2)
    ax[1].axhline(y=0.3, color="tab:blue", linestyle="--", alpha=0.7, label="True $a$ (1.4)")
    ax[1].set_title("PINN Parameter $b$ Convergence")
    ax[1].set_xlabel("Training Epoch ($\times 10$)")
    ax[1].set_ylabel("Parameter $b$")
    ax[1].grid(True, linestyle="--", alpha=0.3)



    output_path = Path(save_filename)
    plt.savefig(output_path.parent / "system parameter.png", dpi=150)
    plt.close()
    
    # Optional: Display the plot in the window before closing
    plt.show() 
    plt.close()
        




def train(map_func : str):

    """
    Argument : Type of map func

    """
    # Setup data

    match map_func:
        case "logistic_map":
            map_data = MapFunction()
            x_trajectory, timestep = map_data.run_trajectory("logistic_map")
            model = LogisticPINN(n_hidden= 8, r_init=1.99)
            train_data, test_data = map_data.make_data_splits( )
            x_curr_train, x_next_train = train_data
            x_curr_test, x_next_test = test_data

        case "henon_map":
            map_data = MapFunction()
            henon_trajectory, timestep = map_data.run_trajectory("henon_map")
            print(f"Henon_full data:")
            print(f"{henon_trajectory[:5]}")
            model = HenonPINN(n_hidden= 8, a_init=0.9 , b_init=0.7)


            
            x_state_input, x_state_target = create_delay_embedding(henon_trajectory) 
            # Convert NumPy arrays to PyTorch Tensors
            x_state_input = torch.tensor(x_state_input, dtype=torch.float32)  # [N, 2] -> (x_n, x_{n-1})
            x_state_target = torch.tensor(x_state_target, dtype=torch.float32).unsqueeze(1)  # [N, 1] -> x_{n+1}

            split_idx = int(len(henon_trajectory) * 0.8)
            x_curr_train = x_state_input[:split_idx]  # [N, 2] -> (x_n, x_{n-1})
            print(f"X_input for training :")    
            print(f"{x_curr_train[:5]}")              # [N, 1] -> x_{n+1}
            x_curr_test = x_state_input[split_idx:]

            x_next_true_train = x_state_target[:split_idx]
            x_next_true_test = x_state_target[split_idx:]

            test_data = (x_curr_test, x_next_true_test)

           


    # Instantiate Model, Loss, and Optimizer
   

    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    epochs = 1000
    pbar = tqdm(range(epochs), desc=f"Training {map_func}PINN")

    param_history = {"a": [], "b": []}
    
    for epoch in pbar:
        model.train()
        
        # Forward pass: predict x_{n+1} from x_n, x_{n-1}
        x_next_pred = model(x_curr_train) 
        
        # Compute combined loss
        total_loss = criterion( model, x_next_pred, x_next_true_train,x_curr_train,calc_physics=True)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:

           
            for name, param in model.named_parameters():
                if name in ["a", "b"]:
                    param_history[name].append(param.item())
            
            postfix_dict = {"Loss": f"{total_loss.item():.5f}"}
            
            # Dynamically add any map parameters to the progress bar
            for name, param in model.named_parameters():
                if "net" not in name:
                    postfix_dict[f"param_{name}"] = f"{param.item():.4f}"
                    
            pbar.set_postfix(postfix_dict)

        

    test_loss, x_next_pred = evaluate(model, criterion, test_data)
    plot_predictions(param_history,x_next_pred,x_next_true_test)
    print(f"Test loss: {test_loss:.5f}")
    print("Discovered Parameters:")
    for name, param in model.named_parameters():
        if "net" not in name:
            print(f"  {name} = {param.item():.4f}")


    return model


if __name__ == "__main__":
    train()