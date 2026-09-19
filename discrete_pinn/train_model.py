import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from model import PINN, LossCalc, LogisticPINN, HenonPINN
from map_func import MapFunction



def evaluate(model, criterion, data):
    x_current, x_next = data
    model.eval()
    with torch.no_grad():
        x_next_pred = model(x_current)
        loss = criterion(model, x_next_pred, x_next, x_current= False, calc_physics = False)
    return loss.item(), x_next_pred


def plot_predictions(pred_data,true_data, save_filename="output_dir/pinn_predictions.png"):

    if not (Path(save_filename)):
        print(f"Error: The path '{save_filename}' does not exist.")
        return

    # Convert PyTorch tensors to NumPy arrays for Matplotlib
    pred_data = pred_data.squeeze().detach().cpu().numpy()
    true_data = true_data.squeeze().detach().cpu().numpy()

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

        case "henon_map":
            map_data = MapFunction()
            x_trajectory, timestep = map_data.run_trajectory("henon_map")
            model = HenonPINN(n_hidden= 8, a_init=0.9 , b_init=0.7)
            

    
    train_data, test_data = map_data.make_data_splits( )
    x_curr, x_next_true = train_data
    x_curr_test, x_next_test = test_data
    print(f"x_current shape: {x_curr.shape}")
    print(f"x_next shape: {x_next_true.shape}")


    # Instantiate Model, Loss, and Optimizer
   

    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    epochs = 500
    pbar = tqdm(range(epochs), desc="Training PINN")
    
    for epoch in pbar:
        model.train()
        
        # Forward pass: predict x_{n+1} from x_n
        x_next_pred = model(x_curr)
        
        # Compute combined loss
        total_loss = criterion( model, x_next_pred, x_next_true,x_curr,calc_physics=True)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            
            postfix_dict = {"Loss": f"{total_loss.item():.5f}"}
            
            # Dynamically add any map parameters to the progress bar
            for name, param in model.named_parameters():
                if "net" not in name:
                    postfix_dict[f"param_{name}"] = f"{param.item():.4f}"
                    
            pbar.set_postfix(postfix_dict)

        

    test_loss, x_next_pred = evaluate(model, criterion, test_data)
    plot_predictions(x_next_pred,x_next_test)
    print(f"Test loss: {test_loss:.5f}")
    print(f"Discovered Parameters: {model.a.item():4f}")
    # for name, param in model.named_parameters():
    #     if "net" not in name:
    #         print(f"  {name} = {param.item():.4f}")


    return model


if __name__ == "__main__":
    train()