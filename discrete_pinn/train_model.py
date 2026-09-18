import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from model import PINN, LossCalc
from logistic_map import LogisticMap



def evaluate(model, criterion, data):
    x_current, x_next = data
    model.eval()
    with torch.no_grad():
        x_next_pred = model(x_current)
        loss = criterion(x_next_pred, x_next)
    return loss.item(), x_next_pred


def plot_predictions(pred_data,true_data, save_filename="output_dir/pinn_predictions.png"):

    if not (Path(save_filename)):
        print(f"Error: The path '{save_filename}' does not exist.")
        return

    # Convert PyTorch tensors to NumPy arrays for Matplotlib
    pred_data = pred_data.detach().cpu().numpy()
    true_data = true_data.detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.title("PINN Predictions - Test Phase")
    plt.xlabel("Transition Index (Time Step)")
    plt.ylabel(r"$x_{n+1}$")
    
    # Plot actual test data vs predicted data
    plt.plot(true_data, label="True Data", color="tab:blue", lw=2)
    plt.plot(pred_data, '--', label="NN Prediction", color="tab:orange", lw=2)
    
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2)
    plt.tight_layout()

    output_path = Path(save_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    
    # Optional: Display the plot in the window before closing
    plt.show() 
    plt.close()
        




def train():
    # Setup data
    logistic_map = LogisticMap(alpha=3.99)
    x_trajectory, timestep = logistic_map.run_trajectory("logistic_map")
    train_data, test_data = logistic_map.make_data_splits( )

    x_curr, x_next_true = train_data
    x_curr_test, x_next_test = test_data
    print(f"x_current shape: {x_curr.shape}")
    print(f"x_next shape: {x_next_true.shape}")


    # Instantiate Model, Loss, and Optimizer
   

    model = PINN(N_INPUT =1, N_HIDDEN= 8, N_OUTPUT=1, R_INIT=1.99)
    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    epochs = 500
    pbar = tqdm(range(epochs), desc="Training PINN")
    
    for epoch in pbar:
        model.train()
        
        # Forward pass: predict x_{n+1} from x_n
        x_next_pred = model(x_curr)
        
        # Compute combined loss
        total_loss = criterion( x_next_pred, x_next_true, model.r, x_curr)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.5f}",
            })

        

    test_loss, x_next_pred = evaluate(model, criterion, test_data)
    plot_predictions(x_next_pred,x_next_test)
    print(f"Test loss: {test_loss:.5f}")
    print(f"Discovered r: {model.r.item():.4f}")


    return model


if __name__ == "__main__":
    train()