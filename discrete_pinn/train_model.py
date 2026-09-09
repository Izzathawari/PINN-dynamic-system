import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from model import PINN
from physics_loss import LossCalc
from logistic_map import LogisticMap



def evaluate(model, criterion, data):
    x_current, x_next = data
    model.eval()
    with torch.no_grad():
        x_pred = model(x_current)
        loss = criterion(x_current, x_pred, x_next, model.r)
    return loss.item()


def plot_predictions(model, datasets, save_filename="output_dir/pinn_predictions.png"):

    if not (Path(save_filename)):
        print(f"Error: The path '{save_filename}' does not exist.")
        return
        

    """Plot observed and predicted transitions for all data splits."""
    split_names = ("Train", "Test")
    split_colors = ("tab:blue", "tab:orange", "tab:green")
    split_predictions = []

    model.eval()
    with torch.no_grad():
        for x_current, x_next in datasets:
            split_predictions.append((
                x_next.squeeze(1).cpu().numpy(),
                model(x_current).squeeze(1).cpu().numpy(),
            ))

    plt.figure(figsize=(11, 5))
    offset = 0
    for split_name, split_color, (observed, predicted) in zip(split_names, split_colors, split_predictions):
        steps = np.arange(offset, offset + len(observed))
        plt.plot(steps, observed, color=split_color, label=f"{split_name} observed")
        plt.plot(
            steps,
            predicted,
            "--",
            color="black",
            alpha=0.8,
            label=f"{split_name} predicted",
        )
        offset += len(observed)

    plt.title(f"PINN Predictions Across Data Splits (r = {model.r.item():.4f})")
    plt.xlabel("Transition index")
    plt.ylabel(r"$x_{n+1}$")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2)
    plt.tight_layout()

    output_path = Path(save_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def train():
    # Setup data
    logistic_map = LogisticMap(alpha=3.99)
    x_trajectory = logistic_map.run_trajectory()
    train_data, test_data = logistic_map.make_data_splits(x_trajectory)
    x_curr, x_next = train_data

    # Instantiate Model, Loss, and Optimizer
   

    model = PINN(N_INPUT =1, N_HIDDEN= 8, N_OUTPUT=1, R_INIT = 1.99)
    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    epochs = 500
    pbar = tqdm(range(epochs), desc="Training PINN")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: predict x_{n+1} from x_n
        x_next_pred = model(x_curr)
        
        # Compute combined loss
        total_loss = criterion(x_curr, x_next_pred, x_next, model.r)
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.5f}",
                "Discovered r": f"{model.r.item():.4f}"
            })

        

    test_loss = evaluate(model, criterion, test_data)
    plot_predictions(model, (train_data,  test_data))
    print(f"Test loss: {test_loss:.5f}")
    print(f"Discovered r: {model.r.item():.4f}")
    return model


if __name__ == "__main__":
    train()