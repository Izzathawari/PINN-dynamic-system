import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os

from model import PINN
from physics_loss import LossCalc
from logistic_map import LogisticMap


def make_data_splits(x_trajectory, train_ratio=0.6, validation_ratio=0.2):
    """Convert a trajectory into chronological train, validation, and test pairs."""
    x_current = torch.tensor(x_trajectory[:-1], dtype=torch.float32).unsqueeze(1)
    x_next = torch.tensor(x_trajectory[1:], dtype=torch.float32).unsqueeze(1)

    n_samples = len(x_current)
    train_end = int(train_ratio * n_samples)
    validation_end = int((train_ratio + validation_ratio) * n_samples)


    return (
        (x_current[:train_end], x_next[:train_end]),
        (x_current[train_end:validation_end], x_next[train_end:validation_end]),
        (x_current[validation_end:], x_next[validation_end:]),
    )

def convert_data2pd(excel_filename="output_dir/dataset_splits.xlsx"):
    """
    Converts PyTorch data splits into Pandas DataFrames and saves them
    into a multi-sheet Excel file.
    """
    logistic_map = LogisticMap(alpha=2)
    x_trajectory = logistic_map.run_trajectory(initial_val=0.5, transient_state=1000, steady_state=200)
    train_data, validation_data, test_data = make_data_splits(x_trajectory)
    splits = {
        "Train": train_data,
        "Validation": validation_data,
        "Test": test_data
    }
    
    dfs = {}
    combined_rows = []
    
    global_idx = 0
    for split_name, (x_curr, x_next) in splits.items():
        curr_arr = x_curr.squeeze().detach().cpu().numpy()
        next_arr = x_next.squeeze().detach().cpu().numpy()
        
        df_split = pd.DataFrame({
            "step_index": np.arange(len(curr_arr)),
            "x_n": curr_arr,
            "x_next_true": next_arr
        })
        dfs[split_name] = df_split
        
        for local_idx, (xn, xnext) in enumerate(zip(curr_arr, next_arr)):
            combined_rows.append({
                "global_index": global_idx,
                "split": split_name,
                "local_index": local_idx,
                "x_n": xn,
                "x_next_true": xnext
            })
            global_idx += 1

    df_all = pd.DataFrame(combined_rows)
    dfs["All"] = df_all

    # Path resolution and auto-folder creation
    excel_path = Path(excel_filename)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="All_Data", index=False)
        for name in ("Train", "Validation", "Test"):
            dfs[name].to_excel(writer, sheet_name=name, index=False)

    print(f"Data successfully saved to {excel_path.resolve()}")
    return dfs

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
    split_names = ("Train", "Validation", "Test")
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
    logistic_map = LogisticMap(alpha=2)
    x_trajectory = logistic_map.run_trajectory(initial_val=0.5, transient_state=1000, steady_state=200)
    train_data, validation_data, test_data = make_data_splits(x_trajectory)
    x_curr, x_next = train_data
    x_curr_init = x_curr[0]

    # Instantiate Model, Loss, and Optimizer
    N_INPUT = 1
    N_HIDDEN = 16
    N_OUTPUT = 1
    R_INIT = 1.99

    model = PINN(N_INPUT, N_HIDDEN, N_OUTPUT, R_INIT)
    criterion = LossCalc()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    epochs = 2000
    pbar = tqdm(range(epochs), desc="Training PINN")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: predict x_{n+1} from x_n
        x_next_pred = model(x_curr)
        
        # Compute combined loss
        total_loss = criterion(x_curr_init, x_next_pred, x_next, model.r)
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.5f}",
                "Val loss": f"{evaluate(model, criterion, validation_data):.5f}",
                "Discovered r": f"{model.r.item():.4f}"
            })

        x_curr_init = x_next_pred[0]

    test_loss = evaluate(model, criterion, test_data)
    plot_predictions(model, (train_data, validation_data, test_data))
    print(f"Test loss: {test_loss:.5f}")
    print(f"Discovered r: {model.r.item():.4f}")
    return model


if __name__ == "__main__":
    train()