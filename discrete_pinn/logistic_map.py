import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class LogisticMap:
    def __init__(self, alpha):
        self.alpha = alpha

    def run_trajectory(self, initial_val, transient_state, steady_state):
        x = initial_val
        
        # Discard transient iterations
        for _ in range(transient_state):
            x = self.alpha * x * (1 - x)
            
        # Collect steady-state / attractor values
        trajectory = []
        for _ in range(steady_state):
            x = self.alpha * x * (1 - x)
            trajectory.append(x)
            
        return np.array(trajectory)

    @staticmethod
    def plot_bifurcation(r_min=2.5, r_max=4.0, num_r=1000, iterations=500, last=100):
        """Plots the bifurcation diagram across a range of alpha values."""
        r_values = np.linspace(r_min, r_max, num_r)
        x = 1e-5 * np.ones(num_r)
        
        # Transient phase
        for _ in range(iterations - last):
            x = r_values * x * (1 - x)
            
        # Plot steady state points
        plt.figure(figsize=(10, 6))
        for _ in range(last):
            x = r_values * x * (1 - x)
            plt.plot(r_values, x, ',k', alpha=0.25)

        output_dir = Path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_filename = output_dir/f"bifurcation_plot.png"

        plt.title("Logistic Map Bifurcation Diagram")
        plt.xlabel(r"Parameter $\alpha$ ($r$)")
        plt.ylabel(r"$x_n$")
        plt.xlim(r_min, r_max)
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(save_filename)

    
    def plot_time_series(self, x0=0.2, steps=100):
        """Plots the time series x_n vs n for a single alpha."""
        model = LogisticMap(alpha=self.alpha)
        vals = model.run_trajectory(initial_val=x0, transient_state=500, steady_state=steps)


        output_dir = Path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_filename = output_dir/f"time_series.png"
        
        plt.figure(figsize=(9, 4))
        plt.plot(vals, 'o-', markersize=4, linewidth=1, color='teal')
        plt.xlabel("Iteration Step ($n$)")
        plt.title(f"Logistic Map Trajectory ($alpha = {self.alpha}$, $x_0 = {x0}$)")
        plt.ylabel(r"$x_n$")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(save_filename)


    def make_data_splits(self,x_trajectory, train_ratio=0.6, validation_ratio=0.2):

        """Convert a trajectory into chronological train, validation, and test pairs."""

        logistic_map = LogisticMap(self.alpha)
        x_trajectory = logistic_map.run_trajectory(initial_val=0.5, transient_state=1000, steady_state=200)


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

    def convert_data2pd(self,excel_filename="output_dir/dataset_splits.xlsx"):
        """
        Converts PyTorch data splits into a hierarchical two-level Excel format:
        Top header:    [      Train      ] [   Validation    ] [      Test       ]
        Sub header:    [  x_n  |  x_n+1  ] [  x_n  |  x_n+1  ] [  x_n  |  x_n+1  ]
        """
        logistic_map = LogisticMap(self.alpha)
        x_trajectory = logistic_map.run_trajectory(initial_val=0.5, transient_state=1000, steady_state=200)


        train_data, validation_data, test_data = LogisticMap.make_data_splits(x_trajectory)

        splits = {
            "Train": train_data,
            "Validation": validation_data,
            "Test": test_data
        }

        split_dfs = {}
        for name, (x_curr, x_next) in splits.items():
            curr_arr = x_curr.squeeze().detach().cpu().numpy()
            next_arr = x_next.squeeze().detach().cpu().numpy()
            
            # Build individual 2-column DataFrame per split
            split_dfs[name] = pd.DataFrame({
                "x_n": curr_arr,
                "x_n+1": next_arr
            })

        # Concatenate side-by-side using keys to form 2-level column headers
        df_hierarchical = pd.concat(split_dfs, axis=1)

        # Path resolution and auto-folder creation
        excel_path = Path(excel_filename)
        excel_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Excel (index=True shows step row numbers 0, 1, 2, ...)
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_hierarchical.to_excel(writer, sheet_name="Data_Splits", index=True)

        print(f"Hierarchical data successfully saved to {excel_path.resolve()}")
        return df_hierarchical

                    