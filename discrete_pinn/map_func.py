import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd



class MapFunction:
    def __init__(self):
        self.alpha = None
        self.trajectory = None
        self.a = None
        self.b = None

    def logistic_map(self, state_x):
         return self.alpha*state_x - self.alpha*state_x**2

    def henon_map (self, state_x, state_y,a,b):
         return 1-a*state_x**2 + state_y, b*state_x

    def run_trajectory(self, map_function, transient_step= 500, steady_step=100):
        

        if map_function == "logistic_map":

            state_x = 0.2
            self.alpha = 3.99
            # Discard transient iterations
            for _ in range(transient_step):
                state_x = self.logistic_map(state_x)
                
            # Collect steady-state / attractor values
            data  = []
            for _ in range(steady_step):
                state_x = self.logistic_map(state_x)
                data.append(state_x)

            self.trajectory = np.array(data)
            return self.trajectory, np.arange(0,steady_step,1)

        if map_function == "henon_map":

            state_x = 0.1
            state_y = 0.1
            a = self.a = 1.4
            b = self.b = 0.3
         
             # Discard transient iterations
            for _ in range(transient_step):
                state_x, state_y = self.henon_map(state_x,state_y,a,b)


            # Collect steady-state / attractor values
            data  = []
            for _ in range(steady_step):
                state_x, state_y = self.henon_map(state_x,state_y,a,b)
                data.append([state_x,state_y])

            self.trajectory = np.array(data)
            return self.trajectory, steady_step



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

    
    def plot_time_series(self, map_function):
        """Plots the time series x_n vs n for a single alpha."""
        
        if map_function == "logistic_map":

            if self.trajectory is None:
                raise ValueError ("No trajectory Found !!")

            output_dir = Path("output_dir")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_filename = output_dir/f"time_series.png"
            
            plt.figure(figsize=(9, 4))
            plt.plot(self.trajectory, 'o-', markersize=4, linewidth=1, color='teal')
            plt.xlabel("Iteration Step ($n$)")
            plt.title(f"{map_function} Trajectory ($alpha = {self.alpha}$)")
            plt.ylabel(r"$x_n$")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(save_filename)


        if map_function=="henon_map":
            if self.trajectory is None:
                            raise ValueError ("No trajectory Found !!")

            output_dir = Path("output_dir")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_filename = output_dir/f"{map_function}time_series.png"

            fig,ax = plt.subplots(3,1, figsize=(9, 6))

            ax[0].plot(self.trajectory[:, 0],
                       "o-",markersize=3,
                       linewidth=1,
                       color="teal",)
            ax[0].set_title(f"{map_function} X Trajectory")
            ax[0].set_ylabel(r"$x_n$")
            ax[0].grid(True, linestyle="--", alpha=0.5)

            ax[1].plot(
                    self.trajectory[:, 1],
                    "o-",
                    markersize=3,
                    linewidth=1,
                    color="coral",)
            
            ax[1].set_title(f"{map_function} Y Trajectory")
            ax[1].set_xlabel("Iteration Step ($n$)")
            ax[1].set_ylabel(r"$y_n$")
            ax[1].grid(True, linestyle="--", alpha=0.5)

            ax[2].plot(self.trajectory[:,0], self.trajectory[:,1], ",", lw=0.9)
            ax[2].set_title(f"{map_function}")
            ax[2].set_ylabel(r"$y_n$")
            ax[2].set_xlabel(r"$x_n$")
            ax[2].grid(True, linestyle="--", alpha=0.5)


            plt.tight_layout()
            fig.savefig(save_filename, bbox_inches="tight")
            plt.close(fig)

    def make_data_splits(self,train_ratio=0.6):

        """Convert a trajectory into chronological train, validation, and test pairs."""
        if self.trajectory is None:
                    raise ValueError ("No trajectory Found !!")
        
        trajectory = self.trajectory 

        x_current = torch.tensor(trajectory[:-1], dtype=torch.float32).unsqueeze(1)
        x_next = torch.tensor(trajectory[1:], dtype=torch.float32).unsqueeze(1)

        if x_current.dim() == 1:
            x_current = x_current.unsqueeze(1)
            x_next = x_next.unsqueeze(1)
            
        n_samples = len(x_current)
        train_end = int(train_ratio * n_samples)
        print(f"Train End index : {train_end}")


        return (
            (x_current[:train_end], x_next[:train_end]), # Train Length
            (x_current[train_end:], x_next[train_end:]), #Test 
           
        )

    def convert_data2pd(self,excel_filename="output_dir/dataset_splits.xlsx"):
        """
        Converts PyTorch data splits into a hierarchical two-level Excel format:
        Top header:    [      Train      ] [   Validation    ] [      Test       ]
        Sub header:    [  x_n  |  x_n+1  ] [  x_n  |  x_n+1  ] [  x_n  |  x_n+1  ]
        """
    

        train_data, test_data = self.make_data_splits()
        print(f"Train data / Test Data {np.shape(train_data)}")

        splits = {
            "Train": train_data,
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

                    