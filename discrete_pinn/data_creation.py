import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

from data_setup import create_delay_embedding 



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

    def run_trajectory(self, map_function, transient_step= 1000, steady_step=5000):
        """
        Argument :
        map_function (ex: logistic_map)

        Process: Run the system for 500 iters, plot the 600 iters (steady_steps)
       
        Return:  Vector of Map time series (N_timestep,2) , Scalar of N_timestep
        
          """

        if map_function == "logistic_map":
            """
            Return:
            (N,1) -> (x_state)
    
            """
    

            state_x = 0.8
            self.alpha = 3.99
            # Discard transient iterations
            for _ in range(transient_step):
                state_x = self.logistic_map(state_x)
                
            # Collect steady-state / attractor values
            data  = []
            for _ in range(steady_step):
                state_x = self.logistic_map(state_x)
                data.append(state_x)


            # ---  Add White Gaussian Noise (Measurement Noise) ---
            # eta = 0.2
            # sigma_logistic_map = np.std(data)
            # sigma =eta*sigma_logistic_map
            # mean = 0
            # white_noise = np.random.normal(mean, sigma, steady_step)
            # ------------------------------------------------------

            self.trajectory = np.array(data)
            return self.trajectory, np.arange(0,steady_step,1)

        if map_function == "henon_map":
            """
            Return:
            (N,2) -> (x_state , y_state)

            """

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

    def make_data_splits(self,train_ratio=0.8):

        """Convert a trajectory into chronological train, 
        validation, and test pairs.

        Logistic Map : (N_timestep, 1)
        Henon Map : (N_timestep,2)
        """
        if self.trajectory is None:
                    raise ValueError ("No trajectory Found !!")
        
        trajectory = self.trajectory 

        x_current = torch.tensor(trajectory[:-1], dtype=torch.float32)
        x_next = torch.tensor(trajectory[1:], dtype=torch.float32)

        if x_current.dim() == 1:
            x_current = x_current.unsqueeze(1)
            x_next = x_next.unsqueeze(1)
            
        n_samples = len(x_current)
        train_end = int(train_ratio * n_samples)


        return (
            (x_current[:train_end], x_next[:train_end]), # Train Length
            (x_current[train_end:], x_next[train_end:]), #Test 
           
        )

    def convert_data2pd(self,output_dir="output_dir/"):
        """
        Converts delay-embedded data splits into a hierarchical two-level Excel format:
        Returns:
            Top header:    [         Train          ] [           Test           ]
            Sub header:    [ x_{n-1} | x_n | x_{n+1} ] [ x_{n-1} | x_n | x_{n+1} ]
        """
        if self.trajectory is None:
            raise ValueError("No trajectory found!")

        # 1. Extract 1D series from trajectory (take column 0 if 2D Hénon map)
        if self.trajectory.ndim == 2:
            x_series = self.trajectory[:, 0:1]
        else:
            x_series = self.trajectory[:, np.newaxis]

        # 2. Generate delay embedding (inputs: [x_n, x_{n-1}], target: x_{n+1})
        x_input, x_target = create_delay_embedding(x_series)

        # Reorder/Combine columns so they appear logically as [x_{n-1}, x_n, x_{n+1}]
        # x_input[:, 0] is x_n, x_input[:, 1] is x_{n-1}
        x_prev = x_input[:, 1:2]  # x_{n-1}       , shape (N,1)
        x_curr = x_input[:, 0:1]  # x_n           , shape (N,1)
        x_next = x_target.reshape(-1,1)# x_{n+1}  , shape (N,1)

        full_data = np.hstack([x_prev, x_curr, x_next])  # Shape [N, 3]

        # 3. Chronological Train/Test Split (80% Train, 20% Test)
        n_samples = len(full_data)
        train_end = int(0.8 * n_samples)

        train_arr = full_data[:train_end]
        test_arr = full_data[train_end:]

        col_names = ["x_{n-1}", "x_n", "x_{n+1}"]

        df_train = pd.DataFrame(train_arr, columns=col_names)
        df_test = pd.DataFrame(test_arr, columns=col_names)

        splits = {"Train": df_train, "Test": df_test}

        # Concatenate side-by-side to form 2-level column headers
        df_hierarchical = pd.concat(splits, axis=1)

        # 4. Save to Excel
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        excel_file = output_path / "dataset_split_delay_embedding.xlsx"

        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df_hierarchical.to_excel(writer, index=True)

        print(f"Hierarchical delay-embedded data saved to: {excel_file.resolve()}")
        return df_hierarchical


if __name__ == "__main__":
    import numpy as np

    map_data = MapFunction()


    henon_traj, henon_time = map_data.run_trajectory("henon_map", steady_step=100)
    print("--- Hénon Map ---")
    print(f"Trajectory shape: {henon_traj.shape}")
    #--- Hénon Map X state---
    #Trajectory shape: (100, 2)
    #

    henon_input, henon_target = create_delay_embedding(henon_traj) 
    # henon_input[:,0:1] = X_n   (,1)
    # henon_input[:,1:2] = X_n-1 (,1)
    # henon_target[:,:] = X_n+1  (,)

    map_data.convert_data2pd()
    
