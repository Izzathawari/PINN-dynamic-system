import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import scipy.stats as stats


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


            self.trajectory = np.array(data)
            return self.trajectory, np.arange(0,steady_step,1)

        if map_function == "logistic_map_with_noise":
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
            time_series_data  = []
            for _ in range(steady_step):
                state_x = self.logistic_map(state_x)
                time_series_data.append(state_x)

            # --- 2. Apply the Paper's Noise Formula ---
            eta = 0.05  # Signal Noise Ratio %
            sigma_x = np.std(time_series_data)  # Standard deviation of the clean signal
            sigma_noise = eta * sigma_x  # Scaled noise standard deviation

            # Generate white noise using the paper's formula
            white_noise = np.random.normal(loc=0.0, scale=sigma_noise, size=len(time_series_data))

            # Construct noisy time series s(k)
            noisy_data = time_series_data + white_noise

            self.trajectory = np.array(noisy_data)
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

        Returns:
            Train_data, Test_data : (N,1),(N,1)
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
    logistic_map, _ = map_data.run_trajectory("logistic_map")

    map_data2 = MapFunction()
    logistic_map_noisy, _ = map_data2.run_trajectory("logistic_map_with_noise")

    plt.figure(figsize=(8, 6))
    plt.plot(logistic_map[:-1],
        logistic_map[1:],
        ".",
        color="tab:blue",
        alpha=0.6,
        label="Clean",)
    
    plt.plot(
        logistic_map_noisy[:-1],
        logistic_map_noisy[1:],
        ".",
        color="tab:red",
        alpha=0.6,
        label="Noisy",)
    plt.xlabel(r"$x_n$")
    plt.ylabel(r"$x_{n+1}$")
    plt.title("Logistic Map Phase Plot")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    
    
    plt.figure(figsize=(8, 5), dpi=150)
    count, bins, ignored = plt.hist(logistic_map, bins=100, density=True, alpha=0.6, color="black", label="Clean")
    count, bins, ignored = plt.hist(logistic_map_noisy, bins=100, density=True, alpha=0.6, color="blue", label="Noisy")

    x_pdf = np.linspace(0.001, 0.999, 500)
    logistic_pdf = 1.0 / (np.pi * np.sqrt(x_pdf * (1.0 - x_pdf)))

    plt.plot(
        x_pdf,
        logistic_pdf,
        "r-",
        linewidth=2.5,
        label="Theoretical Logistic PDF ($r=4$)"
    )



    extracted_noise = logistic_map_noisy - logistic_map
    sigma_actual = np.std(extracted_noise)
    mean_noise= np.mean(extracted_noise)
    print(f"Mean : {mean_noise} , Std Dev: {sigma_actual}")
    plt.figure(figsize=(9, 5), dpi=150)
    count, bins, ignored = plt.hist(
        extracted_noise,
        bins=60,
        density=True,
        alpha=0.6,
        color="black",
        label="Empirical Added Noise",
    )

    # 3. Plot the Correct Theoretical Gaussian PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    # Use mean = 0 and scale = sigma_actual (derived from your eta * sigma_x formula)
    p = stats.norm.pdf(x, loc=0.0, scale=sigma_actual)

    plt.plot(
        x,
        p,
        "r-",
        linewidth=2.5,
        label=f"Theoretical $\mathcal{{N}}(0, \sigma^2)$ ($\sigma$={sigma_actual:.4f})",
    )

    plt.title("Verification of Additive White Gaussian Noise", fontsize=13)
    plt.xlabel("Noise Magnitude", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()
