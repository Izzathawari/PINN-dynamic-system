import numpy as np
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

    @staticmethod
    def plot_time_series(alpha=3.99, x0=0.2, steps=100):
        """Plots the time series x_n vs n for a single alpha."""
        model = LogisticMap(alpha=alpha)
        vals = model.run_map(initial_val=x0, transient_state=0, steady_state=steps)


        output_dir = Path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_filename = output_dir/f"time_series.png"
        
        plt.figure(figsize=(9, 4))
        plt.plot(vals, 'o-', markersize=4, linewidth=1, color='teal')
        plt.title(f"Logistic Map Trajectory ($\alpha = {alpha}$, $x_0 = {x0}$)")
        plt.xlabel("Iteration Step ($n$)")
        plt.ylabel(r"$x_n$")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(save_filename)


                