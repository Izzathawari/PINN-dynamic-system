import numpy as np
import matplotlib.pyplot as plt
 
class VDP:
    def __init__(self,  mu):
        """
        Initialize the Van der Pol oscillator with its physical parameters.
        omega: Natural frequency of the system
        nu: Non-linear damping coefficient (often written as epsilon)
        """
        self.mu = mu

    def func(self, r, t):
        """
        Defines the system of 1st order ODEs for the Van der Pol oscillator.
        r: array [x, v] where x is position and v is velocity (dx/dt)
        t: independent variable (time)
        """
        x = r[0]
        v = r[1]
        
        # dx/dt = v
        fx = v
        # dv/dt = -omega^2 * x + nu * (1 - x**2) * v
        fv = -x + self.mu * (1 - x**2) * v
        
        return np.array([fx, fv], float)

    def rk4_step(self, r, t, h):
        """
        Runge-Kutta 4 method to calculate the next state.
        r: current state array [x, v]
        t: current time
        h: time step size (delta t)
        """
        # Notice we don't need to pass omega and nu anymore, 
        # because self.func already knows them from __init__!
        k1 = h * self.func(r, t)
        k2 = h * self.func(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * self.func(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * self.func(r + k3, t + h)
        
        # Return the new updated state [x_new, v_new]
        return (k1 + 2*k2 + 2*k3 + k4) / 6.0


    def generate_trajectory(self, initial_condition, t_array):
        """
        Automatically extracts step size h, tracks state updates,
        and records the entire history to return to the user.
        """
        # Automatically calculate the step size h from your time array
        h = t_array[1] - t_array[0]
        
        # Initialize your state vector with the passed initial conditions
        r = np.array(initial_condition, dtype=float)
        
        # Create a list to accumulate the history at every step
        history = []
        
        for t in t_array:
            history.append(r.copy())         # 1. Record the current state [x, v]
            r += self.rk4_step(r, t, h)      # 2. Update r to the next time step
            
        return np.array(history)             # Convert to a NumPy array for easy slicing
    
    def plot_trajectory(self, initial_condition, t_array):

        trajectory_data = self.generate_trajectory(initial_condition, t_array)
        x_data = trajectory_data[:, 0]
        v_data = trajectory_data[:, 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 

        # Left Plot: Time Series
        ax1.plot(t_array, x_data, label="Position ($x$)", color="tab:blue", lw=2)
        ax1.set_title(f"Time Series ($\mu$={self.mu})")
        ax1.set_xlabel("Time ($t$)")
        ax1.set_ylabel("Displacement ($x$)")
        ax1.grid(True)
        # ax1.legend()

        # Right Plot: Phase Space Portrait
        ax2.plot(x_data, v_data, label="State Path", color="tab:orange", lw=2)
        ax2.scatter(initial_condition[0], initial_condition[1], color="red", zorder=5, label="Initial State")
        
        # --- ADDED: Explicit Origin Axis Crosshairs ---
        # Draws clean, solid horizontal and vertical lines exactly at zero
        ax2.axhline(0, color='black', linestyle='-', lw=1.0, alpha=0.6)
        ax2.axvline(0, color='black', linestyle='-', lw=1.0, alpha=0.6)
        
        ax2.set_title(f"Phase Space Portrait ($\mu$={self.mu})")
        ax2.set_xlabel("Position ($x$)")
        ax2.set_ylabel("Velocity ($v = \dot{x}$)")
        ax2.grid(True, linestyle=':', alpha=0.6) # Subtle background grid
        # ax2.legend()

        plt.tight_layout()
        plt.savefig('VDP_target.png')