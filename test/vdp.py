import numpy as np

class VDP:
    def __init__(self, omega, nu):
        """
        Initialize the Van der Pol oscillator with its physical parameters.
        omega: Natural frequency of the system
        nu: Non-linear damping coefficient (often written as epsilon)
        """
        self.omega = omega
        self.nu = nu

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
        fv = -self.omega**2 * x + self.nu * (1 - x**2) * v
        
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

    