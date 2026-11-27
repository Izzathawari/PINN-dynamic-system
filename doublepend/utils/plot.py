import matplotlib.pyplot as plt
import numpy as np


def wrap_to_pi(angles):
    """Wraps angles into [-pi, pi] for plotting against Steger et al. Fig. 1."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def plot_rk (timestep, y) :

    timestep = np.asarray(timestep).squeeze()
    y = np.asarray(y)
    y = wrap_to_pi(y)

    # Single 1D curve (e.g., state_true[:, 0])
    if y.ndim == 1:
        plt.plot(timestep, y, c="tab:cyan", label="RK solution")
        plt.xlabel("Time $t$ [s]")
        plt.ylabel("Angle [rad]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        return

    # Full state array (N, 4) -> plot theta1 and theta2 in two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(timestep, y[:, 0], c="tab:cyan", label=r"$\theta_1$ (RK45)")
    axes[0].set_ylabel(r"$\theta_1$ [rad]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timestep, y[:, 1], c="tab:blue", label=r"$\theta_2$ (RK45)")
    axes[1].set_xlabel("Time $t$ [s]")
    axes[1].set_ylabel(r"$\theta_2$ [rad]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()