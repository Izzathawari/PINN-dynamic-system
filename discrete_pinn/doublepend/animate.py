import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dp_lagrangian import DoublePendulumLagrangian


def animate_diff():
    # Simulation parameters
    g = 9.8
    dt = 0.025
    steps = 800  # 800 * 0.025s = 20 seconds
    m1 = 2
    m2 = 2
    L1 = 1.0
    L2 = 1.0

    # Use t1 = 1.5 so the chaotic divergence becomes clearly visible mid-animation
    # (With t1 = 0.0, the 0.001 rad difference stays too small to see visually)
    t1_init = 1.5
    w1 = w2 = 0.0

    obj_1 = DoublePendulumLagrangian(g, m1, m2, t1_init, 1.500, w1, w2, L1, L2)
    obj_2 = DoublePendulumLagrangian(g, m1, m2, t1_init, 1.499, w1, w2, L1, L2)

    # Pre-compute trajectories for smooth animation
    t = []
    x1_1, y1_1, x2_1, y2_1 = [], [], [], []
    x1_2, y1_2, x2_2, y2_2 = [], [], [], []

    for step in range(steps):
        t.append(step * dt)

        # Pendulum 1 coordinates
        px1 = L1 * math.sin(obj_1.t1)
        py1 = -L1 * math.cos(obj_1.t1)
        x1_1.append(px1)
        y1_1.append(py1)
        x2_1.append(px1 + L2 * math.sin(obj_1.t2))
        y2_1.append(py1 - L2 * math.cos(obj_1.t2))

        # Pendulum 2 coordinates
        qx1 = L1 * math.sin(obj_2.t1)
        qy1 = -L1 * math.cos(obj_2.t1)
        x1_2.append(qx1)
        y1_2.append(qy1)
        x2_2.append(qx1 + L2 * math.sin(obj_2.t2))
        y2_2.append(qy1 - L2 * math.cos(obj_2.t2))

        obj_1.rungge_kutta_step(dt)
        obj_2.rungge_kutta_step(dt)

    # --- Set up the figure ---
    fig, ax = plt.subplots(figsize=(7, 7))
    limit = (L1 + L2) * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$ position (m)")
    ax.set_ylabel("$y$ position (m)")
    ax.set_title(r"Double Pendulum Divergence ($\theta_2(0) = 1.500$ vs $1.499$)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Fixed pivot
    ax.plot(0, 0, marker="o", markersize=8, color="black", zorder=6)

    # Trajectory trails for Mass 2
    trail_1, = ax.plot([], [], color="tab:blue", lw=1.0, alpha=0.55)
    trail_2, = ax.plot([], [], color="tab:orange", lw=1.0, alpha=0.55)

    # Rods and bobs for both pendulums
    rod_1, = ax.plot([], [], "o-", color="tab:blue", lw=2.2, markersize=7,
                     label=r"Case 1 ($\theta_2(0) = 1.500$)")
    rod_2, = ax.plot([], [], "o-", color="tab:orange", lw=2.2, markersize=7,
                     label=r"Case 2 ($\theta_2(0) = 1.499$)")

    time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper right")

    def init():
        rod_1.set_data([], [])
        rod_2.set_data([], [])
        trail_1.set_data([], [])
        trail_2.set_data([], [])
        time_text.set_text("")
        return rod_1, rod_2, trail_1, trail_2, time_text

    def update(frame):
        # Update rods + bobs: [pivot, mass1, mass2]
        rod_1.set_data([0, x1_1[frame], x2_1[frame]], [0, y1_1[frame], y2_1[frame]])
        rod_2.set_data([0, x1_2[frame], x2_2[frame]], [0, y1_2[frame], y2_2[frame]])

        # Update cumulative Mass 2 trajectories
        trail_1.set_data(x2_1[:frame + 1], y2_1[:frame + 1])
        trail_2.set_data(x2_2[:frame + 1], y2_2[:frame + 1])

        time_text.set_text(f"Time: {t[frame]:.2f} s")
        return rod_1, rod_2, trail_1, trail_2, time_text

    # Step by 2 frames for faster playback and compact GIF export
    frame_indices = range(0, steps, 2)
    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        interval=dt * 2 * 1000,  # Real-time speed (50 ms per 2 steps)
        blit=True,
    )

    # Save as GIF so you can view it directly in Windows/VS Code even if WSL blocks popups
    print("Saving animation to double_pendulum_diff.gif...")
    ani.save("double_pendulum_diff.gif", writer=PillowWriter(fps=20))
    print("Saved!")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    animate_diff()