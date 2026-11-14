#!/usr/bin/env python3
import math
from plot import plot_graph
from dp_lagrangian import DoublePendulumLagrangian
import matplotlib.pyplot as plt

def diff():
    t = []
    x2_data_1, y2_data_1 = [], []
    x2_data_2, y2_data_2 = [], []
    t1_data_1, t2_data_1 = [], []
    t1_data_2, t2_data_2 = [], []
    diff_t1, diff_t2 = [], []

    # Simulation parameters
    g = 9.8
    dt = 0.025
    m1 = 2
    m2 = 2
    t1 = 0.0
    w1 = w2 = 0
    L1 = 1
    L2 = 1

    # Two pendulums with a 0.001 rad initial difference in theta_2
    obj_1 = DoublePendulumLagrangian(g, m1, m2, t1, 1.5, w1, w2, L1, L2)
    obj_2 = DoublePendulumLagrangian(g, m1, m2, t1, 1.499, w1, w2, L1, L2)

    step = 0
    while step < 1000:
        # Positions for Pendulum 1
        x1_1 = L1 * math.sin(obj_1.t1)
        y1_1 = -L1 * math.cos(obj_1.t1)
        x2_1 = x1_1 + L2 * math.sin(obj_1.t2)
        y2_1 = y1_1 - L2 * math.cos(obj_1.t2)

        # Positions for Pendulum 2
        x1_2 = L1 * math.sin(obj_2.t1)
        y1_2 = -L1 * math.cos(obj_2.t1)
        x2_2 = x1_2 + L2 * math.sin(obj_2.t2)
        y2_2 = y1_2 - L2 * math.cos(obj_2.t2)

        # Record time, positions, angles, and differences
        t.append(step * dt)
        x2_data_1.append(x2_1)
        y2_data_1.append(y2_1)
        x2_data_2.append(x2_2)
        y2_data_2.append(y2_2)

        t1_data_1.append(obj_1.t1)
        t2_data_1.append(obj_1.t2)
        t1_data_2.append(obj_2.t1)
        t2_data_2.append(obj_2.t2)

        diff_t1.append(abs(obj_1.t1 - obj_2.t1))
        diff_t2.append(abs(obj_1.t2 - obj_2.t2))

        # Advance BOTH pendulums one time step
        obj_1.rungge_kutta_step(dt)
        obj_2.rungge_kutta_step(dt)
        step += 1

    # --- Plot 1: Angle Comparison & Divergence ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(t, t2_data_1, label=r"Pendulum 1 ($\theta_2(0) = 1.500$)", color="tab:blue", lw=1.5)
    ax[0].plot(t, t2_data_2, label=r"Pendulum 2 ($\theta_2(0) = 1.499$)", color="tab:orange", linestyle="--", lw=1.5)
    ax[0].set_title(r"Comparison of $\theta_2(t)$ with $0.001\text{ rad}$ Initial Difference")
    ax[0].set_ylabel(r"$\theta_2$ (rad)")
    ax[0].legend()
    ax[0].grid(True, linestyle="--", alpha=0.4)

    ax[1].plot(t, diff_t1, label=r"$|\Delta \theta_1(t)|$", color="tab:green", lw=1.5)
    ax[1].plot(t, diff_t2, label=r"$|\Delta \theta_2(t)|$", color="tab:red", lw=1.5)
    ax[1].set_title("Angular Divergence Over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle Difference (rad)")
    ax[1].legend()
    ax[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("theta_difference.png", dpi=150)

    # --- Plot 2: Mass 2 Trajectory Comparison ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(x2_data_1, y2_data_1, label=r"Mass 2 ($\theta_2(0)=1.500$)", color="tab:blue", lw=1.0)
    ax2.plot(x2_data_2, y2_data_2, label=r"Mass 2 ($\theta_2(0)=1.499$)", color="tab:orange", lw=1.0, alpha=0.8)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Mass 2 Trajectory Comparison (x-y Plane)")
    ax2.set_xlabel("$x$ position (m)")
    ax2.set_ylabel("$y$ position (m)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
    


def main():
    t, x_data_1, x_data_2, y_data_1, y_data_2, w1_data, w2_data= [], [], [], [], [], [], []
    t1_data, t2_data, K_data, P_data, E_data  =  [], [], [], [], []
    # default simulation parameter values
    g = 9.8
    dt = 0.025
    #m1 = float(input("Mass 1 : "))
    m1 = 2
    m2 = 2
    t1 = 0.0
    t2 = 1.5 # 0.9 = 51 deg, 0.2 = 11 deg, 1.5 = 86 deg
    w1 = w2 = 0
    L1 = 1
    L2 = 1

    # initialize the double pendulum
    from dp_lagrangian import DoublePendulumLagrangian
    obj = DoublePendulumLagrangian(g, m1, m2, t1, t2, w1, w2, L1, L2)
    
        
    step = 0
    index = 0
    while (step < 5000):

    
        y1 = -L1 * math.cos(obj.t1)
        y2 = y1 - L2 * math.cos(obj.t2)
        
        x1 = L1 * math.sin(obj.t1)
        x2 = x1 + L2 * math.sin(obj.t2)
        
        
        t.append(index*dt)
        index += 1
        x_data_1.append(x1)
        x_data_2.append(x2)
        y_data_1.append(y1)
        y_data_2.append(y2)
        t1_data.append(obj.t1)
        t2_data.append(obj.t2)
        w1_data.append(obj.w1)
        w2_data.append(obj.w2)
        K_data.append(obj.kinetic_energy())
        P_data.append(obj.potential_energy())
        E_data.append(obj.mechanical_energy())
        
        
        print(x1,x2)
        print(y1, y2)
        print('-----------------')
        # print(,obj.t2)
        
        # advance one time step
        obj.rungge_kutta_step(dt)
        step += 1

    plot_graph(t, t1_data, t2_data,
               x_data_1, y_data_1, 
               x_data_2, y_data_2,
               w1_data, w2_data)

    


if __name__ == "__main__":
    diff()