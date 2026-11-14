import matplotlib.pyplot as plt


def plot_graph (t,t1,t2,x1_data,y1_data,x2_data,y2_data, w1, w2):

    plt.figure(figsize=(6,6))    
    plt.plot(t, t1, label=r"$\theta_1$", color="tab:blue", lw=1.5)
    plt.plot(t, t2, label=r"$\theta_2$", color="tab:orange", lw=1.5)
    plt.title(r"Angles Over Time ($\theta_1, \theta_2$)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.figure(figsize=(6,6))
    plt.plot(t1, t2, color="k", lw=1.2)
    plt.title(r"Configuration Space ($\theta_1$ vs $\theta_2$)")
    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.grid(True, linestyle="--", alpha=0.4)

    #------Phase Space------------
    plt.figure(figsize=(6,6))
    plt.plot(t1, w1, label="Pendulum 1", color="tab:blue", lw=0.8)
    plt.plot(x1_data[0], y1_data[0], color="tab:blue", zorder=5)
    plt.title("Double Pendulum Trajectory (x-y Plane)")
    plt.xlabel(f"$\\theta_{1}$")
    plt.ylabel(f"$\dot \\theta_{1}$")
    plt.grid(True, linestyle="--", alpha=0.4)


    #-----Pendulum 軌道------------
    # Spatial Trajectory (x-y Plane)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(x2_data, y2_data, label="Mass 2 path", color="tab:orange", lw=1.0)
    ax2.plot(x2_data[0], y2_data[0], marker="o", color="tab:red", zorder=5, label="Start")
    ax2.set_title("Mass 2 Trajectory (x-y Plane)")
    ax2.set_xlabel("$x$ position (m)")
    ax2.set_ylabel("$y$ position (m)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    

    plt.show()
    

