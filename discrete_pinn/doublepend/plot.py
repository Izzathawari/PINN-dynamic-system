import matplotlib.pyplot as plt


def plot_graph (t,t1,t2):

    fig,ax = plt.subplots(2,1, figsize=(9, 6))
    ax[0].plot(t,t1, label=f"$\\theta_{1}$", color="k", lw=1.5)
    ax[0].plot(t,t2,label=f"$\\theta_{2}$", color="k", lw=1.5)
    ax[0].set_title(f"$\\theta$")
    ax[0].set_ylabel("Time Step")
    ax[0].grid(True, linestyle="--", alpha=0.1)

    ax[1].plot(t1,t2, label=f"$\\theta_{1}$", color="k", lw=1.5)
    ax[1].set_title(f"$\\theta$")
    ax[1].set_ylabel("Time Step")
    ax[1].grid(True, linestyle="--", alpha=0.1)
    

    plt.show ()
