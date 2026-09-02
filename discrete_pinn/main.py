from logistic_map import LogisticMap



def main ():

    logistic_eq = LogisticMap(alpha=3.99)
    plot_bifurcation = logistic_eq.plot_bifurcation()
    run_map = logistic_eq.plot_time_series()
    return 0



if __name__ == "__main__":
    main()