
from logistic_map import LogisticMap
from train_model import train

def main ():

    lm = LogisticMap(alpha=3.99)  
    lm.run_trajectory("henon_map") 
    lm.plot_time_series("henon_map")
    return 0



if __name__ == "__main__":
    main()