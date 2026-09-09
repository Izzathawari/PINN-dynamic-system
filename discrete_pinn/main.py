
from logistic_map import LogisticMap
from train_model import train

def main ():

    lm = LogisticMap(alpha=3.99)
    lm.run_trajectory()
    lm.plot_time_series()
    lm.make_data_splits()
    lm.convert_data2pd()
    
    return 0



if __name__ == "__main__":
    main()