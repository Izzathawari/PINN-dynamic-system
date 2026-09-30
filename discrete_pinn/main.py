
from data_creation import MapFunction
from train_model import train

def main ():

    
    map_function = MapFunction()

    Henon_map = map_function.run_trajectory("henon_map")
    map_function.convert_data2pd()



if __name__ == "__main__":
    main()