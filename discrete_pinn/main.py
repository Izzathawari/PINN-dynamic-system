
from data_creation import MapFunction
from train_model import train
from model import HenonPINN

def main ():

    
    # map_function = MapFunction()

    # Henon_map, timestep = map_function.run_trajectory("henon_map")
    # (x_curr_train, x_next_train), (x_curr_test, x_next_test) = map_function.make_data_splits()

    # train_x_curr = x_curr_train[:, 0:1]
    # train_x_next = x_next_train[:, 0:1]

    # test_x_curr = x_curr_test[:, 0:1]
    # test_x_next = x_next_test[:, 0:1]

    # print(x_curr_train)

    #--------------------------------------#
    train("henon_map")


if __name__ == "__main__":
    main()