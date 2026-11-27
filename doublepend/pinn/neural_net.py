import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

class PhysicsInformedNN (nn.Module):

    def __init__(self, config, verbose):
        super().__init__()

        # build the neural network
        self.build_network(config, verbose)  
          
        # create data_loader instance
        self.data_loader = DataLoader(config)  
        
        # create loss instance
        # self.loss = Loss(self, config)
        
        # create callback instance
        # self.callback = CustomCallback(config, self.neural_net)   
        # self.save_data = config.get('save_data',False)

        # preprocessing setting
        self.norm = config.get('norm_flag',True)
        self.x_domain = config['x_domain']
        self.preprocessing_domain = config['preprocessing_domain']
        
        # parameters for training
        self.n_epochs = config['n_epochs']
        self.learning_rate = config['learning_rate']
        self.decay_rate = config['decay_rate']
        self.lambda_IC = config['lambda_IC']

        print('*** PINN build & initialized ***')


    def build_network (self, config, verbose):

        torch.manual_seed(config['seed_pinn'])

        #layer setting
        in_dim = config['n_input']
        n_hidden = config['n_hidden']     
        n_neurons = config['n_neurons']      
        n_output = config['n_output']

        activations = {
            "tanh": nn.Tanh,
            "swish": nn.SiLU,
            "silu": nn.SiLU,
            "sigmoid": nn.Sigmoid,
        }
        act_name = config["activation"].lower()
        act_cls = activations[act_name]

        layers = []

        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(act_cls())
            in_dim = n_neurons

        layers.append(nn.Linear(in_dim, n_output))

        self.net = nn.Sequential(*layers)

        if verbose:
            print(self.net)


    # --------------------------------------------------------------------------
    # Normalization of input vector
    def normalization(self, X):
      print('Feature scaling performed')
      X_norm = (X - self.x_domain[0]) / (self.x_domain[1] - self.x_domain[0])
      X_norm = X_norm * (self.preprocessing_domain[1] - self.preprocessing_domain[0]) + self.preprocessing_domain[0]
      return X_norm

    # --------------------------------------------------------------------------
    # Call neural network with normalized input
    def call (self, X):
        if self.norm == True:
            X_norm = self.normalization(X)

        else:
            X_norm = X

        return self.net(X_norm)

    # --------------------------------------------------------------------------
    # Train neural network
    def train(self):
        # learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate)

        #optimizer
        self.optimizer = optim.Adam(lr= )

        print("Training started...")
        for epoch in range(self.n_epochs):

            time_input = self.data_loader


if __name__ == "__main__":

    from utils import rk_solve
    import numpy as np
    from configs.config_loader import load_config, print_config
    from pinn import neural_net

    config_file = "configs/default.yaml"
    config = load_config(config_file)

    L1,L2,m1,m2,g = config["l1"], config["l2"], config["m1"], config["m2"], config["g"]
    theta1, theta2, w1, w2 = config["y0"]
    t_eval = [0,5]

    y0 = [theta1, theta2, w1, w2]
    t_min, t_max = config["x_domain"]
    t_eval = np.linspace(t_min, t_max, 1000)

    doublepend = rk_solve.DoublePendulumLagrangian(g,m1,m2,theta1,theta2, w1, w2, L1, L2)
    t_domain, state_true = doublepend.exact_rk (y0, t_eval)

    print(f"Double Pendulum state shape {state_true.shape}")
    print(f"Time Step shape {t_eval.shape}")


    model = neural_net.PhysicsInformedNN(config, verbose=True)
    t_domain = model.normalization(t_domain)
    print(f"Time domain : {t_domain[0]} , {t_domain[-1]}")

    plt.plot(t_domain)
    plt.show()

    
