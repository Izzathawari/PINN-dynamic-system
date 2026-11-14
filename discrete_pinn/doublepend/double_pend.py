#!/usr/bin/env python3
import math
from plot import plot_graph

def main():
    t, x_data_1, x_data_2, y_data_1, y_data_2, t1_data, t2_data, K_data, P_data, E_data  = [], [], [], [], [], [], [], [], [], []
    # default simulation parameter values
    g = 9.8
    dt = 0.025
    #m1 = float(input("Mass 1 : "))
    m1 = 2
    m2 = 2
    t1 = t2 = 0.5
    w1 = w2 = 0
    L1 = L2 = 1.0

    # initialize the double pendulum
    from dp_lagrangian import DoublePendulumLagrangian
    obj = DoublePendulumLagrangian(g, m1, m2, t1, t2, w1, w2, L1, L2)
    
        
    step = 0
    index = 0
    while (step < 1000):

    
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

    plot_graph(t,t1_data,t2_data)

    


if __name__ == "__main__":
    main()