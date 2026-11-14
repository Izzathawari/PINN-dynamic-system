import math
import numpy as np
import matplotlib.pyplot as plt

class DoublePendulumLagrangian:

    def __init__(self, g, m1, m2, t1, t2, w1, w2, L1, L2):
        """
        Constructing a double pendulum simulator with the help of its
        Euler-Lagrange equations.
        Here,
        g = The gravitational acceleration.
        m1 = The mass of bob #1.
        m2 = The mass of bob #2.
        t1 = The initial angle of bob #1.
        t2 = The initial angle of bob #2.
        w1 = The initial angular velocity of bob #1.
        w2 = The initial angular velocity of bob #2.
        L1 = The length of the rod for bob #1.
        L2 = The length of the rod for bob #2.

        これを参考しながら作る
        https://github.com/Hridoy-31/SUSTPhysics-Computational-Physics-1/blob/master/Final%20Project/
        """

        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.t1 = t1
        self.t2 = t2
        self.w1 = w1
        self.w2 = w2
        self.L1 = L1
        self.L2 = L2


    def potential_energy (self):
        """
        Potential Energy Part of Langragrian Eq.
        """

        m1 = self.m1
        m2 = self.m2
        L1 = self.L1
        L2 = self.L2
        t1 = self.t1
        t2 = self.t2

        g = self.g

        #式を切り分けて，見やすいように
        y1 = -L1 * math.cos(t1)
        y2 = y1 - L2 * math.cos(t2)

        return m1 * g * y1 + m2 * g * y2


    def kinetic_energy (self):
        """
        Kinetic Energy Part of Langragrian Eq.
        """

        m1 = self.m1
        t1 = self.t1
        w1 = self.w1
        L1 = self.L1
        m2 = self.m2
        t2 = self.t2
        w2 = self.w2
        L2 = self.L2

        #the kinetic energy for each bob
        K1 = 0.5 * m1 * (L1 * w1)**2
        K2 = 0.5 * m2 * ((L1 * w1)**2 + (L2 * w2)**2 + 2 * L1 * L2 * w1 * w2 * math.cos(t1 - t2))

        return K1 + K2


    def mechanical_energy (self):

        return self.kinetic_energy() + self.potential_energy()

    def langrange_eq (self, t1, t2, w1, w2):

        m1 = self.m1
        L1 = self.L1
        m2 = self.m2
        L2 = self.L2

        g = self.g

        a1 = (L2 / L1) * (m2 / (m1 + m2)) * math.cos(t1 - t2)
        a2 = (L1 / L2) * math.cos(t1 - t2)

        f1 = -(L2 / L1) * (m2 / (m1 + m2)) * (w2**2) * math.sin(t1 - t2) - \
            (g / L1) * math.sin(t1)
        
        f2 = (L1 / L2) * (w1**2) * math.sin(t1 - t2) - (g / L2) * math.sin(t2)

        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        return np.array([w1, w2, g1, g2])


    def rungge_kutta_step(self, dt):
        """
        Advances one time step using fourth order classical Runge-Kutta
        method. 

        Argument:
        dt = time step

        Output:
        A vetor of [theta 1, omega 1, theta 2, omega 2]
        """
        t1 = self.t1
        w1 = self.w1
        t2 = self.t2
        w2 = self.w2

        #array with the generalized coordinates (angles +
        # angular velocities)
        y = np.array([t1, t2, w1, w2])

        #computing the RK4 constants
        k1 = self.langrange_eq(*y)
        k2 = self.langrange_eq(*(y + dt * k1 / 2))
        k3 = self.langrange_eq(*(y + dt * k2 / 2))
        k4 = self.langrange_eq(*(y + dt * k3))

        #computing the RK4 right-hand side
        R = 1.0 / 6.0 * dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        #finally updating the angles and angular velocities
        self.t1 += R[0]
        self.t2 += R[1]
        self.w1 += R[2]
        self.w2 += R[3]

class SinglePendulum:
    def __init__(self, m, L, init_theta):
        self.m = m
        self.L = L
        self.init_theta = init_theta

    def solution(self, time):
        g = 9.8
        return self.init_theta * np.cos(np.sqrt(g / self.L) * time)



def main():
    single = SinglePendulum(m=1, L=1.0, init_theta=0.2)  # ~23 degrees
    time = np.linspace(0, 10, 500)

    # 1. Calculate angle theta(t)
    theta = single.solution(time)
    x = single.L * np.sin(theta)
    y = -single.L * np.cos(theta)

    # 3. Plot the 2D trajectory
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="tab:blue", lw=2.5, label="Mass trajectory")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-single.L * 1.1, single.L * 1.1)
    ax.set_ylim(-single.L * 1.2, 0.2)

    ax.set_title("Single Pendulum Mass Trajectory (x-y Plane)")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.show()


if __name__ == "__main__":
    main()