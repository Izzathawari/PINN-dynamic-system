from scipy.integrate import solve_ivp 
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS FOR OBTAINING THE REFERENCE SOLUTION USING TRADITIONAL ODE SOLVERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        y1 = -L1 * np.cos(t1)
        y2 = y1 - L2 * np.cos(t2)

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
        K2 = 0.5 * m2 * ((L1 * w1)**2 + (L2 * w2)**2 + 2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2))

        return K1 + K2


    def mechanical_energy (self):

        return self.kinetic_energy() + self.potential_energy()

    def langrange_eq (self, t1, t2, w1, w2):

        m1 = self.m1
        L1 = self.L1
        m2 = self.m2
        L2 = self.L2

        g = self.g

        a1 = (L2 / L1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
        a2 = (L1 / L2) * np.cos(t1 - t2)

        f1 = -(L2 / L1) * (m2 / (m1 + m2)) * (w2**2) * np.sin(t1 - t2) - \
            (g / L1) * np.sin(t1)
        
        f2 = (L1 / L2) * (w1**2) * np.sin(t1 - t2) - (g / L2) * np.sin(t2)

        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        return np.array([w1, w2, g1, g2])

    
    def exact_rk (self, t_eval, method = "RK45"):
        """
        Numerically solve the double pendulum ODE.

        Parameters
        -----------
        t_domain
            Time domain for simulation

        L1, L2, m1, m2, g: System parameters
            Pendulum rod mass (m1,m2), length (L1, L2), gravitational constant

        Returns
        -------
        X_true: list
            Reference solution of the system vector X = [theta_1, theta_2, omega_1, omega_2]
        """

        sol = solve_ivp(
                    self._ode_rhs,
                    t_span=[t_eval.min(), t_eval.max()],
                    y0=y0,
                    method=method,
                    t_eval=t_eval,
                    rtol=1e-10,
                    atol=1e-12,
                )
