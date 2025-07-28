from scipy import signal
import numpy as np
import control as ctrl
from matplotlib import pyplot as plt


class DynamicalSystem:
    """Dynamical System whos order is defined by number of poles

    This class implements a dynamical system wth variable simulation timesteps.
    It keepts it simulation history and continues the simulation for every call of the
    simulate method and reacts according to the command value given in the function call.
    It returens the inner states as well as the output.

    Attributes:
        poles:
            Complex values with negative real part that define the dynamical behavior the number of poles defines
                the system order
        zeros:
            Zeros of the transferfunction.
        dt:
            Time in [s] between two datapoints of the simulation
        sys:
            Dynamical system.
        val_t:
            Timeline of the simulation.
        val_y:
            Simulated values of the output.
        val_x:
            Simulated values of the internal states.


    """

    def __init__(self, poles, zeros, dt=0.01):
        """
        Creates the system from the poles and zeros and inits the simulation
        Args:
            poles:
                Complex values with negative real part that define the dynamical behavior the number of poles defines
                the system order
            zeros:
                Zeros of the transferfunction.
            dt:
                Time in [s] between two datapoints of the simulation
        """

        # get variables
        self.poles = poles  # klein schreiben
        self.zeros = zeros
        self.dt = dt

        # create system
        self.sys = self.create_system()
        # init simulation

        self.init_sim()

    def init_sim(self):
        """Initialize simulation """
        # initialise simulation
        val_t, val_y, val_x = ctrl.initial_response(self.sys, return_x=True, transpose=True)

        # only get the first value from the simulation
        self.val_t = val_t[0:1]
        self.val_y = val_y[0:1]
        self.val_x = val_x[0:1, :]

    def create_system(self):
        """This Method creates a state space system (A,B,C,D Matrices) of the poles contained in the
        'self' parameter and returns a system of type control.StateSpace.
        """
        # define continous state space by zero pole gain repesentation
        ss_model = signal.zpk2ss(self.zeros, self.poles, 1)
        # get A,B,C,D Matrices
        A = ss_model[0]
        B = ss_model[1]
        C = ss_model[2]
        D = ss_model[3]

        # Transfer Model into controls Toolbox
        sys = ctrl.ss(A, B, C, D)

        # Calculate gain necessary for stationary accuracy (val_y=u for t=inf)
        gain = 1 / sys.dcgain()
        C = C * gain  # Add gain to Output matrix

        # redefine System in controls toolbox and show system characteristics
        sys = ctrl.ss(A, B, C, D)

        return sys

    def show_behavior(self):
        """This Method gives back values and plots that characterize the system"""

        # print step response
        t, y = ctrl.step_response(self.sys)
        plt.title("Stepresponse of System")
        plt.plot(t, y)
        plt.show()

        # print info about poles
        ctrl.pzmap(self.sys, Plot=True, grid=True, title='Pole Zero Map')
        print("Information about poles and zeros of the transferfunction:")
        ctrl.damp(self.sys, doprint=True)

    def simulate(self, t_sim=1, u=0):
        """This Method continues simulation of the model for time t_sim

        Args:
            t_sim:
                time in seconds that the simulation should run.
            u:
                command value for the system during simulation.
        Returns:
            val_t:
                Timeline of the simulation.
            val_y:
                Simulated values of the output.
            val_x:
                Simulated values of the internal states.
            y_dot:
                First order derivation of the last datapoint.
        """

        if self.dt > t_sim:
            t_sim = self.dt

        # prepare Time axis
        steps = np.rint(t_sim / self.dt).astype('i') + 1
        t = np.linspace(self.val_t[-1], self.val_t[-1] + t_sim, steps, endpoint=True)

        # get last state vector of previous step to initialize the system for the following simulation step
        x0 = self.val_x[-1, :]

        # generate constant input vector for simulation step
        u_in = np.ones_like(t) * u

        # simulate with ODE solver
        t_step, y_step, x_step = ctrl.forced_response(self.sys, X0=x0, U=u_in, T=t, transpose=True, return_x=True)

        # save to output
        self.val_x = np.append(self.val_x, x_step[1:, :], axis=0)
        self.val_y = np.append(self.val_y, y_step[1:], axis=0)
        self.val_t = np.append(self.val_t, t_step[1:], axis=0)
        # calculate first oder derivate of val_y for the last simulated timestep
        self.y_dot = (self.val_y[-1] - self.val_y[-2]) / self.dt

        return self.val_t, self.val_y, self.val_x, self.y_dot

    def reset_simulation(self):
        """
        This Method resets the model and the outputs
        """
        # recreate system
        self.sys = self.create_system()
        # init simulation
        self.init_sim()

    def redef_poles(self, poles, zeros):
        """
        This Method redefines the poles of the System
        Args:
            poles:
                Complex values with negative real part that define the dynamical behavior the number of poles defines
                the system order
            zeros:
                Zeros of the transferfunction.
        """
        # change values for poles
        self.poles = poles
        self.zeros = zeros
        # recreate system
        self.sys = self.create_system()
