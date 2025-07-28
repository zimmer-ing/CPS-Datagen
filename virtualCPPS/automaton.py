import sys
import numpy as np
from virtualCPPS.dynamical_system import DynamicalSystem
from virtualCPPS.dataclasses import TimeSeries, Errors
import math
import pandas as pd


class Automaton:
    """
    Automation that generates a manifold in a x dimensional statespace.

    Attributes:
        act_state:
            Array that contains the actual state for the simulated data points
        cycle:
            Array that contains the actual cycle of the simulated data points
        Ts_Data:
            Object of class TimeSeries that contains results.
        dynamic_error:
            Flag for dynamic errors (default= false).
        states:
            List of target values for the states the automaton can take
        state_names:
            List of names for the states (numbers). Needed to have a unique allocation in case
            states are added or removed.
    """

    highest_val_state = 100

    def __init__(self, system: DynamicalSystem, num_states, noise=5, random_seed: int = 42, distance_to_targetstate=10,
                 gradient_at_target=5,cycletime_controller=0.1):
        """
        Args:
            system:
                System of class 'Dynamical_System' that is able to simulate its behavior
            num_states:
                Number of states of the automaton
            noise:
                Noise [%] that is added to the states between the cycles
            random_seed:
                Optional parameter to ensure reproducibility
            distance_to_targetstate:
                Allowed distance in [%] +- from target state that leads to switching to the next state.
            gradient_at_target:
                Allowed derivate of y within the target area (defined by :param distance_to_targetstate)
            cycletime_controller:
                Cycletime that is used to check the condition for a statechange.
        """

        # save inputs
        self.num_states = num_states
        self.random_seed: int = random_seed
        self.system = system
        self.distance_to_targetstate = distance_to_targetstate
        self.gradient_at_target = gradient_at_target
        self.noise = noise
        self.cycletime_controller=cycletime_controller
        # Set random seed
        np.random.seed(self.random_seed)

        # init empty arrays & vars
        self.act_state = np.array([])
        self.cycle = np.array([])
        self.Ts_Data = TimeSeries()
        self.dynamic_error = False

        # generate random states
        arr = np.arange(Automaton.highest_val_state + 1)
        np.random.shuffle(arr)
        np.random.shuffle(arr)  # shuffle twice
        self.states = arr[0:abs(num_states)]

        self.state_names = np.arange(1, self.num_states + 1)

    def redefine_random_seed(self, new_seed):
        self.random_seed = new_seed
        np.random.seed(self.random_seed)

    def simulate_single_state(self, target, name):
        """
        This method simulates change of the system from the current system state to the target system state

        Args:
            name:
                Name of the state:
            target:
                Target state that should be reached from the current system state.
        """

        self.T, self.y, self.x, self.y_dot = self.system.simulate(u=target,t_sim=self.cycletime_controller)
        # simulate system as long as while holds true
        while self.y[-1] < (target - self.distance_to_targetstate) \
                or self.y[-1] > (target + self.distance_to_targetstate) \
                or (np.abs(self.y_dot) > self.gradient_at_target):
                self.T, self.y, self.x, self.y_dot = self.system.simulate(u=target,t_sim=self.cycletime_controller)

        self.act_state = np.append(self.act_state,
                                   np.ones(self.T.size - self.act_state.size) * name
                                   )

    def simulate_states(self, state_vector, name_vector):
        """
        This method simulates the system behavior over all states in the statevector

        Args:
            state_vector:
                Vector that contains the states of the automaton
            name_vector:
                Vector that contains names of the states
        """
        for target, name in zip(state_vector, name_vector):
            self.simulate_single_state(target, name)

    def add_noise_to_states(self):
        """Applies Noise to the statevector that contains the targets of the automaton"""
        # add noise to the states
        noisy_states = self.states + self.states * np.random.randn(self.states.size) * self.noise / 200
        return noisy_states

    def simulate(self, cycles):
        """
        This method simulates the behavior of the system/automaton

        Args:
            cycles:
                Cycles (processing of all states) that the automaton should process.

        """

        # reset the simulation and initialize
        self.system.reset_simulation()
        if self.dynamic_error == True:
            list_errors = self.calc_degeneration_rate(cycles)

        # loop that processes cycles of the system (1 cycle = processing whole statevector)
        for cnt in range(0, cycles):
            if self.dynamic_error == True:
                self.apply_all_errors(list_errors[cnt])
                df = pd.DataFrame([{'Noise': list_errors[cnt].noise,
                                    'add_remove_state': list_errors[cnt].add_remove_state,
                                    'offset_states': list_errors[cnt].offset_states,
                                    'poles_shift': list_errors[cnt].poles_shift
                                    }])
                self.Ts_Data.applied_errors = pd.concat([self.Ts_Data.applied_errors, df], ignore_index=True)

            noisy_states = self.add_noise_to_states()
            self.simulate_states(noisy_states, self.state_names)
            self.cycle = np.append(self.cycle, np.ones(self.T.size - self.cycle.size) * (cnt + 1))

        self.Ts_Data.Time = self.T
        self.Ts_Data.y = self.y
        self.Ts_Data.x = self.x
        self.Ts_Data.cycle = self.cycle
        self.Ts_Data.statename = self.act_state

        self.dynamic_error = False
        return self.Ts_Data

    def apply_all_errors(self, errors: Errors):
        """Collection function that applies all available errors using the error object.

        Args:
            errors: Object that contains parameters for error that should be applied
        """
        self.add_offset_states(errors.offset_states)
        self.add_remove_states(errors.add_remove_state)
        self.poles_shift(errors.poles_shift)
        self.increase_noise(errors.noise)

    def add_remove_states(self, num_states_change: int):
        """
        This method removes or adds states at random indexes of the state vector.

        If states should be added, it is checked weather the state that is inserted equals the neighbor states.
        In case it equals, a new random integer is sampled.

        Args:
            num_states_change:
                positive (add) or negative (remove) integer that defines the number of states to be added or removed.
        """

        change_index = -1  # return 0 if no index was changed

        # generate random indexes for removing / adding states
        # make sure that no index is doubled
        arr = np.arange(self.num_states)
        np.random.shuffle(arr)
        num_states_change = int(num_states_change)
        change_index = arr[0:abs(num_states_change)]

        # add states
        if num_states_change > 0:
            counter = 0

            for index in change_index:
                counter = counter + 1
                # case index = first element
                if index == 0:
                    val = np.random.randint(1, Automaton.highest_val_state + 1, size=1)
                    while (self.states[index] == val):
                        val = np.random.randint(1, Automaton.highest_val_state + 1, size=1)
                    self.states = np.insert(self.states, index, val)
                    self.state_names = np.insert(self.state_names, index, (self.num_states + counter))

                # case others
                else:
                    val = np.random.randint(1, Automaton.highest_val_state + 1, size=1)
                    while (self.states[index - 1] == val) or (self.states[index] == val):
                        val = np.random.randint(1, Automaton.highest_val_state + 1, size=1)
                    self.states = np.insert(self.states, index, val)
                    self.state_names = np.insert(self.state_names, index, (self.num_states + counter))

        # remove states
        if num_states_change < 0:
            if abs(num_states_change) >= self.num_states:
                print(
                    "Error: Number of states that should be removed higher or equal than the number of states present")
                sys.exit(-1)
            self.states = np.delete(self.states, change_index)
            self.state_names = np.delete(self.state_names, change_index)
            self.num_states = self.states.size
        return change_index + 1

    def add_offset_states(self, offset):
        """
        This method adds a bias to the state vector.

        Args:
            offset: Bias that is added to the states.
        """
        self.states = self.states + offset

    def poles_shift(self, shift):
        """This Method shifts the real part of the poles

        Args:
            shift:
                Factor that defines how much the poles are in or decreased (0.5 results in Re= 1.5 times faster poles).
        """
        poles = self.system.poles

        for i, p in enumerate(poles):
            real = p.real + p.real * shift
            # avoid instability by non negative real part
            if real > -0.01:
                real = -0.01
            poles[i] = (real) + p.imag * 1j
        self.system.poles = poles
        self.system.create_system()

    def increase_noise(self, noise):
        """This Method is increasing the noise for the states

        Args:
            noise:
                Amount of noise that should be added to the objects noise.
        """
        self.noise = self.noise + noise

    def init_dynamic_error(self, noise, add_remove_state, offset_states, poles_shift):
        """Function that initializes the dynamical application of errors

        Args:
            noise:
                Noise that is applied at the last cycle of the dynamic increase of the error
            add_remove_state:
                Count of states that should be removed or added at the end of the dynamic increase of the error.
            offset_states:
                Total offset that should be added to the states at the end of the dynamic increase of the error.
            poles_shift:
                Total pole shift that should be added to the states at the end of the dynamic increase of the error.

        """
        self.dynamic_error = True
        self.dyn_error_data = Errors(noise=noise,
                                     add_remove_state=add_remove_state,
                                     offset_states=offset_states,
                                     poles_shift=poles_shift
                                     )

    def calc_degeneration_rate(self, cycles):
        """Subfunction that calculates the rate the errors are increased every cycle

        Args:
            cycles:
             Number of cycles that the automaton should perform.

        Returns:
            List of objects from class Error that contain how much the error has to be increased every cycle
        """

        # calculate degeneration rates for every var in error object
        temp_dict = {}
        for var_name in vars(self.dyn_error_data):
            var_val = getattr(self.dyn_error_data, var_name)

            if var_name == "add_remove_state":
                temp_dict[var_name] = self.calc_dynerr_integer(cycles, var_val)

            else:
                temp_dict[var_name] = self.calc_dynerr_continous(cycles, var_val)

        # create list of objects
        list_errors = []
        for cnt in range(0, cycles):
            list_errors.append(
                Errors(
                    noise=temp_dict['noise'][cnt],
                    add_remove_state=temp_dict['add_remove_state'][cnt],
                    offset_states=temp_dict['offset_states'][cnt],
                    poles_shift=temp_dict['poles_shift'][cnt]
                )
            )

        return list_errors

    @staticmethod
    def calc_dynerr_continous(cycles, integral):
        """Subfunction that calculates the added error per cycle for a continous variable

        Args:
            cycles: Number of cycles.
            integral: Target intensity of the error.
        """
        scaled_derivate = (np.ones(cycles) / cycles) * integral

        return scaled_derivate

    @staticmethod
    def calc_dynerr_integer(cycles, total_count):
        """Subfunction that calculates the error added per cycle for an integer Value.

        Args:
            cycles:
                Cycles that should be simulated.
            total_count:
                Count that should be added or removed over the cycles.
        """

        deriv_arr = np.zeros(cycles)

        if total_count != 0:
            d_cycle = math.floor(cycles / abs(total_count))
            increment = total_count / abs(total_count)  # get sign for increment

            # case less cycles than total count
            if d_cycle == 0:
                increment = math.ceil(total_count / cycles)
                d_cycle = 1

            # write in array
            for counter in range(0, cycles):
                if (counter % d_cycle) == 0 and abs(total_count) > 0:
                    deriv_arr[counter] = increment
                    total_count = abs(total_count) - 1

        return deriv_arr

    def print_info(self):
        """Print info about the automaton and its dynamical system."""
        print("Values of States", self.states)
        print("Name of States", self.state_names)
        print("Dynamical System:\n")
        self.system.show_behavior()
