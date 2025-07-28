from __future__ import annotations


from virtualCPPS.dynamical_system import DynamicalSystem
from virtualCPPS.automaton import Automaton
from virtualCPPS.config import Config
from virtualCPPS.dataclasses import TimeSeries, Errors


def get(config: Config, dataset_type):
    if dataset_type == 'train':
        automaton = init(config, config.random_seed_training)
        data = automaton.simulate(config.cycles)

    if dataset_type == 'val':
        automaton = init(config, config.random_seed_training)
        automaton.redefine_random_seed(config.random_seed_val)
        data = automaton.simulate(config.cycles)

    if dataset_type == 'error':
        automaton = init(config, config.random_seed_training)
        automaton.redefine_random_seed(config.random_seed_error)

        if config.error_type =='static':
            param_error=Errors(noise=config.add_noise_states_error,
                               add_remove_state=config.cnt_remov_add_states,
                               offset_states=config.offset_states,
                               poles_shift=config.poles_shift
                               )
            automaton.apply_all_errors(param_error)

        if config.error_type=='dynamic':
            automaton.init_dynamic_error(noise=config.add_noise_states_error,
                                         add_remove_state=config.cnt_remov_add_states,
                                         offset_states=config.offset_states,
                                         poles_shift=config.poles_shift
                                         )


        data = automaton.simulate(config.cycles)


    return  data, data.Time[-1]


def init(config: Config, random_seed):
    # parameters

    system = DynamicalSystem(poles=config.poles,
                             zeros=config.zeros,
                             dt=config.delta_t)

    automaton = Automaton(system,
                          num_states=config.num_states,
                          random_seed=random_seed,
                          distance_to_targetstate=config.distance_to_targetstate,
                          gradient_at_target=config.gradient_at_targetstate,
                          cycletime_controller=config.cycletime_controller)

    return automaton
