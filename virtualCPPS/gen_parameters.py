import virtualCPPS as vCPPS
import itertools
import copy


def cross_param():
    # fixed parameters
    cycletime_controller = 0.1
    delta_t = 0.01
    distance_to_targetstate = 0.5
    gradient_at_targetstate = 10

    list_error_type = ["dynamic"]
    list_poles = [[-15, -5+3.j, -5.-3j]]
    list_zeros = [[10]]
    list_num_states = [5,10]
    list_cycles = [20]
    list_noise_states = [2,5]
    list_add_noise_states_error=[5]
    list_num_removed_states = [-2,4]
    list_offset_states = [0,5,25]
    list_poles_shift = [0,0.2,0.5]
    list_signal_noise = [0.025]
    list_signal_noise_error = [0,0.1]
    list_dims_obs = [10]
    list_mapping_factor_sin = [0,1.0]
    list_mapping_factor_exp = [0,0.1]




    param_combinations = list(
        itertools.product(
            list_error_type,
            list_poles,
            list_zeros,
            list_num_states,
            list_cycles,
            list_noise_states,
            list_add_noise_states_error,
            list_num_removed_states,
            list_offset_states,
            list_poles_shift,
            list_signal_noise,
            list_signal_noise_error,
            list_dims_obs,
            list_mapping_factor_sin,
            list_mapping_factor_exp,
        )
    )

    print("Number of parameter combinations:", len(param_combinations))
    config = vCPPS.Config()
    configs=[]
    for (
        config.error_type,
        config.poles,
        config.zeros ,
        config.num_states,
        config.cycles,
        config.noise_states,
        config.add_noise_states_error,
        config.cnt_remov_add_states,
        config.offset_states,
        config.poles_shift,
        config.signal_noise,
        config.signal_noise_error,
        config.dims_obs,
        config.mapping_factor_sin,
        config.mapping_factor_exp,
    ) in param_combinations:
        # generate default configuration object


        #fixed
        config.cycletime_controller = cycletime_controller
        config.delta_t = delta_t
        config.distance_to_targetstate = distance_to_targetstate
        config.gradient_at_targetstate = gradient_at_targetstate
        config.dim_latent=len(config.poles)
        config.device ="cpu"


        config.collection_name = (
            f"{config.error_type}_"
            f"p{config.poles}_"
            f"z{config.zeros}_"
            f"s#{config.num_states}_"
            f"c#{config.cycles}_"
            f"n{config.noise_states}_"
            f"ne{config.add_noise_states_error}_"
            f"rs{config.cnt_remov_add_states}_"
            f"os{config.offset_states}_"
            f"ps{config.poles_shift}_"
            f"sn{config.signal_noise}_"
            f"sne{config.signal_noise_error}_"
            f"dobs{config.dims_obs}_"
            f"f_sin{config.mapping_factor_sin}_"
            f"f_exp{config.mapping_factor_exp}"
        )


        configs.append(copy.deepcopy(config))

    return configs



