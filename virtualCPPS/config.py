#!/usr/bin/env python3

from typing import Union, List, Tuple
from dataclasses import dataclass, field
import datetime

import torch


@dataclass
class Config:
    """This dataclass holds all configuration necessary to generate dataset collections.

    Args:
        device: The device used for the calculations. Options: "cpu", "cuda".
        num_samples: The number of samples a generation run will produce.
        dims_cc_input: The number of dimensions of the manifold.
        spread_cc_input: The size of the latent space in the cc_input dimension.
            Used as boundaries for the random walk.
        on_boundary_hit: Specifies what happens when a random walk hits the boundary of the latent space.
            Available options: "wrap", "reflect", "restart" ("restart" has lower performance).
        walk_bias: not implemented
        walk_distance: If this value is of type float, it specifies the length each step in the walk has.
            If this value is of type tuple or list, it specifies the min and max values of the step length.
        walk_reset_every: Start a new random walk every n steps.
        max_cc_param: The maximum value the parameters of the polynomial mapping function can have.
        dims_cc_output: The number of dimensions of the output of the polynomial function.
            This means, that the latent space has dims_cc_input + dims_cc_output dimensions.
        degree_cc: The degree of the polynomial function.
        spread_cc_output: The size of the latent space in the cc_output dimension.
            Used as multiplier for the normalized output of the polynomial function.
        force_term_exp_sum: If "True", term in the polynomial where sum(exponents) != degree_cc will be ignored.
        no_error_offset: Specifies the error offset in "normal data".
            If this value is of type float, it specifies the distance of the error mean from the manifold.
            If this value is of type tuple or list, it specifies the distances at the start and end of the run.
        no_error_spread: Specifies the error spread in "normal data".
            If this value is of type float, it specifies the variance of the error from the manifold.
            If this value is of type tuple or list, it specifies the variance at the start and end of the run.
        error_offset: Specifies the error offset in "anormal data".
            If this value is of type float, it specifies the distance of the error mean from the manifold.
            If this value is of type tuple or list, it specifies the distances at the start and end of the run.
        error_spread: Specifies the error spread in "anormal data".
            If this value is of type float, it specifies the variance of the error from the manifold.
            If this value is of type tuple or list, it specifies the variance at the start and end of the run.
        max_obs_param: The maximum value the parameters of the obs mapping function can have.
        dims_obs: The number of dimensions the observable space has.
        mapping_factor_lin: Weighting parameter defining the strength of the linear part of the obs mapping.
        mapping_factor_sin: Weighting parameter defining the strength of the sinusoidal part of the obs mapping.
        mapping_factor_exp: Weighting parameter defining the strength of the exponential part of the obs mapping.
        calc_coverage: Whether or not to calculate a metric specifying the fraction of the manifold that is occupied by at least one point.
        coverage_granularity: The number of section each cc_input dimension is split into.
            In the case of a 2-dimensional manifold this can be thought of as the number of cells in a grid.
        save_data: Whether or not to save the generated data to dist.
        save_datavis: Whether or not to save visualizations of the generated data.
        save_path: The relative path where the data is saved. A new folder will be created at that path.
        delimiter: The value delimiter used in the saved data.
        timestamp: The current timestamp of the run.
            If save_data is "True" it is used in the folder of the exported data.

    Returns:
        Config object holding all the configuration described above.
    """

    collection_name: str = "Not defined"
    # device
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    error_type: str = "None"
    # dynamical system parameter
    poles: List[complex] = field(default_factory=lambda: [-10, -2+3.j, -2.-3j])
    zeros: List[complex] = field(default_factory=lambda: [-11])
    delta_t: float = 0.01
    dim_latent: int = 3

    # automaton parameter
    num_states: int = 10
    cycles: int = 50
    cycletime_controller: float = 0.1
    noise_states: float = 5
    signal_noise: float = 0.005
    distance_to_targetstate: float = 0.1
    gradient_at_targetstate: float = 5

    #random seed
    random_seed_training: int = 42
    random_seed_val: int = 1992
    random_seed_error: int = 2020

    # implied errors
    add_noise_states_error: float = 10
    cnt_remov_add_states: int = 2
    poles_shift: float = 10
    offset_states: float = 20
    signal_noise_error: float =-0.01



    # obs parameter generation
    max_obs_param: float = 1.0

    # obs generation
    dims_obs: int = 50
    mapping_factor_lin: float = 1.0
    mapping_factor_sin: float = 1.0
    mapping_factor_exp: float = 0.1

    # data save parameters
    calc_coverage: bool = True
    coverage_granularity: int = 100
    save_data: bool = True
    save_datavis: bool = False
    save_path: str = "./data"
    delimiter: str = ","
    timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
