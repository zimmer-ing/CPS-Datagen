#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass

import torch

from virtualCPPS.config import Config


def get_cc_output_params(config: Config) -> torch.Tensor:
    num_terms = (config.degree_cc + 1) ** config.dims_cc_input

    cc_output_params = (
        (torch.rand(size=[num_terms, config.dims_cc_output]) - 0.5)
        * config.max_cc_param
    ).to(config.device)

    return cc_output_params


def get_obs_params(config: Config) -> torch.Tensor:
    obs_params = (
        (
            torch.rand(
                size=[
                    config.dim_latent,
                    config.dims_obs,
                ]
            )
            - 0.5
        )
        * 2
        * config.max_obs_param
    ).to(config.device)

    return obs_params
