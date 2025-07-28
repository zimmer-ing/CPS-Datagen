#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass

import torch

from virtualCPPS.config import Config
from virtualCPPS import gen_mapping_params


@dataclass
class MappingParams:
    obs_lin: torch.Tensor
    obs_sin: torch.Tensor
    obs_exp: torch.Tensor

    @classmethod
    def generate(cls, config: Config) -> cls:
        mapping_params = cls(
            obs_lin=gen_mapping_params.get_obs_params(config),
            obs_sin=gen_mapping_params.get_obs_params(config),
            obs_exp=gen_mapping_params.get_obs_params(config),
        )
        return mapping_params
