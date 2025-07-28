#!/usr/bin/env python3

from __future__ import annotations
from dataclasses import dataclass

import torch

from virtualCPPS.mapping_params import MappingParams
from virtualCPPS.config import Config

from virtualCPPS import gen_obs
from virtualCPPS import gen_latent
from virtualCPPS.dataclasses import TimeSeries
from virtualCPPS.gen_noise import noise_to_signal

@dataclass
class DataSet:
    dataset_type: str
    lenght: float
    raw_data: TimeSeries
    latent: torch.Tensor
    obs: torch.Tensor

    @classmethod
    def generate(
        cls, config: Config, mapping_params: MappingParams, dataset_type: str
    ) -> cls:

        data, time = gen_latent.get(config, dataset_type)

        if dataset_type == "train":
            latent = noise_to_signal(data.x,config.signal_noise,config.random_seed_training)

        if dataset_type == "val":
            latent = noise_to_signal(data.x, config.signal_noise, config.random_seed_val)

        if dataset_type == "error":
            #calc noise for error
            noise=(config.signal_noise + config.signal_noise_error)
            if noise < 0:
                noise = 0
            #apply
            latent = noise_to_signal(data.x ,noise, config.random_seed_error)

        obs = gen_obs.get(config, latent, mapping_params)

        dataset = cls(
            dataset_type=dataset_type,
            lenght=time,
            raw_data=data,
            latent=latent,
            obs=obs,
        )

        return dataset



    def get_all(self) -> torch.Tensor:
        dataset = torch.cat(
            [self.latent, self.obs], dim=1,
        )
        return dataset
