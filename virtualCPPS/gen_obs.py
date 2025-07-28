#!/usr/bin/env python3

import torch


def get(config, latent_raw , mapping_params):

    latent = torch.tensor(latent_raw, device=config.device, dtype=torch.float32)

    obs_lin = torch.matmul(latent, mapping_params.obs_lin)

    obs_sin = torch.matmul(torch.sin(latent), mapping_params.obs_sin)

    obs_exp = torch.matmul(torch.exp(latent), mapping_params.obs_exp)

    obs = (
        config.mapping_factor_lin * obs_lin
        + config.mapping_factor_sin * obs_sin
        + config.mapping_factor_exp * obs_exp
    ).to(config.device)

    return obs
