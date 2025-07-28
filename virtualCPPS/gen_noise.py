import numpy as np


def noise_to_signal(input,intensity,seed):

    np.random.seed(seed)
    noise=np.random.standard_normal(input.shape)
    noise=intensity*noise*input

    output = np.add(input, noise)
    return output