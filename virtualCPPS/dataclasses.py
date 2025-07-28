from dataclasses import dataclass
import numpy as np
import pandas as pd





@dataclass
class Errors:
    """Dataclass that handles the Errors which can be applied"""
    noise: float
    add_remove_state: int
    offset_states: float
    poles_shift: float

@dataclass
class TimeSeries:
    """This Dataclass contains all information that are needed for the latent Timeseries"""
    applied_errors: pd.DataFrame = pd.DataFrame()
    Time: np.ndarray = np.array([])
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    statename: np.ndarray = np.array([])
    cycle: np.ndarray = np.array([])

