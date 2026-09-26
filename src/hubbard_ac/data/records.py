from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalSample:
    sample_id: int
    L: int
    n_up: int
    n_down: int
    U: float
    beta: float
    k_index: int
    k: float
    mu: float
    poles: np.ndarray
    weights: np.ndarray
    spectrum: np.ndarray
    green: np.ndarray


@dataclass(frozen=True)
class Observation:
    physical_sample_id: int
    sigma: float
    realization: int
    green_input: np.ndarray
