import numpy as np


def gaussian_noise(
    signal: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)

    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")

    if sigma == 0.0:
        return signal.copy()

    return signal + rng.normal(
        loc=0.0,
        scale=sigma,
        size=signal.shape,
    )
