import numpy as np


def fermionic_kernel(
    tau: np.ndarray,
    omega: np.ndarray,
    beta: float,
) -> np.ndarray:
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    tau = np.atleast_1d(
        np.asarray(tau, dtype=np.float64)
    )
    omega = np.atleast_1d(
        np.asarray(omega, dtype=np.float64)
    )

    if np.any(tau < 0.0) or np.any(tau > beta):
        raise ValueError("tau must satisfy 0 <= tau <= beta")

    exponent = (
        -tau[:, None] * omega[None, :]
        - np.logaddexp(
            0.0,
            -beta * omega[None, :],
        )
    )

    return -np.exp(exponent)


def greens_from_poles(
    tau: np.ndarray,
    beta: float,
    poles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    poles = np.asarray(poles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if poles.shape != weights.shape:
        raise ValueError("poles and weights must have the same shape")

    kernel = fermionic_kernel(
        tau=tau,
        omega=poles,
        beta=beta,
    )

    return kernel @ weights


def greens_from_spectrum(
    tau: np.ndarray,
    beta: float,
    omega: np.ndarray,
    spectrum: np.ndarray,
) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if omega.ndim != 1:
        raise ValueError("omega must be one-dimensional")

    if spectrum.shape != omega.shape:
        raise ValueError("spectrum and omega must have the same shape")

    kernel = fermionic_kernel(
        tau=tau,
        omega=omega,
        beta=beta,
    )

    return np.trapezoid(
        kernel * spectrum[None, :],
        x=omega,
        axis=1,
    )
