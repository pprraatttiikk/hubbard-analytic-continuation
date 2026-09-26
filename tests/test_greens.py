import numpy as np
import pytest

from hubbard_ac.physics.greens import (
    fermionic_kernel,
    greens_from_poles,
    greens_from_spectrum,
)
from hubbard_ac.physics.sector_operators import momentum_grid
from hubbard_ac.physics.spectra import (
    broaden_spectrum,
    hubbard_lehmann_spectrum,
)


def test_kernel_shape():
    tau = np.linspace(0.0, 10.0, 101)
    omega = np.linspace(-5.0, 5.0, 201)

    kernel = fermionic_kernel(
        tau=tau,
        omega=omega,
        beta=10.0,
    )

    assert kernel.shape == (101, 201)


def test_kernel_is_negative():
    tau = np.linspace(0.0, 10.0, 101)
    omega = np.linspace(-5.0, 5.0, 201)

    kernel = fermionic_kernel(
        tau=tau,
        omega=omega,
        beta=10.0,
    )

    assert np.all(kernel <= 0.0)


def test_single_zero_energy_pole():
    beta = 10.0
    tau = np.linspace(0.0, beta, 101)

    G = greens_from_poles(
        tau=tau,
        beta=beta,
        poles=np.array([0.0]),
        weights=np.array([1.0]),
    )

    assert np.allclose(
        G,
        -0.5,
        atol=1e-14,
    )


def test_endpoint_identity_for_normalized_spectrum():
    beta = 20.0

    poles = np.array(
        [-3.0, -1.0, 0.5, 2.5]
    )

    weights = np.array(
        [0.15, 0.35, 0.30, 0.20]
    )

    G = greens_from_poles(
        tau=np.array([0.0, beta]),
        beta=beta,
        poles=poles,
        weights=weights,
    )

    assert np.isclose(
        G[0] + G[1],
        -1.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "U,k_index",
    [
        (0.0, 0),
        (2.0, 1),
        (4.0, 2),
        (8.0, 3),
    ],
)
def test_hubbard_endpoint_identity(U, k_index):
    L = 4
    beta = 20.0

    k = momentum_grid(L)[k_index]

    spectrum = hubbard_lehmann_spectrum(
        L=L,
        n_up=2,
        n_down=2,
        U=U,
        k=k,
        spin="up",
        t=1.0,
        mu=U / 2.0,
        periodic=True,
    )

    G = greens_from_poles(
        tau=np.array([0.0, beta]),
        beta=beta,
        poles=spectrum.poles,
        weights=spectrum.weights,
    )

    assert np.isclose(
        G[0] + G[1],
        -1.0,
        atol=1e-10,
    )


def test_noninteracting_green_function():
    L = 6
    beta = 10.0
    k = momentum_grid(L)[0]

    spectrum = hubbard_lehmann_spectrum(
        L=L,
        n_up=3,
        n_down=3,
        U=0.0,
        k=k,
        spin="up",
        t=1.0,
        mu=0.0,
        periodic=True,
    )

    tau = np.linspace(
        0.0,
        beta,
        51,
    )

    G = greens_from_poles(
        tau=tau,
        beta=beta,
        poles=spectrum.poles,
        weights=spectrum.weights,
    )

    epsilon = -2.0 * np.cos(k)

    expected = (
        -np.exp(-tau * epsilon)
        / (1.0 + np.exp(-beta * epsilon))
    )

    assert np.allclose(
        G,
        expected,
        atol=1e-10,
    )


def test_broadened_and_pole_greens_agree():
    beta = 5.0

    poles = np.array([-1.5, 0.75])
    weights = np.array([0.4, 0.6])

    tau = np.linspace(
        0.0,
        beta,
        31,
    )

    G_poles = greens_from_poles(
        tau=tau,
        beta=beta,
        poles=poles,
        weights=weights,
    )

    omega = np.linspace(
        -20.0,
        20.0,
        40001,
    )

    spectrum = broaden_spectrum(
        omega=omega,
        poles=poles,
        weights=weights,
        eta=0.01,
    )

    G_grid = greens_from_spectrum(
        tau=tau,
        beta=beta,
        omega=omega,
        spectrum=spectrum,
    )

    assert np.max(
        np.abs(G_grid - G_poles)
    ) < 1e-2


def test_invalid_tau_raises():
    with pytest.raises(ValueError):
        fermionic_kernel(
            tau=np.array([-0.1]),
            omega=np.array([1.0]),
            beta=10.0,
        )

    with pytest.raises(ValueError):
        fermionic_kernel(
            tau=np.array([10.1]),
            omega=np.array([1.0]),
            beta=10.0,
        )


def test_invalid_beta_raises():
    with pytest.raises(ValueError):
        fermionic_kernel(
            tau=np.array([0.0]),
            omega=np.array([1.0]),
            beta=0.0,
        )
