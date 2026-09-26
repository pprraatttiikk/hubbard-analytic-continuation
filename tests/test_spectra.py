import numpy as np
import pytest

from hubbard_ac.physics.sector_operators import momentum_grid
from hubbard_ac.physics.spectra import (
    broaden_spectrum,
    hubbard_lehmann_spectrum,
    lorentzian,
)


@pytest.mark.parametrize(
    "U",
    [
        0.0,
        2.0,
        4.0,
        8.0,
    ],
)
def test_spectral_sum_rule(U):
    L = 4
    k = momentum_grid(L)[0]

    spectrum = hubbard_lehmann_spectrum(
        L=L,
        n_up=2,
        n_down=2,
        U=U,
        k=k,
        spin="up",
        t=1.0,
        periodic=True,
    )

    assert np.isclose(
        spectrum.total_weight,
        1.0,
        atol=1e-10,
    )


@pytest.mark.parametrize("spin", ["up", "down"])
def test_spectral_sum_rule_both_spins(spin):
    L = 4
    k = momentum_grid(L)[1]

    spectrum = hubbard_lehmann_spectrum(
        L=L,
        n_up=2,
        n_down=2,
        U=4.0,
        k=k,
        spin=spin,
        t=1.0,
        periodic=True,
    )

    assert np.isclose(
        spectrum.total_weight,
        1.0,
        atol=1e-10,
    )


def test_noninteracting_dispersion_L6():
    L = 6

    for k in momentum_grid(L):
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

        expected_energy = -2.0 * np.cos(k)

        weights = spectrum.weights
        poles = spectrum.poles

        assert np.isclose(
            np.sum(weights),
            1.0,
            atol=1e-10,
        )

        mean_energy = np.sum(
            weights * poles
        )

        variance = np.sum(
            weights * (poles - expected_energy) ** 2
        )

        assert np.isclose(
            mean_energy,
            expected_energy,
            atol=1e-10,
        )

        assert variance < 1e-18


def test_half_filled_particle_hole_weight():
    L = 4
    U = 4.0

    k = momentum_grid(L)[1]

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

    assert np.isclose(
        np.sum(spectrum.addition_weights),
        np.sum(spectrum.removal_weights),
        atol=1e-10,
    )


def test_lorentzian_normalization():
    omega = np.linspace(
        -100.0,
        100.0,
        200001,
    )

    values = lorentzian(
        omega,
        center=0.0,
        eta=0.1,
    )

    integral = np.trapezoid(
        values,
        x=omega,
    )

    assert np.isclose(
        integral,
        1.0,
        atol=1e-3,
    )


def test_broadened_spectrum_weight():
    omega = np.linspace(
        -100.0,
        100.0,
        200001,
    )

    poles = np.array([-2.0, 1.0, 4.0])
    weights = np.array([0.2, 0.5, 0.3])

    spectrum = broaden_spectrum(
        omega=omega,
        poles=poles,
        weights=weights,
        eta=0.1,
    )

    integral = np.trapezoid(
        spectrum,
        x=omega,
    )

    assert np.isclose(
        integral,
        np.sum(weights),
        atol=1e-3,
    )


def test_broadened_spectrum_is_nonnegative():
    omega = np.linspace(
        -10.0,
        10.0,
        1001,
    )

    spectrum = broaden_spectrum(
        omega=omega,
        poles=np.array([-1.0, 2.0]),
        weights=np.array([0.4, 0.6]),
        eta=0.1,
    )

    assert np.all(spectrum >= 0.0)
