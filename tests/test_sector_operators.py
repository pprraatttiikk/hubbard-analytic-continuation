import numpy as np
import pytest

from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.sector_operators import (
    momentum_annihilation_operator,
    momentum_creation_operator,
    momentum_grid,
    site_annihilation_operator,
    site_creation_operator,
)


def test_momentum_grid_L6():
    k = momentum_grid(6)

    expected = np.array(
        [
            0.0,
            np.pi / 3.0,
            2.0 * np.pi / 3.0,
            np.pi,
            4.0 * np.pi / 3.0,
            5.0 * np.pi / 3.0,
        ]
    )

    assert np.allclose(k, expected)


@pytest.mark.parametrize("spin", ["up", "down"])
def test_site_creation_annihilation_are_adjoints(spin):
    L = 4

    if spin == "up":
        lower = HubbardBasis(L, 1, 2)
        upper = HubbardBasis(L, 2, 2)
    else:
        lower = HubbardBasis(L, 2, 1)
        upper = HubbardBasis(L, 2, 2)

    for site in range(L):
        creation = site_creation_operator(
            lower,
            upper,
            site,
            spin,
        )

        annihilation = site_annihilation_operator(
            upper,
            lower,
            site,
            spin,
        )

        assert np.allclose(
            creation.toarray(),
            annihilation.getH().toarray(),
        )


@pytest.mark.parametrize("spin", ["up", "down"])
def test_momentum_creation_annihilation_are_adjoints(spin):
    L = 4

    if spin == "up":
        lower = HubbardBasis(L, 1, 2)
        upper = HubbardBasis(L, 2, 2)
    else:
        lower = HubbardBasis(L, 2, 1)
        upper = HubbardBasis(L, 2, 2)

    for k in momentum_grid(L):
        creation = momentum_creation_operator(
            lower,
            upper,
            k,
            spin,
        )

        annihilation = momentum_annihilation_operator(
            upper,
            lower,
            k,
            spin,
        )

        assert np.allclose(
            creation.toarray(),
            annihilation.getH().toarray(),
            atol=1e-12,
        )


@pytest.mark.parametrize("spin", ["up", "down"])
def test_site_anticommutator(spin):
    L = 4
    site = 2

    if spin == "up":
        basis = HubbardBasis(L, 2, 2)
        lower = HubbardBasis(L, 1, 2)
        upper = HubbardBasis(L, 3, 2)
    else:
        basis = HubbardBasis(L, 2, 2)
        lower = HubbardBasis(L, 2, 1)
        upper = HubbardBasis(L, 2, 3)

    c_down = site_annihilation_operator(
        basis,
        lower,
        site,
        spin,
    )

    cd_up = site_creation_operator(
        basis,
        upper,
        site,
        spin,
    )

    cd_from_lower = site_creation_operator(
        lower,
        basis,
        site,
        spin,
    )

    c_from_upper = site_annihilation_operator(
        upper,
        basis,
        site,
        spin,
    )

    anticommutator = (
        cd_from_lower @ c_down
        + c_from_upper @ cd_up
    )

    assert np.allclose(
        anticommutator.toarray(),
        np.eye(basis.dim),
        atol=1e-12,
    )


@pytest.mark.parametrize("spin", ["up", "down"])
def test_momentum_anticommutator(spin):
    L = 4
    k = momentum_grid(L)[1]

    if spin == "up":
        basis = HubbardBasis(L, 2, 2)
        lower = HubbardBasis(L, 1, 2)
        upper = HubbardBasis(L, 3, 2)
    else:
        basis = HubbardBasis(L, 2, 2)
        lower = HubbardBasis(L, 2, 1)
        upper = HubbardBasis(L, 2, 3)

    c_down = momentum_annihilation_operator(
        basis,
        lower,
        k,
        spin,
    )

    cd_up = momentum_creation_operator(
        basis,
        upper,
        k,
        spin,
    )

    cd_from_lower = momentum_creation_operator(
        lower,
        basis,
        k,
        spin,
    )

    c_from_upper = momentum_annihilation_operator(
        upper,
        basis,
        k,
        spin,
    )

    anticommutator = (
        cd_from_lower @ c_down
        + c_from_upper @ cd_up
    )

    assert np.allclose(
        anticommutator.toarray(),
        np.eye(basis.dim),
        atol=1e-12,
    )


def test_different_momenta_are_orthogonal():
    L = 4

    vacuum = HubbardBasis(L, 0, 0)
    one_particle = HubbardBasis(L, 1, 0)

    k_values = momentum_grid(L)

    for i, k in enumerate(k_values):
        annihilation = momentum_annihilation_operator(
            one_particle,
            vacuum,
            k,
            "up",
        )

        for j, q in enumerate(k_values):
            creation = momentum_creation_operator(
                vacuum,
                one_particle,
                q,
                "up",
            )

            overlap = (annihilation @ creation).toarray()[0, 0]

            expected = 1.0 if i == j else 0.0

            assert np.isclose(
                overlap,
                expected,
                atol=1e-12,
            )


def test_wrong_target_sector_raises():
    source = HubbardBasis(4, 2, 2)
    wrong_target = HubbardBasis(4, 2, 2)

    with pytest.raises(ValueError):
        site_annihilation_operator(
            source,
            wrong_target,
            site=0,
            spin="up",
        )
