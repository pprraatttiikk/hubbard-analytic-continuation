import numpy as np
import pytest

from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.hamiltonian import (
    hubbard_hamiltonian,
    nearest_neighbor_bonds,
)


def test_open_bonds():
    assert nearest_neighbor_bonds(4, periodic=False) == (
        (0, 1),
        (1, 2),
        (2, 3),
    )


def test_periodic_bonds():
    assert nearest_neighbor_bonds(4, periodic=True) == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    )


@pytest.mark.parametrize(
    "L,n_up,n_down,U,periodic",
    [
        (2, 1, 1, 0.0, False),
        (2, 1, 1, 4.0, False),
        (4, 2, 2, 2.0, False),
        (4, 2, 2, 4.0, True),
        (6, 3, 3, 8.0, True),
    ],
)
def test_hamiltonian_is_hermitian(L, n_up, n_down, U, periodic):
    basis = HubbardBasis(L, n_up, n_down)
    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=U,
        periodic=periodic,
    )

    difference = H - H.getH()

    assert np.max(np.abs(difference.toarray())) < 1e-12


def test_interaction_only_diagonal():
    basis = HubbardBasis(4, 2, 2)

    U = 3.7

    H = hubbard_hamiltonian(
        basis,
        t=0.0,
        U=U,
        periodic=True,
    ).toarray()

    expected = np.array(
        [
            U * (up & down).bit_count()
            for up, down in basis.states
        ]
    )

    assert np.allclose(np.diag(H), expected)

    off_diagonal = H - np.diag(np.diag(H))
    assert np.allclose(off_diagonal, 0.0)


def test_free_particle_periodic_spectrum():
    basis = HubbardBasis(4, 1, 0)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=0.0,
        periodic=True,
    )

    eigenvalues = np.linalg.eigvalsh(H.toarray())

    expected = np.array([-2.0, 0.0, 0.0, 2.0])

    assert np.allclose(eigenvalues, expected, atol=1e-12)


def test_free_particle_open_spectrum():
    L = 4

    basis = HubbardBasis(L, 1, 0)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=0.0,
        periodic=False,
    )

    eigenvalues = np.linalg.eigvalsh(H.toarray())

    n = np.arange(1, L + 1)
    expected = np.sort(-2.0 * np.cos(n * np.pi / (L + 1)))

    assert np.allclose(eigenvalues, expected, atol=1e-12)


@pytest.mark.parametrize(
    "U,t",
    [
        (0.0, 1.0),
        (2.0, 1.0),
        (4.0, 1.0),
        (8.0, 1.0),
    ],
)
def test_hubbard_dimer_spectrum(U, t):
    basis = HubbardBasis(2, 1, 1)

    H = hubbard_hamiltonian(
        basis,
        t=t,
        U=U,
        periodic=False,
    )

    eigenvalues = np.linalg.eigvalsh(H.toarray())

    root = np.sqrt(U**2 + 16.0 * t**2)

    expected = np.sort(
        np.array(
            [
                0.5 * (U - root),
                0.0,
                U,
                0.5 * (U + root),
            ]
        )
    )

    assert np.allclose(eigenvalues, expected, atol=1e-12)
