import numpy as np
import pytest

from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.ed import (
    full_eigensystem,
    ground_state,
    lowest_eigenpairs,
)
from hubbard_ac.physics.hamiltonian import hubbard_hamiltonian


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
def test_full_eigensystem_residual(
    L,
    n_up,
    n_down,
    U,
    periodic,
):
    basis = HubbardBasis(L, n_up, n_down)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=U,
        periodic=periodic,
    )

    eigenvalues, eigenvectors = full_eigensystem(H)

    residual = (
        H.toarray() @ eigenvectors
        - eigenvectors * eigenvalues[np.newaxis, :]
    )

    assert np.max(np.abs(residual)) < 1e-10


def test_full_eigenvectors_are_orthonormal():
    basis = HubbardBasis(4, 2, 2)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=4.0,
        periodic=True,
    )

    _, eigenvectors = full_eigensystem(H)

    overlap = eigenvectors.T @ eigenvectors

    assert np.allclose(
        overlap,
        np.eye(basis.dim),
        atol=1e-12,
    )


def test_eigenvalues_are_sorted():
    basis = HubbardBasis(4, 2, 2)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=4.0,
        periodic=True,
    )

    eigenvalues, _ = full_eigensystem(H)

    assert np.all(np.diff(eigenvalues) >= -1e-12)


@pytest.mark.parametrize(
    "U",
    [
        0.0,
        1.0,
        2.0,
        4.0,
        8.0,
    ],
)
def test_dimer_ground_state_energy(U):
    t = 1.0

    basis = HubbardBasis(2, 1, 1)

    H = hubbard_hamiltonian(
        basis,
        t=t,
        U=U,
        periodic=False,
    )

    energy, state = ground_state(H)

    expected = 0.5 * (
        U - np.sqrt(U**2 + 16.0 * t**2)
    )

    assert np.isclose(
        energy,
        expected,
        atol=1e-12,
    )

    assert np.isclose(
        np.linalg.norm(state),
        1.0,
        atol=1e-12,
    )


def test_sparse_and_dense_lowest_states_agree():
    basis = HubbardBasis(4, 2, 2)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=3.0,
        periodic=True,
    )

    dense_values, _ = full_eigensystem(H)

    sparse_values, _ = lowest_eigenpairs(
        H,
        k=4,
    )

    assert np.allclose(
        sparse_values,
        dense_values[:4],
        atol=1e-10,
    )


def test_ground_state_residual():
    basis = HubbardBasis(6, 3, 3)

    H = hubbard_hamiltonian(
        basis,
        t=1.0,
        U=4.0,
        periodic=True,
    )

    energy, state = ground_state(H)

    residual = H @ state - energy * state

    assert np.linalg.norm(residual) < 1e-9


def test_invalid_number_of_eigenpairs():
    basis = HubbardBasis(2, 1, 1)

    H = hubbard_hamiltonian(
        basis,
        periodic=False,
    )

    with pytest.raises(ValueError):
        lowest_eigenpairs(H, k=0)
