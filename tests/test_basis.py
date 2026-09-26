from math import comb

import pytest

from hubbard_ac.physics.basis import HubbardBasis, particle_masks


@pytest.mark.parametrize(
    "L,N,expected",
    [
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 1),
        (4, 2, 6),
        (6, 3, 20),
    ],
)
def test_particle_masks_count(L, N, expected):
    masks = particle_masks(L, N)
    assert len(masks) == expected
    assert expected == comb(L, N)


@pytest.mark.parametrize(
    "L,N",
    [
        (2, 0),
        (2, 1),
        (2, 2),
        (4, 2),
        (6, 3),
    ],
)
def test_particle_masks_have_correct_particle_number(L, N):
    for mask in particle_masks(L, N):
        assert mask.bit_count() == N


@pytest.mark.parametrize(
    "L,n_up,n_down",
    [
        (2, 1, 1),
        (4, 2, 2),
        (6, 3, 3),
        (6, 4, 3),
        (6, 2, 3),
    ],
)
def test_basis_dimension(L, n_up, n_down):
    basis = HubbardBasis(L, n_up, n_down)

    expected = comb(L, n_up) * comb(L, n_down)

    assert basis.dim == expected
    assert basis.expected_dim == expected


def test_half_filled_L6_dimension():
    basis = HubbardBasis(6, 3, 3)
    assert basis.dim == 400


def test_basis_states_are_unique():
    basis = HubbardBasis(6, 3, 3)
    assert len(set(basis.states)) == basis.dim


def test_basis_particle_numbers():
    basis = HubbardBasis(6, 3, 3)

    for up, down in basis.states:
        assert up.bit_count() == 3
        assert down.bit_count() == 3


def test_index_roundtrip():
    basis = HubbardBasis(4, 2, 2)

    for i, (up, down) in enumerate(basis.states):
        assert basis.index(up, down) == i
        assert basis.state(i) == (up, down)


def test_invalid_particle_number():
    with pytest.raises(ValueError):
        particle_masks(4, -1)

    with pytest.raises(ValueError):
        particle_masks(4, 5)
