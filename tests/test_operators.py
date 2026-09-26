import pytest

from hubbard_ac.physics.operators import (
    annihilate,
    apply_creation,
    apply_hop,
    apply_annihilation,
    combine_state,
    create,
    orbital_index,
    split_state,
)


def apply_sequence(state, operations):
    amplitude = 1

    for operation, orbital in operations:
        if operation == "create":
            result = create(state, orbital)
        else:
            result = annihilate(state, orbital)

        if result is None:
            return None

        state, sign = result
        amplitude *= sign

    return state, amplitude


def test_combine_split_roundtrip():
    L = 6

    up = 0b001011
    down = 0b101100

    state = combine_state(up, down, L)

    assert split_state(state, L) == (up, down)


def test_orbital_ordering():
    L = 4

    assert orbital_index(0, "up", L) == 0
    assert orbital_index(3, "up", L) == 3
    assert orbital_index(0, "down", L) == 4
    assert orbital_index(3, "down", L) == 7


def test_creation_on_empty_orbital():
    state = 0

    result = create(state, 2)

    assert result == (0b100, 1)


def test_annihilation_on_empty_orbital():
    assert annihilate(0, 2) is None


def test_creation_on_occupied_orbital():
    state = 0b100

    assert create(state, 2) is None


def test_annihilation_sign():
    state = 0b1011

    new_state, sign = annihilate(state, 3)

    assert new_state == 0b0011
    assert sign == 1


def test_creation_sign():
    state = 0b0011

    new_state, sign = create(state, 3)

    assert new_state == 0b1011
    assert sign == 1


@pytest.mark.parametrize("i", range(6))
def test_number_operator_identity(i):
    empty = 0

    first = create(empty, i)

    assert first is not None

    occupied, sign_1 = first
    second = annihilate(occupied, i)

    assert second is not None

    final_state, sign_2 = second

    assert final_state == empty
    assert sign_1 * sign_2 == 1


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 1),
        (0, 3),
        (1, 4),
        (2, 5),
    ],
)
def test_creation_anticommutation(i, j):
    state = 0

    result_ij = apply_sequence(
        state,
        [
            ("create", j),
            ("create", i),
        ],
    )

    result_ji = apply_sequence(
        state,
        [
            ("create", i),
            ("create", j),
        ],
    )

    assert result_ij is not None
    assert result_ji is not None

    state_ij, amp_ij = result_ij
    state_ji, amp_ji = result_ji

    assert state_ij == state_ji
    assert amp_ij == -amp_ji


def test_cross_spin_anticommutation():
    L = 4
    up_orbital = orbital_index(1, "up", L)
    down_orbital = orbital_index(2, "down", L)

    result_ud = apply_sequence(
        0,
        [
            ("create", down_orbital),
            ("create", up_orbital),
        ],
    )

    result_du = apply_sequence(
        0,
        [
            ("create", up_orbital),
            ("create", down_orbital),
        ],
    )

    assert result_ud is not None
    assert result_du is not None

    state_ud, amp_ud = result_ud
    state_du, amp_du = result_du

    assert state_ud == state_du
    assert amp_ud == -amp_du


def test_apply_creation_and_annihilation():
    L = 4

    up = 0b0010
    down = 0b0100

    created = apply_creation(up, down, 3, "up", L)

    assert created is not None

    (new_up, new_down), _ = created

    assert new_up == 0b1010
    assert new_down == down

    removed = apply_annihilation(new_up, new_down, 3, "up", L)

    assert removed is not None

    (final_up, final_down), _ = removed

    assert final_up == up
    assert final_down == down


def test_nearest_neighbor_hop():
    L = 4

    up = 0b0010
    down = 0

    result = apply_hop(
        up,
        down,
        destination=2,
        source=1,
        spin="up",
        L=L,
    )

    assert result is not None

    (new_up, new_down), sign = result

    assert new_up == 0b0100
    assert new_down == 0
    assert sign == 1


def test_hop_to_occupied_site_is_zero():
    L = 4

    up = 0b0110
    down = 0

    result = apply_hop(
        up,
        down,
        destination=2,
        source=1,
        spin="up",
        L=L,
    )

    assert result is None
