def orbital_index(site: int, spin: str, L: int) -> int:
    if site < 0 or site >= L:
        raise ValueError("site out of range")

    if spin == "up":
        return site
    if spin == "down":
        return L + site

    raise ValueError("spin must be 'up' or 'down'")


def combine_state(up: int, down: int, L: int) -> int:
    return up | (down << L)


def split_state(state: int, L: int) -> tuple[int, int]:
    mask = (1 << L) - 1
    up = state & mask
    down = (state >> L) & mask
    return up, down


def fermion_parity(state: int, orbital: int) -> int:
    occupied_before = (state & ((1 << orbital) - 1)).bit_count()
    return -1 if occupied_before % 2 else 1


def annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not (state & (1 << orbital)):
        return None

    sign = fermion_parity(state, orbital)
    new_state = state ^ (1 << orbital)

    return new_state, sign


def create(state: int, orbital: int) -> tuple[int, int] | None:
    if state & (1 << orbital):
        return None

    sign = fermion_parity(state, orbital)
    new_state = state | (1 << orbital)

    return new_state, sign


def apply_annihilation(
    up: int,
    down: int,
    site: int,
    spin: str,
    L: int,
) -> tuple[tuple[int, int], int] | None:
    state = combine_state(up, down, L)
    orbital = orbital_index(site, spin, L)

    result = annihilate(state, orbital)

    if result is None:
        return None

    new_state, sign = result
    return split_state(new_state, L), sign


def apply_creation(
    up: int,
    down: int,
    site: int,
    spin: str,
    L: int,
) -> tuple[tuple[int, int], int] | None:
    state = combine_state(up, down, L)
    orbital = orbital_index(site, spin, L)

    result = create(state, orbital)

    if result is None:
        return None

    new_state, sign = result
    return split_state(new_state, L), sign


def apply_hop(
    up: int,
    down: int,
    destination: int,
    source: int,
    spin: str,
    L: int,
) -> tuple[tuple[int, int], int] | None:
    state = combine_state(up, down, L)

    source_orbital = orbital_index(source, spin, L)
    destination_orbital = orbital_index(destination, spin, L)

    first = annihilate(state, source_orbital)

    if first is None:
        return None

    intermediate, sign_1 = first

    second = create(intermediate, destination_orbital)

    if second is None:
        return None

    final_state, sign_2 = second

    return split_state(final_state, L), sign_1 * sign_2
