import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.operators import (
    apply_annihilation,
    apply_creation,
)


def momentum_grid(L: int) -> np.ndarray:
    return 2.0 * np.pi * np.arange(L) / L


def _target_particle_numbers(
    source: HubbardBasis,
    spin: str,
    delta: int,
) -> tuple[int, int]:
    if spin == "up":
        return source.n_up + delta, source.n_down

    if spin == "down":
        return source.n_up, source.n_down + delta

    raise ValueError("spin must be 'up' or 'down'")


def site_annihilation_operator(
    source: HubbardBasis,
    target: HubbardBasis,
    site: int,
    spin: str,
) -> csr_matrix:
    expected = _target_particle_numbers(source, spin, -1)

    if (target.n_up, target.n_down) != expected:
        raise ValueError("target basis has wrong particle-number sector")

    if target.L != source.L:
        raise ValueError("source and target must have the same L")

    rows = []
    cols = []
    data = []

    for column, (up, down) in enumerate(source.states):
        result = apply_annihilation(
            up,
            down,
            site=site,
            spin=spin,
            L=source.L,
        )

        if result is None:
            continue

        state, sign = result
        row = target.index(*state)

        rows.append(row)
        cols.append(column)
        data.append(sign)

    return coo_matrix(
        (data, (rows, cols)),
        shape=(target.dim, source.dim),
        dtype=np.complex128,
    ).tocsr()


def site_creation_operator(
    source: HubbardBasis,
    target: HubbardBasis,
    site: int,
    spin: str,
) -> csr_matrix:
    expected = _target_particle_numbers(source, spin, 1)

    if (target.n_up, target.n_down) != expected:
        raise ValueError("target basis has wrong particle-number sector")

    if target.L != source.L:
        raise ValueError("source and target must have the same L")

    rows = []
    cols = []
    data = []

    for column, (up, down) in enumerate(source.states):
        result = apply_creation(
            up,
            down,
            site=site,
            spin=spin,
            L=source.L,
        )

        if result is None:
            continue

        state, sign = result
        row = target.index(*state)

        rows.append(row)
        cols.append(column)
        data.append(sign)

    return coo_matrix(
        (data, (rows, cols)),
        shape=(target.dim, source.dim),
        dtype=np.complex128,
    ).tocsr()


def momentum_annihilation_operator(
    source: HubbardBasis,
    target: HubbardBasis,
    k: float,
    spin: str,
) -> csr_matrix:
    operator = csr_matrix(
        (target.dim, source.dim),
        dtype=np.complex128,
    )

    normalization = 1.0 / np.sqrt(source.L)

    for site in range(source.L):
        phase = np.exp(-1j * k * site)

        operator += (
            normalization
            * phase
            * site_annihilation_operator(
                source,
                target,
                site,
                spin,
            )
        )

    return operator


def momentum_creation_operator(
    source: HubbardBasis,
    target: HubbardBasis,
    k: float,
    spin: str,
) -> csr_matrix:
    operator = csr_matrix(
        (target.dim, source.dim),
        dtype=np.complex128,
    )

    normalization = 1.0 / np.sqrt(source.L)

    for site in range(source.L):
        phase = np.exp(1j * k * site)

        operator += (
            normalization
            * phase
            * site_creation_operator(
                source,
                target,
                site,
                spin,
            )
        )

    return operator
