import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.operators import apply_hop


def nearest_neighbor_bonds(L: int, periodic: bool) -> tuple[tuple[int, int], ...]:
    if L < 2:
        return ()

    if periodic:
        return tuple((site, (site + 1) % L) for site in range(L))

    return tuple((site, site + 1) for site in range(L - 1))


def hubbard_hamiltonian(
    basis: HubbardBasis,
    t: float = 1.0,
    U: float = 0.0,
    periodic: bool = True,
) -> csr_matrix:
    rows = []
    cols = []
    data = []

    bonds = nearest_neighbor_bonds(basis.L, periodic)

    for column, (up, down) in enumerate(basis.states):
        double_occupancy = (up & down).bit_count()

        if double_occupancy:
            rows.append(column)
            cols.append(column)
            data.append(U * double_occupancy)

        for spin in ("up", "down"):
            for left, right in bonds:
                for destination, source in ((right, left), (left, right)):
                    result = apply_hop(
                        up,
                        down,
                        destination=destination,
                        source=source,
                        spin=spin,
                        L=basis.L,
                    )

                    if result is None:
                        continue

                    state, sign = result
                    row = basis.index(*state)

                    rows.append(row)
                    cols.append(column)
                    data.append(-t * sign)

    matrix = coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(basis.dim, basis.dim),
        dtype=np.float64,
    )

    return matrix.tocsr()
