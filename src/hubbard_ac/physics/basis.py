from itertools import combinations
from math import comb


def particle_masks(L: int, N: int) -> tuple[int, ...]:
    if L < 1:
        raise ValueError("L must be positive")
    if N < 0 or N > L:
        raise ValueError("N must satisfy 0 <= N <= L")

    masks = [
        sum(1 << site for site in occupied)
        for occupied in combinations(range(L), N)
    ]
    return tuple(sorted(masks))


class HubbardBasis:
    def __init__(self, L: int, n_up: int, n_down: int):
        self.L = L
        self.n_up = n_up
        self.n_down = n_down

        up_masks = particle_masks(L, n_up)
        down_masks = particle_masks(L, n_down)

        self.states = tuple(
            (up, down)
            for up in up_masks
            for down in down_masks
        )

        self._index = {
            state: i
            for i, state in enumerate(self.states)
        }

    @property
    def dim(self) -> int:
        return len(self.states)

    @property
    def expected_dim(self) -> int:
        return comb(self.L, self.n_up) * comb(self.L, self.n_down)

    def state(self, index: int) -> tuple[int, int]:
        return self.states[index]

    def index(self, up: int, down: int) -> int:
        return self._index[(up, down)]
