from dataclasses import dataclass

import numpy as np

from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.ed import full_eigensystem, ground_state
from hubbard_ac.physics.hamiltonian import hubbard_hamiltonian
from hubbard_ac.physics.sector_operators import (
    momentum_annihilation_operator,
    momentum_creation_operator,
)


@dataclass(frozen=True)
class LehmannSpectrum:
    addition_poles: np.ndarray
    addition_weights: np.ndarray
    removal_poles: np.ndarray
    removal_weights: np.ndarray

    @property
    def poles(self) -> np.ndarray:
        return np.concatenate(
            (self.removal_poles, self.addition_poles)
        )

    @property
    def weights(self) -> np.ndarray:
        return np.concatenate(
            (self.removal_weights, self.addition_weights)
        )

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.weights))


def lehmann_spectrum(
    ground_energy: float,
    ground_vector: np.ndarray,
    addition_energies: np.ndarray,
    addition_vectors: np.ndarray,
    removal_energies: np.ndarray,
    removal_vectors: np.ndarray,
    creation_operator,
    annihilation_operator,
    mu: float,
    weight_cutoff: float = 1e-14,
) -> LehmannSpectrum:
    addition_state = creation_operator @ ground_vector
    removal_state = annihilation_operator @ ground_vector

    addition_amplitudes = addition_vectors.conj().T @ addition_state
    removal_amplitudes = removal_vectors.conj().T @ removal_state

    addition_weights = np.abs(addition_amplitudes) ** 2
    removal_weights = np.abs(removal_amplitudes) ** 2

    addition_poles = (
        addition_energies
        - ground_energy
        - mu
    )

    removal_poles = (
        ground_energy
        - removal_energies
        - mu
    )

    addition_mask = addition_weights > weight_cutoff
    removal_mask = removal_weights > weight_cutoff

    return LehmannSpectrum(
        addition_poles=np.asarray(
            addition_poles[addition_mask],
            dtype=np.float64,
        ),
        addition_weights=np.asarray(
            addition_weights[addition_mask],
            dtype=np.float64,
        ),
        removal_poles=np.asarray(
            removal_poles[removal_mask],
            dtype=np.float64,
        ),
        removal_weights=np.asarray(
            removal_weights[removal_mask],
            dtype=np.float64,
        ),
    )


def hubbard_lehmann_spectrum(
    L: int,
    n_up: int,
    n_down: int,
    U: float,
    k: float,
    spin: str = "up",
    t: float = 1.0,
    mu: float | None = None,
    periodic: bool = True,
    weight_cutoff: float = 1e-14,
) -> LehmannSpectrum:
    if mu is None:
        mu = 0.5 * U

    ground_basis = HubbardBasis(
        L=L,
        n_up=n_up,
        n_down=n_down,
    )

    if spin == "up":
        addition_basis = HubbardBasis(
            L=L,
            n_up=n_up + 1,
            n_down=n_down,
        )
        removal_basis = HubbardBasis(
            L=L,
            n_up=n_up - 1,
            n_down=n_down,
        )
    elif spin == "down":
        addition_basis = HubbardBasis(
            L=L,
            n_up=n_up,
            n_down=n_down + 1,
        )
        removal_basis = HubbardBasis(
            L=L,
            n_up=n_up,
            n_down=n_down - 1,
        )
    else:
        raise ValueError("spin must be 'up' or 'down'")

    H_ground = hubbard_hamiltonian(
        ground_basis,
        t=t,
        U=U,
        periodic=periodic,
    )

    H_addition = hubbard_hamiltonian(
        addition_basis,
        t=t,
        U=U,
        periodic=periodic,
    )

    H_removal = hubbard_hamiltonian(
        removal_basis,
        t=t,
        U=U,
        periodic=periodic,
    )

    ground_energy, ground_vector = ground_state(H_ground)

    addition_energies, addition_vectors = full_eigensystem(
        H_addition
    )

    removal_energies, removal_vectors = full_eigensystem(
        H_removal
    )

    creation_operator = momentum_creation_operator(
        ground_basis,
        addition_basis,
        k=k,
        spin=spin,
    )

    annihilation_operator = momentum_annihilation_operator(
        ground_basis,
        removal_basis,
        k=k,
        spin=spin,
    )

    return lehmann_spectrum(
        ground_energy=ground_energy,
        ground_vector=ground_vector,
        addition_energies=addition_energies,
        addition_vectors=addition_vectors,
        removal_energies=removal_energies,
        removal_vectors=removal_vectors,
        creation_operator=creation_operator,
        annihilation_operator=annihilation_operator,
        mu=mu,
        weight_cutoff=weight_cutoff,
    )


def lorentzian(
    omega: np.ndarray,
    center: float,
    eta: float,
) -> np.ndarray:
    if eta <= 0.0:
        raise ValueError("eta must be positive")

    return (
        eta
        / np.pi
        / ((omega - center) ** 2 + eta**2)
    )


def broaden_spectrum(
    omega: np.ndarray,
    poles: np.ndarray,
    weights: np.ndarray,
    eta: float,
) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64)
    poles = np.asarray(poles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if poles.shape != weights.shape:
        raise ValueError("poles and weights must have the same shape")

    spectrum = np.zeros_like(omega)

    for pole, weight in zip(poles, weights):
        spectrum += weight * lorentzian(
            omega,
            center=pole,
            eta=eta,
        )

    return spectrum
