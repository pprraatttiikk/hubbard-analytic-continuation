import json
from pathlib import Path

import h5py
import numpy as np

from hubbard_ac.data.noise import gaussian_noise
from hubbard_ac.physics.basis import HubbardBasis
from hubbard_ac.physics.ed import full_eigensystem, ground_state
from hubbard_ac.physics.greens import (
    greens_from_poles,
    greens_from_spectrum,
)
from hubbard_ac.physics.hamiltonian import hubbard_hamiltonian
from hubbard_ac.physics.sector_operators import (
    momentum_annihilation_operator,
    momentum_creation_operator,
    momentum_grid,
)
from hubbard_ac.physics.spectra import (
    broaden_spectrum,
    lehmann_spectrum,
)


TRAIN = 0
VALIDATION = 1
TEST = 2


def parameter_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1)

    if not np.isclose(values[-1], stop):
        raise ValueError("parameter range is not divisible by step")

    return values


def split_for_u(
    U: float,
    validation_u: list[float],
    test_u: list[float],
) -> int:
    if any(np.isclose(U, value) for value in test_u):
        return TEST

    if any(np.isclose(U, value) for value in validation_u):
        return VALIDATION

    return TRAIN


def build_bases(
    L: int,
    n_up: int,
    n_down: int,
    spin: str,
):
    ground = HubbardBasis(
        L=L,
        n_up=n_up,
        n_down=n_down,
    )

    if spin == "up":
        addition = HubbardBasis(
            L=L,
            n_up=n_up + 1,
            n_down=n_down,
        )
        removal = HubbardBasis(
            L=L,
            n_up=n_up - 1,
            n_down=n_down,
        )
    elif spin == "down":
        addition = HubbardBasis(
            L=L,
            n_up=n_up,
            n_down=n_down + 1,
        )
        removal = HubbardBasis(
            L=L,
            n_up=n_up,
            n_down=n_down - 1,
        )
    else:
        raise ValueError("spin must be 'up' or 'down'")

    return ground, addition, removal


def diagonalize_sectors(
    ground_basis,
    addition_basis,
    removal_basis,
    t: float,
    U: float,
    periodic: bool,
):
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

    return (
        ground_energy,
        ground_vector,
        addition_energies,
        addition_vectors,
        removal_energies,
        removal_vectors,
    )


def observation_rng(
    seed: int,
    spectrum_id: int,
    beta_index: int,
    sigma_index: int,
    realization: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            seed,
            spectrum_id,
            beta_index,
            sigma_index,
            realization,
        ]
    )

    return np.random.default_rng(sequence)


def generate_dataset(
    config: dict,
    output_path: str | Path | None = None,
) -> dict:
    system = config["system"]
    parameters = config["parameters"]
    grid = config["grid"]
    noise_config = config["noise"]
    random_config = config["random"]
    split_config = config["split"]

    L = int(system["L"])
    n_up = int(system["n_up"])
    n_down = int(system["n_down"])
    t = float(system["t"])
    periodic = bool(system["periodic"])
    spin = str(system["spin"])

    U_values = parameter_grid(
        float(parameters["U"]["start"]),
        float(parameters["U"]["stop"]),
        float(parameters["U"]["step"]),
    )

    beta_values = np.asarray(
        parameters["beta"],
        dtype=np.float64,
    )

    omega = np.linspace(
        float(grid["omega_min"]),
        float(grid["omega_max"]),
        int(grid["n_omega"]),
    )

    tau_fraction = np.linspace(
        0.0,
        1.0,
        int(grid["n_tau"]),
    )

    eta = float(grid["eta"])

    sigma_values = np.asarray(
        noise_config["sigma"],
        dtype=np.float64,
    )

    clean_realizations = int(
        noise_config["realizations"]["clean"]
    )

    noisy_realizations = int(
        noise_config["realizations"]["noisy"]
    )

    seed = int(random_config["seed"])

    validation_u = [
        float(value)
        for value in split_config["validation_U"]
    ]

    test_u = [
        float(value)
        for value in split_config["test_U"]
    ]

    if output_path is None:
        output_path = config["output"]["path"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k_values = momentum_grid(L)

    n_spectra = len(U_values) * len(k_values)
    n_physical = n_spectra * len(beta_values)

    observations_per_physical = 0

    for sigma in sigma_values:
        if np.isclose(sigma, 0.0):
            observations_per_physical += clean_realizations
        else:
            observations_per_physical += noisy_realizations

    n_observations = (
        n_physical * observations_per_physical
    )

    ground_basis, addition_basis, removal_basis = build_bases(
        L=L,
        n_up=n_up,
        n_down=n_down,
        spin=spin,
    )

    vlen_float = h5py.vlen_dtype(
        np.dtype(np.float64)
    )

    with h5py.File(output_path, "w") as file:
        file.attrs["format_version"] = "1.0"
        file.attrs["L"] = L
        file.attrs["n_up"] = n_up
        file.attrs["n_down"] = n_down
        file.attrs["t"] = t
        file.attrs["periodic"] = periodic
        file.attrs["spin"] = spin
        file.attrs["eta"] = eta
        file.attrs["seed"] = seed

        grids = file.create_group("grids")

        grids.create_dataset(
            "omega",
            data=omega,
        )

        grids.create_dataset(
            "tau_fraction",
            data=tau_fraction,
        )

        grids.create_dataset(
            "U",
            data=U_values,
        )

        grids.create_dataset(
            "beta",
            data=beta_values,
        )

        grids.create_dataset(
            "k",
            data=k_values,
        )

        spectra_group = file.create_group("spectra")

        spectra_group.create_dataset(
            "id",
            shape=(n_spectra,),
            dtype=np.int64,
        )

        spectra_group.create_dataset(
            "U",
            shape=(n_spectra,),
            dtype=np.float64,
        )

        spectra_group.create_dataset(
            "k_index",
            shape=(n_spectra,),
            dtype=np.int32,
        )

        spectra_group.create_dataset(
            "k",
            shape=(n_spectra,),
            dtype=np.float64,
        )

        spectra_group.create_dataset(
            "mu",
            shape=(n_spectra,),
            dtype=np.float64,
        )

        spectra_group.create_dataset(
            "A",
            shape=(n_spectra, len(omega)),
            dtype=np.float64,
            compression="gzip",
            shuffle=True,
        )

        spectra_group.create_dataset(
            "addition_poles",
            shape=(n_spectra,),
            dtype=vlen_float,
        )

        spectra_group.create_dataset(
            "addition_weights",
            shape=(n_spectra,),
            dtype=vlen_float,
        )

        spectra_group.create_dataset(
            "removal_poles",
            shape=(n_spectra,),
            dtype=vlen_float,
        )

        spectra_group.create_dataset(
            "removal_weights",
            shape=(n_spectra,),
            dtype=vlen_float,
        )

        physical_group = file.create_group("physical")

        physical_group.create_dataset(
            "id",
            shape=(n_physical,),
            dtype=np.int64,
        )

        physical_group.create_dataset(
            "spectrum_id",
            shape=(n_physical,),
            dtype=np.int64,
        )

        physical_group.create_dataset(
            "beta",
            shape=(n_physical,),
            dtype=np.float64,
        )

        physical_group.create_dataset(
            "split",
            shape=(n_physical,),
            dtype=np.int8,
        )

        physical_group.create_dataset(
            "G_exact",
            shape=(n_physical, len(tau_fraction)),
            dtype=np.float64,
            compression="gzip",
            shuffle=True,
        )

        physical_group.create_dataset(
            "G_clean",
            shape=(n_physical, len(tau_fraction)),
            dtype=np.float64,
            compression="gzip",
            shuffle=True,
        )

        observation_group = file.create_group("observations")

        observation_group.create_dataset(
            "id",
            shape=(n_observations,),
            dtype=np.int64,
        )

        observation_group.create_dataset(
            "physical_id",
            shape=(n_observations,),
            dtype=np.int64,
        )

        observation_group.create_dataset(
            "sigma",
            shape=(n_observations,),
            dtype=np.float64,
        )

        observation_group.create_dataset(
            "realization",
            shape=(n_observations,),
            dtype=np.int32,
        )

        observation_group.create_dataset(
            "G_input",
            shape=(
                n_observations,
                len(tau_fraction),
            ),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
        )

        spectrum_id = 0
        physical_id = 0
        observation_id = 0

        for U in U_values:
            mu = 0.5 * U

            (
                ground_energy,
                ground_vector,
                addition_energies,
                addition_vectors,
                removal_energies,
                removal_vectors,
            ) = diagonalize_sectors(
                ground_basis=ground_basis,
                addition_basis=addition_basis,
                removal_basis=removal_basis,
                t=t,
                U=U,
                periodic=periodic,
            )

            split = split_for_u(
                U=U,
                validation_u=validation_u,
                test_u=test_u,
            )

            for k_index, k in enumerate(k_values):
                creation_operator = (
                    momentum_creation_operator(
                        ground_basis,
                        addition_basis,
                        k=k,
                        spin=spin,
                    )
                )

                annihilation_operator = (
                    momentum_annihilation_operator(
                        ground_basis,
                        removal_basis,
                        k=k,
                        spin=spin,
                    )
                )

                spectrum = lehmann_spectrum(
                    ground_energy=ground_energy,
                    ground_vector=ground_vector,
                    addition_energies=addition_energies,
                    addition_vectors=addition_vectors,
                    removal_energies=removal_energies,
                    removal_vectors=removal_vectors,
                    creation_operator=creation_operator,
                    annihilation_operator=annihilation_operator,
                    mu=mu,
                )

                A = broaden_spectrum(
                    omega=omega,
                    poles=spectrum.poles,
                    weights=spectrum.weights,
                    eta=eta,
                )

                spectral_weight = np.trapezoid(
                    A,
                    x=omega,
                )

                if spectral_weight <= 0.0:
                    raise RuntimeError(
                        "broadened spectrum has nonpositive weight"
                    )

                A = A / spectral_weight

                spectra_group["id"][spectrum_id] = spectrum_id
                spectra_group["U"][spectrum_id] = U
                spectra_group["k_index"][spectrum_id] = k_index
                spectra_group["k"][spectrum_id] = k
                spectra_group["mu"][spectrum_id] = mu
                spectra_group["A"][spectrum_id] = A

                spectra_group["addition_poles"][
                    spectrum_id
                ] = spectrum.addition_poles

                spectra_group["addition_weights"][
                    spectrum_id
                ] = spectrum.addition_weights

                spectra_group["removal_poles"][
                    spectrum_id
                ] = spectrum.removal_poles

                spectra_group["removal_weights"][
                    spectrum_id
                ] = spectrum.removal_weights

                for beta_index, beta in enumerate(beta_values):
                    tau = beta * tau_fraction

                    G_exact = greens_from_poles(
                        tau=tau,
                        beta=beta,
                        poles=spectrum.poles,
                        weights=spectrum.weights,
                    )

                    G_clean = greens_from_spectrum(
                        tau=tau,
                        beta=beta,
                        omega=omega,
                        spectrum=A,
                    )

                    physical_group["id"][
                        physical_id
                    ] = physical_id

                    physical_group["spectrum_id"][
                        physical_id
                    ] = spectrum_id

                    physical_group["beta"][
                        physical_id
                    ] = beta

                    physical_group["split"][
                        physical_id
                    ] = split

                    physical_group["G_clean"][
                        physical_id
                    ] = G_clean

                    for sigma_index, sigma in enumerate(
                        sigma_values
                    ):
                        if np.isclose(sigma, 0.0):
                            n_realizations = clean_realizations
                        else:
                            n_realizations = noisy_realizations

                        for realization in range(
                            n_realizations
                        ):
                            rng = observation_rng(
                                seed=seed,
                                spectrum_id=spectrum_id,
                                beta_index=beta_index,
                                sigma_index=sigma_index,
                                realization=realization,
                            )

                            G_input = gaussian_noise(
                                signal=G_clean,
                                sigma=sigma,
                                rng=rng,
                            )

                            observation_group["id"][
                                observation_id
                            ] = observation_id

                            observation_group["physical_id"][
                                observation_id
                            ] = physical_id

                            observation_group["sigma"][
                                observation_id
                            ] = sigma

                            observation_group["realization"][
                                observation_id
                            ] = realization

                            observation_group["G_input"][
                                observation_id
                            ] = G_input.astype(
                                np.float32
                            )

                            observation_id += 1

                    physical_id += 1

                spectrum_id += 1

            file.flush()

            print(
                f"U={U:5.2f}  "
                f"spectra={spectrum_id:4d}/{n_spectra}  "
                f"physical={physical_id:5d}/{n_physical}  "
                f"observations={observation_id:6d}/{n_observations}"
            )

    manifest = {
        "format_version": "1.0",
        "path": str(output_path),
        "n_spectra": n_spectra,
        "n_physical": n_physical,
        "n_observations": n_observations,
        "observations_per_physical": observations_per_physical,
        "n_omega": len(omega),
        "n_tau": len(tau_fraction),
        "U_values": U_values.tolist(),
        "beta_values": beta_values.tolist(),
        "k_values": k_values.tolist(),
        "sigma_values": sigma_values.tolist(),
        "eta": eta,
        "seed": seed,
        "split_codes": {
            "train": TRAIN,
            "validation": VALIDATION,
            "test": TEST,
        },
    }

    manifest_path = output_path.with_suffix(
        ".manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    return manifest
