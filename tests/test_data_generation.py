import h5py
import numpy as np

from hubbard_ac.data.generate import (
    TEST,
    TRAIN,
    VALIDATION,
    generate_dataset,
    parameter_grid,
    split_for_u,
)


def test_parameter_grid():
    values = parameter_grid(
        0.0,
        1.0,
        0.25,
    )

    assert np.allclose(
        values,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )


def test_split_assignment():
    validation = [2.5, 7.5]
    test = [3.0, 6.0, 9.0]

    assert split_for_u(
        1.0,
        validation,
        test,
    ) == TRAIN

    assert split_for_u(
        2.5,
        validation,
        test,
    ) == VALIDATION

    assert split_for_u(
        6.0,
        validation,
        test,
    ) == TEST


def test_small_dataset(tmp_path):
    output = tmp_path / "test.h5"

    config = {
        "system": {
            "L": 4,
            "n_up": 2,
            "n_down": 2,
            "t": 1.0,
            "periodic": True,
            "spin": "up",
        },
        "parameters": {
            "U": {
                "start": 0.0,
                "stop": 2.0,
                "step": 2.0,
            },
            "beta": [
                5.0,
                10.0,
            ],
        },
        "grid": {
            "omega_min": -8.0,
            "omega_max": 8.0,
            "n_omega": 401,
            "n_tau": 40,
            "eta": 0.1,
        },
        "noise": {
            "sigma": [
                0.0,
                1.0e-3,
            ],
            "realizations": {
                "clean": 1,
                "noisy": 3,
            },
        },
        "random": {
            "seed": 123,
        },
        "split": {
            "validation_U": [],
            "test_U": [2.0],
        },
        "output": {
            "path": str(output),
        },
    }

    manifest = generate_dataset(
        config,
        output_path=output,
    )

    assert manifest["n_spectra"] == 8
    assert manifest["n_physical"] == 16
    assert manifest["n_observations"] == 64

    with h5py.File(output, "r") as file:
        assert file["spectra/A"].shape == (
            8,
            401,
        )

        assert file["physical/G_clean"].shape == (
            16,
            40,
        )

        assert file["observations/G_input"].shape == (
            64,
            40,
        )

        spectrum_ids = file[
            "physical/spectrum_id"
        ][:]

        splits = file[
            "physical/split"
        ][:]

        U = file["spectra/U"][:]

        for physical_id, spectrum_id in enumerate(
            spectrum_ids
        ):
            if np.isclose(U[spectrum_id], 2.0):
                assert splits[physical_id] == TEST
            else:
                assert splits[physical_id] == TRAIN


def test_noise_copies_do_not_cross_splits(tmp_path):
    output = tmp_path / "test.h5"

    config = {
        "system": {
            "L": 4,
            "n_up": 2,
            "n_down": 2,
            "t": 1.0,
            "periodic": True,
            "spin": "up",
        },
        "parameters": {
            "U": {
                "start": 0.0,
                "stop": 1.0,
                "step": 1.0,
            },
            "beta": [5.0],
        },
        "grid": {
            "omega_min": -8.0,
            "omega_max": 8.0,
            "n_omega": 201,
            "n_tau": 20,
            "eta": 0.1,
        },
        "noise": {
            "sigma": [
                0.0,
                1.0e-3,
            ],
            "realizations": {
                "clean": 1,
                "noisy": 5,
            },
        },
        "random": {
            "seed": 7,
        },
        "split": {
            "validation_U": [],
            "test_U": [1.0],
        },
        "output": {
            "path": str(output),
        },
    }

    generate_dataset(
        config,
        output_path=output,
    )

    with h5py.File(output, "r") as file:
        observation_physical = file[
            "observations/physical_id"
        ][:]

        physical_splits = file[
            "physical/split"
        ][:]

        for physical_id in np.unique(
            observation_physical
        ):
            mask = (
                observation_physical
                == physical_id
            )

            observation_splits = (
                physical_splits[
                    observation_physical[mask]
                ]
            )

            assert np.all(
                observation_splits
                == observation_splits[0]
            )


def test_generated_spectra_are_normalized(tmp_path):
    output = tmp_path / "normalized.h5"

    config = {
        "system": {
            "L": 4,
            "n_up": 2,
            "n_down": 2,
            "t": 1.0,
            "periodic": True,
            "spin": "up",
        },
        "parameters": {
            "U": {
                "start": 0.0,
                "stop": 2.0,
                "step": 2.0,
            },
            "beta": [5.0],
        },
        "grid": {
            "omega_min": -8.0,
            "omega_max": 8.0,
            "n_omega": 801,
            "n_tau": 40,
            "eta": 0.1,
        },
        "noise": {
            "sigma": [0.0],
            "realizations": {
                "clean": 1,
                "noisy": 1,
            },
        },
        "random": {
            "seed": 123,
        },
        "split": {
            "validation_U": [],
            "test_U": [],
        },
        "output": {
            "path": str(output),
        },
    }

    generate_dataset(
        config,
        output_path=output,
    )

    with h5py.File(output, "r") as file:
        omega = file["grids/omega"][:]
        spectra = file["spectra/A"][:]

        integrals = np.trapezoid(
            spectra,
            x=omega,
            axis=1,
        )

        assert np.allclose(
            integrals,
            1.0,
            atol=1e-12,
        )
