import h5py
import numpy as np
import torch

from hubbard_ac.data.generate import generate_dataset
from hubbard_ac.data.torch_dataset import (
    HubbardObservationDataset,
)


def small_config(path):
    return {
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
                "step": 1.0,
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
                "noisy": 2,
            },
        },
        "random": {
            "seed": 123,
        },
        "split": {
            "validation_U": [1.0],
            "test_U": [2.0],
        },
        "output": {
            "path": str(path),
        },
    }


def test_dataset_split_sizes(tmp_path):
    path = tmp_path / "dataset.h5"

    generate_dataset(
        small_config(path),
        output_path=path,
    )

    train = HubbardObservationDataset(
        path,
        "train",
    )

    validation = HubbardObservationDataset(
        path,
        "validation",
    )

    test = HubbardObservationDataset(
        path,
        "test",
    )

    assert len(train) == 24
    assert len(validation) == 24
    assert len(test) == 24


def test_dataset_shapes_and_types(tmp_path):
    path = tmp_path / "dataset.h5"

    generate_dataset(
        small_config(path),
        output_path=path,
    )

    dataset = HubbardObservationDataset(
        path,
        "train",
    )

    sample = dataset[0]

    assert sample["g_input"].shape == (40,)
    assert sample["a_target"].shape == (401,)
    assert sample["conditioning"].shape == (4,)

    assert sample["g_input"].dtype == torch.float32
    assert sample["a_target"].dtype == torch.float32
    assert sample["conditioning"].dtype == torch.float32


def test_target_matches_spectrum_id(tmp_path):
    path = tmp_path / "dataset.h5"

    generate_dataset(
        small_config(path),
        output_path=path,
    )

    dataset = HubbardObservationDataset(
        path,
        "train",
    )

    with h5py.File(path, "r") as f:
        for i in range(len(dataset)):
            sample = dataset[i]

            expected = f[
                "spectra/A"
            ][sample["spectrum_id"]]

            assert np.allclose(
                sample["a_target"].numpy(),
                expected,
            )


def test_conditioning_matches_metadata(tmp_path):
    path = tmp_path / "dataset.h5"

    generate_dataset(
        small_config(path),
        output_path=path,
    )

    dataset = HubbardObservationDataset(
        path,
        "validation",
    )

    with h5py.File(path, "r") as f:
        for i in range(len(dataset)):
            sample = dataset[i]

            obs_id = sample["observation_id"]
            pid = sample["physical_id"]
            sid = sample["spectrum_id"]

            expected = np.array(
                [
                    f["spectra/U"][sid],
                    f["physical/beta"][pid],
                    f["spectra/k"][sid],
                    f["observations/sigma"][obs_id],
                ],
                dtype=np.float32,
            )

            assert np.allclose(
                sample["conditioning"].numpy(),
                expected,
            )


def test_physical_samples_do_not_cross_splits(tmp_path):
    path = tmp_path / "dataset.h5"

    generate_dataset(
        small_config(path),
        output_path=path,
    )

    train = HubbardObservationDataset(
        path,
        "train",
    )

    validation = HubbardObservationDataset(
        path,
        "validation",
    )

    test = HubbardObservationDataset(
        path,
        "test",
    )

    train_ids = set(train.physical_id.tolist())
    validation_ids = set(validation.physical_id.tolist())
    test_ids = set(test.physical_id.tolist())

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
