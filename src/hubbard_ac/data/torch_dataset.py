from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


class HubbardObservationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        split: str,
    ):
        if split not in SPLIT_CODES:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'"
            )

        self.path = Path(path)
        self.split = split

        with h5py.File(self.path, "r") as f:
            observation_physical = f[
                "observations/physical_id"
            ][:]

            physical_split = f[
                "physical/split"
            ][:]

            observation_split = physical_split[
                observation_physical
            ]

            global_indices = np.flatnonzero(
                observation_split
                == SPLIT_CODES[split]
            )

            self.global_indices = global_indices.astype(
                np.int64
            )

            self.g_input = f[
                "observations/G_input"
            ][global_indices].astype(
                np.float32
            )

            self.noise_sigma = f[
                "observations/sigma"
            ][global_indices].astype(
                np.float32
            )

            self.realization = f[
                "observations/realization"
            ][global_indices].astype(
                np.int32
            )

            self.physical_id = observation_physical[
                global_indices
            ].astype(
                np.int64
            )

            physical_spectrum = f[
                "physical/spectrum_id"
            ][:]

            physical_beta = f[
                "physical/beta"
            ][:]

            self.spectrum_id = physical_spectrum[
                self.physical_id
            ].astype(
                np.int64
            )

            self.beta = physical_beta[
                self.physical_id
            ].astype(
                np.float32
            )

            spectra_u = f[
                "spectra/U"
            ][:]

            spectra_k = f[
                "spectra/k"
            ][:]

            spectra_k_index = f[
                "spectra/k_index"
            ][:]

            self.u = spectra_u[
                self.spectrum_id
            ].astype(
                np.float32
            )

            self.k = spectra_k[
                self.spectrum_id
            ].astype(
                np.float32
            )

            self.k_index = spectra_k_index[
                self.spectrum_id
            ].astype(
                np.int32
            )

            self.a_all = f[
                "spectra/A"
            ][:].astype(
                np.float32
            )

            self.omega = f[
                "grids/omega"
            ][:].astype(
                np.float32
            )

            self.tau_fraction = f[
                "grids/tau_fraction"
            ][:].astype(
                np.float32
            )

        self.conditioning = np.column_stack(
            (
                self.u,
                self.beta,
                self.k,
                self.noise_sigma,
            )
        ).astype(
            np.float32
        )

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, index: int) -> dict:
        sid = self.spectrum_id[index]

        return {
            "g_input": torch.from_numpy(
                self.g_input[index]
            ),
            "a_target": torch.from_numpy(
                self.a_all[sid]
            ),
            "conditioning": torch.from_numpy(
                self.conditioning[index]
            ),
            "physical_id": int(
                self.physical_id[index]
            ),
            "spectrum_id": int(sid),
            "observation_id": int(
                self.global_indices[index]
            ),
            "realization": int(
                self.realization[index]
            ),
            "k_index": int(
                self.k_index[index]
            ),
        }
