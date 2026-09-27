import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from hubbard_ac.plotting.style import use_paper_style


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "hubbard_L6_v2.h5"
FIGURE_DIR = ROOT / "figures" / "validation"
RESULT_DIR = ROOT / "results" / "validation"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

use_paper_style()


def main():
    results = {}

    with h5py.File(DATA, "r") as f:
        omega = f["grids/omega"][:]
        tau_fraction = f["grids/tau_fraction"][:]

        U = f["spectra/U"][:]
        k_index = f["spectra/k_index"][:]
        A = f["spectra/A"][:]

        physical_id = f["physical/id"][:]
        spectrum_id = f["physical/spectrum_id"][:]
        beta = f["physical/beta"][:]
        split = f["physical/split"][:]
        G_clean = f["physical/G_clean"][:]

        observation_id = f["observations/id"][:]
        observation_physical = f["observations/physical_id"][:]
        sigma = f["observations/sigma"][:]
        realization = f["observations/realization"][:]
        G_input = f["observations/G_input"][:]

        assert len(U) == 294
        assert len(physical_id) == 1470
        assert len(observation_id) == 89670

        assert A.shape == (294, 4001)
        assert G_clean.shape == (1470, 240)
        assert G_input.shape == (89670, 240)

        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(G_clean))
        assert np.all(np.isfinite(G_input))

        assert np.all(A >= 0.0)

        split_counts_physical = {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
            "test": int(np.sum(split == 2)),
        }

        observation_split = split[observation_physical]

        split_counts_observations = {
            "train": int(np.sum(observation_split == 0)),
            "validation": int(np.sum(observation_split == 1)),
            "test": int(np.sum(observation_split == 2)),
        }

        assert split_counts_physical == {
            "train": 1320,
            "validation": 60,
            "test": 90,
        }

        assert split_counts_observations == {
            "train": 80520,
            "validation": 3660,
            "test": 5490,
        }

        spectral_integrals = np.trapezoid(
            A,
            x=omega,
            axis=1,
        )

        endpoint_errors = np.empty(len(physical_id))

        for i in range(len(physical_id)):
            endpoint_errors[i] = abs(
                G_clean[i, 0]
                + G_clean[i, -1]
                + 1.0
            )

        noise_std_by_sigma = {}

        for noise_level in np.unique(sigma):
            if np.isclose(noise_level, 0.0):
                continue

            mask = np.isclose(sigma, noise_level)

            parent = observation_physical[mask]

            residual = (
                G_input[mask].astype(np.float64)
                - G_clean[parent]
            )

            measured = float(np.std(residual))

            noise_std_by_sigma[
                f"{noise_level:.0e}"
            ] = measured

        max_clean_error = 0.0

        clean_mask = np.isclose(sigma, 0.0)

        for obs_index in np.where(clean_mask)[0]:
            pid = observation_physical[obs_index]

            error = np.max(
                np.abs(
                    G_input[obs_index].astype(np.float64)
                    - G_clean[pid]
                )
            )

            max_clean_error = max(
                max_clean_error,
                float(error),
            )

        for pid in physical_id:
            mask = observation_physical == pid

            assert np.all(
                split[observation_physical[mask]]
                == split[pid]
            )

        results["n_spectra"] = int(len(U))
        results["n_physical"] = int(len(physical_id))
        results["n_observations"] = int(len(observation_id))

        results["physical_split_counts"] = (
            split_counts_physical
        )

        results["observation_split_counts"] = (
            split_counts_observations
        )

        results["spectral_integral_min"] = float(
            np.min(spectral_integrals)
        )

        results["spectral_integral_mean"] = float(
            np.mean(spectral_integrals)
        )

        results["spectral_integral_max"] = float(
            np.max(spectral_integrals)
        )

        results["max_endpoint_error"] = float(
            np.max(endpoint_errors)
        )

        results["max_clean_storage_error"] = (
            max_clean_error
        )

        results["noise_std"] = noise_std_by_sigma

        fig, ax = plt.subplots(
            figsize=(3.35, 2.65)
        )

        ax.hist(
            spectral_integrals,
            bins=30,
        )

        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_xlabel(
            r"$\int d\omega\,A(k,\omega)$"
        )

        ax.set_ylabel("Count")

        fig.savefig(
            FIGURE_DIR
            / "dataset_spectral_sum_rule.pdf"
        )

        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(3.35, 2.65)
        )

        selected = [
            (0.0, 0),
            (4.0, 1),
            (8.0, 2),
            (12.0, 3),
        ]

        for target_U, target_k in selected:
            matches = np.where(
                np.isclose(U, target_U)
                & (k_index == target_k)
            )[0]

            assert len(matches) == 1

            sid = matches[0]

            ax.plot(
                omega,
                A[sid],
                label=rf"$U/t={target_U:g}$",
            )

        ax.set_xlabel(r"$\omega/t$")
        ax.set_ylabel(r"$A(k,\omega)$")
        ax.set_xlim(
            omega[0],
            omega[-1],
        )
        ax.legend()

        fig.savefig(
            FIGURE_DIR
            / "dataset_example_spectra.pdf"
        )

        plt.close(fig)

        target_U = 4.0
        target_k = 1
        target_beta = 20.0

        sid = np.where(
            np.isclose(U, target_U)
            & (k_index == target_k)
        )[0][0]

        pid = np.where(
            (spectrum_id == sid)
            & np.isclose(beta, target_beta)
        )[0][0]

        fig, ax = plt.subplots(
            figsize=(3.35, 2.65)
        )

        ax.plot(
            tau_fraction,
            G_clean[pid],
            label="Clean",
            linewidth=1.6,
        )

        for noise_level in [
            1.0e-5,
            1.0e-4,
            1.0e-3,
        ]:
            matches = np.where(
                (observation_physical == pid)
                & np.isclose(sigma, noise_level)
                & (realization == 0)
            )[0]

            assert len(matches) == 1

            ax.plot(
                tau_fraction,
                G_input[matches[0]],
                label=rf"$\sigma={noise_level:.0e}$",
                linewidth=0.9,
            )

        ax.set_xlabel(r"$\tau/\beta$")
        ax.set_ylabel(r"$G(k,\tau)$")
        ax.legend()

        fig.savefig(
            FIGURE_DIR
            / "dataset_noise_example.pdf"
        )

        plt.close(fig)

    with open(
        RESULT_DIR / "dataset_validation.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
