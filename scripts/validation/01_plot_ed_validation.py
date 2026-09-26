import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hubbard_ac.physics.greens import greens_from_poles
from hubbard_ac.physics.sector_operators import momentum_grid
from hubbard_ac.physics.spectra import (
    broaden_spectrum,
    hubbard_lehmann_spectrum,
)
from hubbard_ac.plotting.style import use_paper_style


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "figures" / "validation"
RESULT_DIR = ROOT / "results" / "validation"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

use_paper_style()


def free_dispersion_validation():
    L = 6
    k_values = momentum_grid(L)

    exact = -2.0 * np.cos(k_values)

    reconstructed = []
    variances = []
    sum_rule_errors = []

    for k in k_values:
        spectrum = hubbard_lehmann_spectrum(
            L=L,
            n_up=3,
            n_down=3,
            U=0.0,
            k=k,
            spin="up",
            t=1.0,
            mu=0.0,
            periodic=True,
        )

        mean_energy = np.sum(
            spectrum.weights * spectrum.poles
        )

        variance = np.sum(
            spectrum.weights
            * (spectrum.poles - mean_energy) ** 2
        )

        reconstructed.append(mean_energy)
        variances.append(variance)
        sum_rule_errors.append(
            abs(spectrum.total_weight - 1.0)
        )

    reconstructed = np.asarray(reconstructed)

    fig, ax = plt.subplots(figsize=(3.35, 2.65))

    dense_k = np.linspace(
        0.0,
        2.0 * np.pi,
        500,
    )

    ax.plot(
        dense_k,
        -2.0 * np.cos(dense_k),
        label=r"$-2t\cos k$",
    )

    ax.scatter(
        k_values,
        reconstructed,
        zorder=3,
        label="ED",
    )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega/t$")

    ax.set_xticks(
        [
            0.0,
            np.pi / 2.0,
            np.pi,
            3.0 * np.pi / 2.0,
            2.0 * np.pi,
        ]
    )

    ax.set_xticklabels(
        [
            r"$0$",
            r"$\pi/2$",
            r"$\pi$",
            r"$3\pi/2$",
            r"$2\pi$",
        ]
    )

    ax.legend()

    fig.savefig(
        FIGURE_DIR / "free_dispersion.pdf"
    )

    plt.close(fig)

    return {
        "free_dispersion_max_error": float(
            np.max(np.abs(reconstructed - exact))
        ),
        "free_dispersion_max_variance": float(
            np.max(variances)
        ),
        "free_dispersion_max_sum_rule_error": float(
            np.max(sum_rule_errors)
        ),
    }


def interacting_spectrum():
    L = 6
    U = 4.0
    eta = 0.08

    omega = np.linspace(
        -8.0,
        8.0,
        4001,
    )

    k_values = momentum_grid(L)

    fig, ax = plt.subplots(figsize=(3.35, 2.65))

    for index in [0, 1, 2, 3]:
        k = k_values[index]

        spectrum = hubbard_lehmann_spectrum(
            L=L,
            n_up=3,
            n_down=3,
            U=U,
            k=k,
            spin="up",
            t=1.0,
            mu=U / 2.0,
            periodic=True,
        )

        A = broaden_spectrum(
            omega=omega,
            poles=spectrum.poles,
            weights=spectrum.weights,
            eta=eta,
        )

        label = {
            0: r"$k=0$",
            1: r"$k=\pi/3$",
            2: r"$k=2\pi/3$",
            3: r"$k=\pi$",
        }[index]

        ax.plot(
            omega,
            A,
            label=label,
        )

    ax.axvline(
        0.0,
        linewidth=0.7,
        linestyle="--",
    )

    ax.set_xlabel(r"$\omega/t$")
    ax.set_ylabel(r"$A(k,\omega)$")
    ax.set_xlim(-8.0, 8.0)

    ax.legend(
        ncol=2,
        handlelength=1.8,
    )

    fig.savefig(
        FIGURE_DIR / "interacting_spectra_U4.pdf"
    )

    plt.close(fig)


def imaginary_time_green_function():
    L = 6
    U = 4.0
    beta = 20.0

    tau = np.linspace(
        0.0,
        beta,
        241,
    )

    k_values = momentum_grid(L)

    endpoint_errors = []
    sum_rule_errors = []

    fig, ax = plt.subplots(figsize=(3.35, 2.65))

    for index in [0, 1, 2, 3]:
        k = k_values[index]

        spectrum = hubbard_lehmann_spectrum(
            L=L,
            n_up=3,
            n_down=3,
            U=U,
            k=k,
            spin="up",
            t=1.0,
            mu=U / 2.0,
            periodic=True,
        )

        G = greens_from_poles(
            tau=tau,
            beta=beta,
            poles=spectrum.poles,
            weights=spectrum.weights,
        )

        endpoint_errors.append(
            abs(G[0] + G[-1] + 1.0)
        )

        sum_rule_errors.append(
            abs(spectrum.total_weight - 1.0)
        )

        label = {
            0: r"$k=0$",
            1: r"$k=\pi/3$",
            2: r"$k=2\pi/3$",
            3: r"$k=\pi$",
        }[index]

        ax.plot(
            tau / beta,
            G,
            label=label,
        )

    ax.set_xlabel(r"$\tau/\beta$")
    ax.set_ylabel(r"$G(k,\tau)$")

    ax.legend(
        ncol=2,
        handlelength=1.8,
    )

    fig.savefig(
        FIGURE_DIR / "imaginary_time_G_U4.pdf"
    )

    plt.close(fig)

    return {
        "U4_max_endpoint_error": float(
            np.max(endpoint_errors)
        ),
        "U4_max_sum_rule_error": float(
            np.max(sum_rule_errors)
        ),
    }


def particle_hole_validation():
    L = 6
    U = 4.0

    k_values = momentum_grid(L)

    max_weight_error = 0.0

    for index in range(L):
        k = k_values[index]
        q = k_values[(index + L // 2) % L]

        spectrum_k = hubbard_lehmann_spectrum(
            L=L,
            n_up=3,
            n_down=3,
            U=U,
            k=k,
            spin="up",
            t=1.0,
            mu=U / 2.0,
            periodic=True,
        )

        spectrum_q = hubbard_lehmann_spectrum(
            L=L,
            n_up=3,
            n_down=3,
            U=U,
            k=q,
            spin="up",
            t=1.0,
            mu=U / 2.0,
            periodic=True,
        )

        error = abs(
            np.sum(spectrum_k.addition_weights)
            - np.sum(spectrum_q.removal_weights)
        )

        max_weight_error = max(
            max_weight_error,
            error,
        )

    return {
        "particle_hole_max_weight_error": float(
            max_weight_error
        )
    }


def main():
    results = {}

    results.update(
        free_dispersion_validation()
    )

    interacting_spectrum()

    results.update(
        imaginary_time_green_function()
    )

    results.update(
        particle_hole_validation()
    )

    with open(
        RESULT_DIR / "ed_validation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    for key, value in results.items():
        print(f"{key}: {value:.6e}")


if __name__ == "__main__":
    main()
