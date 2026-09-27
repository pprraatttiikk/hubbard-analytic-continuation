import csv
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hubbard_ac.data.torch_dataset import (
    HubbardObservationDataset,
)
from hubbard_ac.models.losses import (
    AnalyticContinuationLoss,
)
from hubbard_ac.models.mlp import HubbardMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def move_batch(
    batch: dict,
    device: torch.device,
):
    return {
        "g_input": batch["g_input"].to(
            device,
            non_blocking=True,
        ),
        "a_target": batch["a_target"].to(
            device,
            non_blocking=True,
        ),
        "conditioning": batch["conditioning"].to(
            device,
            non_blocking=True,
        ),
    }


def run_epoch(
    model,
    criterion,
    loader,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_spectrum = 0.0
    total_green = 0.0
    total_samples = 0

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            prediction = model(
                batch["g_input"],
                batch["conditioning"],
            )

            losses = criterion(
                prediction=prediction,
                target=batch["a_target"],
                g_input=batch["g_input"],
                beta=batch["conditioning"][:, 1],
            )

            if training:
                losses["loss"].backward()
                optimizer.step()

        n = batch["g_input"].shape[0]

        total_samples += n

        total_loss += (
            losses["loss"].item()
            * n
        )

        total_spectrum += (
            losses["spectrum_loss"].item()
            * n
        )

        total_green += (
            losses["green_loss"].item()
            * n
        )

    return {
        "loss": total_loss / total_samples,
        "spectrum_loss": (
            total_spectrum / total_samples
        ),
        "green_loss": (
            total_green / total_samples
        ),
    }


def train_model(
    config: dict,
    epochs_override: int | None = None,
):
    data_config = config["data"]
    model_config = config["model"]
    loss_config = config["loss"]
    training_config = config["training"]
    output_config = config["output"]

    seed = int(
        training_config["seed"]
    )

    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training"
        )

    device = torch.device("cuda")

    data_path = Path(
        data_config["path"]
    )

    output_dir = Path(
        output_config["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = (
        HubbardObservationDataset(
            data_path,
            "train",
        )
    )

    validation_dataset = (
        HubbardObservationDataset(
            data_path,
            "validation",
        )
    )

    batch_size = int(
        training_config["batch_size"]
    )

    num_workers = int(
        training_config["num_workers"]
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )

    validation_loader = make_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )

    omega = torch.from_numpy(
        train_dataset.omega
    )

    tau_fraction = torch.from_numpy(
        train_dataset.tau_fraction
    )

    with h5py.File(
        data_path,
        "r",
    ) as f:
        beta_values = torch.from_numpy(
            f["grids/beta"][:]
        ).float()

    model = HubbardMLP(
        omega=omega,
        n_tau=len(
            train_dataset.tau_fraction
        ),
        hidden_dim=int(
            model_config["hidden_dim"]
        ),
        parameter_dim=int(
            model_config["parameter_dim"]
        ),
        dropout=float(
            model_config["dropout"]
        ),
        u_max=float(
            model_config["u_max"]
        ),
        beta_max=float(
            model_config["beta_max"]
        ),
    ).to(device)

    criterion = (
        AnalyticContinuationLoss(
            omega=omega,
            tau_fraction=tau_fraction,
            beta_values=beta_values,
            lambda_spectrum=float(
                loss_config[
                    "lambda_spectrum"
                ]
            ),
            lambda_green=float(
                loss_config[
                    "lambda_green"
                ]
            ),
        ).to(device)
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(
            training_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
    )

    epochs = int(
        training_config["epochs"]
    )

    if epochs_override is not None:
        epochs = int(
            epochs_override
        )

    patience = int(
        training_config["patience"]
    )

    min_delta = float(
        training_config["min_delta"]
    )

    best_validation = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    start_time = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            criterion=criterion,
            loader=validation_loader,
            device=device,
            optimizer=None,
        )

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_spectrum_loss": (
                train_metrics[
                    "spectrum_loss"
                ]
            ),
            "train_green_loss": (
                train_metrics[
                    "green_loss"
                ]
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_spectrum_loss": (
                validation_metrics[
                    "spectrum_loss"
                ]
            ),
            "validation_green_loss": (
                validation_metrics[
                    "green_loss"
                ]
            ),
            "seconds": elapsed,
        }

        history.append(record)

        print(
            f"epoch={epoch:4d} "
            f"train={record['train_loss']:.6e} "
            f"val={record['validation_loss']:.6e} "
            f"A={record['validation_spectrum_loss']:.6e} "
            f"G={record['validation_green_loss']:.6e} "
            f"time={elapsed:.2f}s"
        )

        if (
            validation_metrics["loss"]
            < best_validation
            - min_delta
        ):
            best_validation = (
                validation_metrics[
                    "loss"
                ]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_loss": (
                        best_validation
                    ),
                    "config": config,
                },
                output_dir
                / "best_model.pt",
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "early stopping"
            )
            break

    total_time = (
        time.perf_counter()
        - start_time
    )

    history_path = (
        output_dir
        / "history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)

    summary = {
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation
        ),
        "total_seconds": total_time,
        "parameter_count": sum(
            p.numel()
            for p in model.parameters()
        ),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(0)
        ),
    }

    with open(
        output_dir
        / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    return summary
