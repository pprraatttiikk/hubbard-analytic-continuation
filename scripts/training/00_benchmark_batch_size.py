import time

import h5py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hubbard_ac.data.torch_dataset import HubbardObservationDataset
from hubbard_ac.models.losses import AnalyticContinuationLoss
from hubbard_ac.models.mlp import HubbardMLP


PATH = "data/processed/hubbard_L6_v2.h5"
BATCH_SIZES = [64, 128, 256, 512, 1024]
STEPS = 100
SEED = 1234


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    dataset = HubbardObservationDataset(
        PATH,
        "train",
    )

    omega = torch.from_numpy(
        dataset.omega
    )

    tau_fraction = torch.from_numpy(
        dataset.tau_fraction
    )

    with h5py.File(PATH, "r") as f:
        beta_values = torch.from_numpy(
            f["grids/beta"][:]
        ).float()

    print(
        f"{'batch':>8} "
        f"{'steps/s':>10} "
        f"{'samples/s':>12} "
        f"{'VRAM GiB':>10}"
    )

    for batch_size in BATCH_SIZES:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model = HubbardMLP(
            omega=omega,
        ).to(device)

        criterion = AnalyticContinuationLoss(
            omega=omega,
            tau_fraction=tau_fraction,
            beta_values=beta_values,
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=3.0e-4,
            weight_decay=1.0e-5,
        )

        iterator = iter(loader)

        for _ in range(5):
            batch = next(iterator)

            G = batch["g_input"].to(
                device,
                non_blocking=True,
            )

            A = batch["a_target"].to(
                device,
                non_blocking=True,
            )

            conditioning = batch[
                "conditioning"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                G,
                conditioning,
            )

            losses = criterion(
                prediction=prediction,
                target=A,
                g_input=G,
                beta=conditioning[:, 1],
            )

            losses["loss"].backward()
            optimizer.step()

        torch.cuda.synchronize()

        start = time.perf_counter()
        completed = 0

        for batch in loader:
            if completed >= STEPS:
                break

            G = batch["g_input"].to(
                device,
                non_blocking=True,
            )

            A = batch["a_target"].to(
                device,
                non_blocking=True,
            )

            conditioning = batch[
                "conditioning"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                G,
                conditioning,
            )

            losses = criterion(
                prediction=prediction,
                target=A,
                g_input=G,
                beta=conditioning[:, 1],
            )

            losses["loss"].backward()
            optimizer.step()

            completed += 1

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        steps_per_second = (
            completed / elapsed
        )

        samples_per_second = (
            completed
            * batch_size
            / elapsed
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"{batch_size:8d} "
            f"{steps_per_second:10.2f} "
            f"{samples_per_second:12.0f} "
            f"{peak_vram:10.3f}"
        )

        del model
        del criterion
        del optimizer


if __name__ == "__main__":
    main()
