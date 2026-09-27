import argparse
from pathlib import Path

import yaml

from hubbard_ac.training.train import train_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    summary = train_model(
        config=config,
        epochs_override=args.epochs,
    )

    print()
    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()
