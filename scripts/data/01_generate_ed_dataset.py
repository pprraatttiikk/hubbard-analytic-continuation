import argparse
from pathlib import Path

import yaml

from hubbard_ac.data.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    manifest = generate_dataset(
        config=config,
        output_path=args.output,
    )

    print()
    print(f"spectra:      {manifest['n_spectra']}")
    print(f"physical:     {manifest['n_physical']}")
    print(f"observations: {manifest['n_observations']}")


if __name__ == "__main__":
    main()
