from pathlib import Path

import matplotlib.pyplot as plt


def use_paper_style() -> None:
    root = Path(__file__).resolve().parents[3]
    style = root / "styles" / "paper.mplstyle"
    plt.style.use(style)
