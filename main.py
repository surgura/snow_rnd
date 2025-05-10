from pathlib import Path
from matplotlib import pyplot as plt
from src.read_data import read_data
import numpy as np


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    data = read_data("Data_img_01_20170410_01_001")

    # red_overlay = np.zeros((*data.shape, 3))
    # red_overlay[..., 0] = 1

    # blue_overlay = np.zeros((*data.shape, 3))
    # blue_overlay[..., 2] = 1

    data = data.T
    data = 10 * np.log10(data)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(data, aspect="auto", cmap="gray", interpolation="none")
    # axes[0, 0].imshow(
    #     red_overlay,
    #     alpha=mask.astype(np.float32),
    #     aspect="auto",
    #     zorder=1,
    #     interpolation="none",
    # )
    # axes[0, 0].imshow(
    #     blue_overlay,
    #     alpha=window_mask.astype(np.float32),
    #     aspect="auto",
    #     zorder=1,
    #     interpolation="none",
    # )
    # axes[1, 0].imshow(data, aspect="auto", cmap="gray", interpolation="none")

    # axes[0, 1].plot(sample_signal)
    # axes[0, 1].set_title(f"signal at x={idx}")
    # axes[1, 1].plot(sample_denoised_signal)
    # axes[1, 1].set_title(f"denoised signal at x={idx}")
    plt.show()


if __name__ == "__main__":
    main()
