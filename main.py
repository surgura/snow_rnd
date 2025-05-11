from pathlib import Path
from matplotlib import pyplot as plt
from src.read_data import read_data
import numpy as np
from remove_coherent_noise import boxcar_filter, tukey_filter


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    data, gps_time = read_data("Data_img_01_20170410_01_001")

    data = 10 * np.log10(data)
    # data = data[:, :100]

    # red_overlay = np.zeros((*data.shape, 3))
    # red_overlay[..., 0] = 1

    # blue_overlay = np.zeros((*data.shape, 3))
    # blue_overlay[..., 2] = 1

    boxcar, boxcar_noise = boxcar_filter(data)
    # tukey, tukey_noise = tukey_filter(data)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(data.T, aspect="auto", cmap="gray", interpolation="none")
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(boxcar.T, aspect="auto", cmap="gray", interpolation="none")
    axes[1, 0].set_title("Cleaned")
    axes[1, 1].imshow(boxcar_noise.T, aspect="auto", cmap="gray", interpolation="none")
    axes[1, 1].set_title("Coherent noise")
    # axes[2, 0].imshow(
    #     10 * np.log10(tukey.T), aspect="auto", cmap="gray", interpolation="none"
    # )
    # axes[2, 1].imshow(
    #     10 * np.log10(tukey_noise.T), aspect="auto", cmap="gray", interpolation="none"
    # )
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
