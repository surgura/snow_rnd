import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def main() -> None:
    data = xr.open_datatree("results/3_aligned_coh_boxcar.zarr")
    fig, axes = plt.subplots(
        nrows=len(data),
        # figsize=(3840 / 100, 2160 / 100),
        # dpi=100,
        constrained_layout=True,
    )
    for ax, (ds_name, ds) in zip(
        [axes] if len(data) == 1 else axes.flatten(), data.items()
    ):
        ds.dataset.transpose().power_aligned.pipe(
            lambda x: np.abs(20 * np.log10(x))
        ).plot(ax=ax)
        ax.set_title(ds_name)
    fig.savefig("results/3_aligned_coh_boxcar_power.png")


if __name__ == "__main__":
    main()
