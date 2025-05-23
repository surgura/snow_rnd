import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def main() -> None:
    data = xr.open_datatree("results/data.zarr")
    fig, axes = plt.subplots(
        nrows=len(data),
        figsize=(3840 / 100, 2160 / 100),
        dpi=100,
        constrained_layout=True,
    )
    for ax, (ds_name, ds) in zip(axes.flatten(), data.items()):
        coarse = ds.dataset.coarsen(sample_number=4, time=4, boundary="trim").mean()
        coarse.transpose().power.pipe(lambda x: 20 * np.log10(x)).plot(ax=ax)
        ax.set_title(ds_name)
    fig.savefig("results/data_power.png")


if __name__ == "__main__":
    main()
