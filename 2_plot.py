import xarray as xr
import matplotlib.pyplot as plt


def plot_denoised_waveform(sample_number, threshold):
    ds = xr.open_datatree("results/coh_boxcar.zarr", engine="zarr")["transect_01"].ds

    fig, axes = plt.subplots(
        nrows=5, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True
    )

    # create a loop iterating over all axes flattened, and a list of sample numbers, doing the code below
    # for ax, sample_number in zip(axes.flatten(), [2000, 2001, 2002, 2003, 2004]):
    for ax, sample_number in zip(axes.flatten(), [4600, 4601, 4602, 4603, 4604]):
        waveform = ds.power_no_coh.isel(sample_number=sample_number)

        # create a window around the peak
        n = waveform.sizes["time"]
        peak_idx = waveform.argmax("time").item()
        half_window = int(n * 0.01)

        start = max(0, peak_idx - half_window)
        end = min(n, peak_idx + half_window + 1)

        cropped = waveform.isel(time=slice(start, end))

        # plot window
        cropped_rel = cropped.assign_coords(time=cropped.time - cropped.time[0])
        cropped_rel.plot(ax=ax, label=f"Sample {sample_number}")
        # cropped.plot(ax=ax, label=f"Sample {sample_number}")
        ax.set_title(f"Sample {sample_number} (Zoomed Around Peak)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()
        fig.savefig("results/2_coh_boxcar.png", dpi=300, bbox_inches="tight")


def main():
    plot_denoised_waveform(sample_number=1000, threshold=0.6e-7)


if __name__ == "__main__":
    main()
