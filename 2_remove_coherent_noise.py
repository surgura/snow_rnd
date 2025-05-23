import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import xarray as xr


def boxcar_filter(
    data: npt.NDArray,
    sampling_interval: float,
    cutoff_period: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply a boxcar (moving average) low-pass filter to remove low-frequency coherent noise.

    Parameters
    ----------
    data
        Input radar data array.
        shape(along_track, range_bins)
    sampling_interval
        Time between samples along track, in seconds.
    cutoff_period
        Period corresponding to the desired cutoff frequency, in seconds.

    Returns
    -------
    filtered_data
        High-frequency component of the data after removing low-frequency trends.
    coherent_noise
        Low-frequency component (moving average) estimated from the data.

    Notes
    -----
    This filter uses a uniform boxcar kernel (equal weights) and reflects
    data at the edges to handle boundaries. The filter length is rounded from
    cutoff_period / sampling_interval and adjusted to be odd (adding one).
    """
    return tukey_filter(
        data=data,
        sampling_interval=sampling_interval,
        cutoff_period=cutoff_period,
        alpha=0,
    )


def tukey_filter(
    data: npt.NDArray,
    sampling_interval: float,
    cutoff_period: float,
    alpha: float = 0.5,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply a Tukey-windowed FIR low-pass filter to remove low-frequency coherent noise.

    Parameters
    ----------
    data
        Input radar data array.
        shape (along_track, range_bins)
    sampling_interval
        Time between samples along track, in seconds.
    cutoff_period
        Period corresponding to the desired cutoff frequency.
    alpha
        Shape parameter for the Tukey window (0 = boxcar, 1 = Hann).

    Returns
    -------
    filtered_data
        High-frequency component of the data after removing low-frequency trends.
    coherent_noise
        Low-frequency component estimated using a Tukey-windowed FIR filter.

    Notes
    -----
    The Tukey window provides smoother spectral roll-off and lower side lobes than
    a simple boxcar, making it more effective at isolating low-frequency structure
    without introducing artifacts. The filter length is determined by
    cutoff_period / sampling_interval and adjusted to be odd (adding one).
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D (range_bins x along_track)")

    dt = sampling_interval
    fcut = 1 / cutoff_period
    filt_len = int(round(1 / (fcut * dt)))
    if filt_len % 2 == 0:
        filt_len += 1

    window = tukey(filt_len, alpha)
    window /= np.sum(window)
    coherent_noise = convolve1d(data, window, axis=0, mode="reflect")
    filtered_data = data - coherent_noise

    return filtered_data, coherent_noise


def remove_coh_boxcar(data: xr.Dataset) -> xr.Dataset:
    # estimate sampling interval
    sampling_interval = (
        data.gps_time[len(data.gps_time) // 2 + 1]
        - data.gps_time[len(data.gps_time) // 2]
    ).item()
    cutoff_period = 1

    clean, noise = boxcar_filter(
        data=data.power.values,
        sampling_interval=sampling_interval,
        cutoff_period=cutoff_period,
    )
    gps_time = data.gps_time.data

    return xr.Dataset(
        data_vars=dict(
            power_no_coh=(["sample_number", "time"], clean),
            power_coh=(["sample_number", "time"], noise),
            gps_time=(["sample_number"], gps_time),
        ),
        coords=dict(time=("time", data.time.data)),
        attrs=dict(description=f"{data.description}_coh_boxcar"),
    )


def main() -> None:
    raw_data = xr.open_datatree("results/data.zarr")

    results = xr.DataTree()
    for transect_name, transect in raw_data.items():
        print(f"Removing coherent noise using boxcar filter for {transect_name}")
        no_coh = xr.DataTree(
            remove_coh_boxcar(transect.dataset), name=f"{transect_name}_coh=boxcar"
        )
        no_coh.to_zarr(f"results/intermediate_coh_boxcar/{transect_name}")
        results[transect_name] = no_coh
    results.to_zarr("results/coh_boxcar.zarr")


if __name__ == "__main__":
    main()
