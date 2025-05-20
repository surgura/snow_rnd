import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey
from pathlib import Path
from read_data import read_data
import matplotlib.pyplot as plt


def boxcar_filter(
    data: npt.NDArray,
    sampling_interval: float,
    cutoff_period: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply a boxcar (moving average) low-pass filter to remove low-frequency coherent noise.

    Parameters
    ----------
    data (along_track, range_bins)
        Input radar data array.
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
    data : ndarray of shape (along_track, range_bins)
        Input radar data array.
    sampling_interval : float, optional
        Time between samples along track, in seconds.
    cutoff_period : float, optional
        Period corresponding to the desired cutoff frequency.
    alpha : float, optional
        Shape parameter for the Tukey window (0 = boxcar, 1 = Hann).

    Returns
    -------
    filtered_data
        High-frequency component of the data after removing low-frequency trends.
    coherent_noise : ndarray
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


def main() -> None:
    # create results dir
    Path("results").mkdir(exist_ok=True)

    # load data
    data, gps_time = read_data()
    # data = (data - data.min()) / (data.max() - data.min())

    # estimate sampling interval
    sampling_interval = (
        gps_time[len(gps_time) // 2 + 1] - gps_time[len(gps_time) // 2]
    ).item()
    cutoff_period = 1

    # apply filters
    boxcar, boxcar_noise = boxcar_filter(
        data=data, sampling_interval=sampling_interval, cutoff_period=cutoff_period
    )
    tukey, tukey_noise = tukey_filter(
        data, sampling_interval=sampling_interval, cutoff_period=cutoff_period
    )

    # save data
    np.save("results/coh_clean_boxcar.npy", boxcar)
    np.save("results/coh_noise_boxcar.npy", boxcar_noise)
    np.save("results/coh_clean_tukey.npy", tukey)
    np.save("results/coh_noise_tukey.npy", tukey_noise)

    # plot data and save plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    # axes[0, 0].imshow(
    #     10 * np.log10(data.T), aspect="auto", cmap="gray", interpolation="none"
    # )
    # axes[0, 0].set_title("Original")
    # axes[0, 1].plot(tukey[1700])
    # axes[0, 1].set_title("Cleaned signal @ 1700")
    # axes[0, 1].imshow(tukey_noise.T, aspect="auto", cmap="gray", interpolation="none")
    # axes[0, 1].set_title("Clean (by paper)")
    axes[0, 0].plot(data[1700])
    axes[0, 0].set_title("Original signal @ 1700")
    axes[0, 1].plot(tukey[1700])
    axes[0, 1].set_title("Cleaned signal @ 1700")
    axes[1, 0].imshow(
        # 10 * np.log10(tukey).T,
        # np.log10(np.clip(np.abs(tukey.T), 1e-10, None)),
        20 * np.log10(np.abs(tukey.T)),
        aspect="auto",
        cmap="gray",
        interpolation="none",
    )
    axes[1, 0].set_title("Clean (Tukey)")
    axes[1, 1].imshow(
        20 * np.log10(np.abs(tukey_noise.T)),
        aspect="auto",
        cmap="gray",
        interpolation="none",
    )
    axes[1, 1].set_title("Noise (Tukey)")
    fig.savefig("results/coh_tukey.png")
    plt.show()


if __name__ == "__main__":
    main()
