import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey


# def boxcar_manual(
#     data: npt.NDArray,
#     window_len: int = 21,
# ) -> npt.NDArray:
#     """
#     Boxcar (moving average) filter implemented with explicit loops.
#     No convolution, no interpolation, no external dependencies.

#     Parameters
#     ----------
#     data : ndarray, shape (range_bins, along_track)
#         Input radar data.
#     window_len : int
#         Length of the moving average window (must be odd).

#     Returns
#     -------
#     smoothed : ndarray
#         Low-frequency estimate (same shape as input).
#     """
#     # if data.ndim != 2:
#     #     raise ValueError("Input must be 2D (range_bins x along_track)")
#     # if window_len % 2 == 0:
#     #     raise ValueError("window_len must be odd")

#     # pad = window_len // 2
#     # R, T = data.shape

#     # # Pad with edge values for boundary handling
#     # padded = np.pad(data, ((pad, pad), (0, 0)), mode="reflect")

#     # # Cumulative sum
#     # cumsum = np.cumsum(padded, axis=0)
#     # cumsum = np.vstack([np.zeros((1, T)), cumsum])

#     # # Compute moving average with correct alignment
#     # smoothed = (cumsum[window_len:] - cumsum[:-window_len]) / window_len

#     # # Result has shape (R, T)
#     # filtered = data - smoothed
#     # return filtered, smoothed

#     if data.ndim != 2:
#         raise ValueError("Input must be 2D (range_bins x along_track)")
#     if window_len % 2 == 0:
#         raise ValueError("window_len must be odd")

#     pad = window_len // 2
#     padded = np.pad(data, ((pad, pad), (0, 0)), mode="reflect")

#     R, T = data.shape
#     noise = np.empty_like(data)

#     for i in range(R):
#         noise[i] = np.mean(padded[i : i + window_len], axis=0)

#     filtered = data - noise
#     return filtered, noise


def boxcar_filter(
    data: npt.NDArray,
    sampling_interval: float = 7.5,
    cutoff_period: float = 30.0,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply a boxcar (moving average) low-pass filter to remove low-frequency coherent noise.

    Parameters
    ----------
    data : ndarray of shape (range_bins, along_track)
        Input radar data array.
    sampling_interval : float, optional
        Time between samples along track, in seconds (default is 7.5 s).
    cutoff_period : float, optional
        Period corresponding to the desired cutoff frequency, in seconds (default is 30 s).

    Returns
    -------
    filtered_data : ndarray
        High-frequency component of the data after removing low-frequency trends.
    coherent_noise : ndarray
        Low-frequency component (moving average) estimated from the data.

    Notes
    -----
    This filter uses a uniform boxcar kernel (equal weights) and reflects
    data at the edges to handle boundaries. The filter length is rounded from
    cutoff_period / sampling_interval and adjusted to be odd.
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D (range_bins x along_track)")

    filt_len = 15  # int(round(cutoff_period / sampling_interval))
    if filt_len % 2 == 0:
        filt_len += 1

    kernel = np.ones(filt_len) / filt_len
    coherent_noise = convolve1d(data, kernel, axis=0, mode="reflect")
    filtered_data = data - coherent_noise

    return filtered_data, coherent_noise


def tukey_filter(
    data: npt.NDArray,
    sampling_interval: float = 7.5,
    cutoff_period: float = 30.0,
    alpha: float = 0.5,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply a Tukey-windowed FIR low-pass filter to remove low-frequency coherent noise.

    Parameters
    ----------
    data : ndarray of shape (range_bins, along_track)
        Input radar data array.
    sampling_interval : float, optional
        Time between samples along track, in seconds (default is 7.5 s).
    cutoff_period : float, optional
        Period corresponding to the desired cutoff frequency, in seconds (default is 30 s).
    alpha : float, optional
        Shape parameter for the Tukey window (0 = boxcar, 1 = Hann). Default is 0.5.

    Returns
    -------
    filtered_data : ndarray
        High-frequency component of the data after removing low-frequency trends.
    coherent_noise : ndarray
        Low-frequency component estimated using a Tukey-windowed FIR filter.

    Notes
    -----
    The Tukey window provides smoother spectral roll-off and lower side lobes than
    a simple boxcar, making it more effective at isolating low-frequency structure
    without introducing artifacts. The filter length is determined by
    1 / (cutoff_freq * sampling_interval) and adjusted to be odd.
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
