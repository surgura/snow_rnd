import numpy as np
from scipy.signal.windows import gaussian
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyrpca import rpca_pcp_ialm
from sklearn.preprocessing import MinMaxScaler


def sliding_columnwise_transform(matrix, window_width, function):
    """
    Applies a sliding gaussian window across columns of a 2D matrix, transforming each window
    with a custom function. Uses reflective boundary conditions.

    Args:
        matrix: Input 2D array of shape (rows, columns)
        window_width: Width of the sliding window
        function: Function to apply to each window (shape: [rows, window_width])
                         Must return a 1D array of shape (rows,)
        stddev: Standard deviation of gaussian window

    Returns:
        ndarray: Transformed matrix of shape (rows, columns)
    """
    rows, cols = matrix.shape
    stddev = window_width / 6
    pad = window_width // 2
    padded = np.pad(matrix, ((0, 0), (pad, pad)), mode="reflect")
    gaus = gaussian(window_width, stddev, sym=True)[None, :]  # shape (1, window_width)

    result = np.empty((rows, cols))

    for c in range(cols):  # [80:120]:
        print(f"Column: {c} / {cols}")
        window = padded[:, c : c + window_width]  # shape (rows, window_width)
        # window = window * gaus
        result[:, c] = function(window)  # func should return shape (rows,)

    return result


def denoise(matrix: npt.ArrayLike) -> npt.ArrayLike:
    lmbda_factor = 1.5
    lmbda = 1.0 / np.sqrt(max(matrix.shape))
    low_rank, sparse = rpca_pcp_ialm(
        matrix,
        sparsity_factor=lmbda_factor * lmbda,
        verbose=True,
    )
    return sparse


def main() -> None:
    makevis = lambda d: 10 * np.log10(d)

    data = np.load("data/AWI_SR_array4.npy")
    mask = np.isnan(data) | (data < 1e-12)
    data = 10 * np.log10(data)
    data[mask] = np.nan
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    cutoff = 100
    data = data[:, :cutoff]
    mask = mask[:, :cutoff]

    denoised = denoise(np.nan_to_num(data, nan=0.0))
    # normed = np.clip(denoised, a_min=0.0, a_max=None) / np.max(denoised)

    # denoised = sliding_columnwise_transform(
    #     data, window_width=window_width, function=denoise
    # )

    # denoised_normed = (denoised - denoised.min()) / (denoised.max() - denoised.min())

    red_overlay = np.zeros((*denoised.shape, 3))
    red_overlay[..., 0] = 1

    rangemin = 4600
    rangemax = 5400
    snapshot = 86
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(data, aspect="auto", cmap="gray")
    axes[0, 0].imshow(
        red_overlay, alpha=mask.astype(np.float32), aspect="auto", zorder=1
    )
    axes[0, 1].plot(data[rangemin:rangemax, snapshot])
    axes[1, 0].imshow(denoised, aspect="auto", cmap="gray")
    axes[1, 0].imshow(
        red_overlay, alpha=mask.astype(np.float32), aspect="auto", zorder=1
    )
    axes[1, 1].plot(denoised[rangemin:rangemax, snapshot])

    plt.show()


if __name__ == "__main__":
    main()
