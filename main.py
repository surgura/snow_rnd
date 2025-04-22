import numpy as np
from scipy.signal.windows import gaussian
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyrpca import rpca_pcp_ialm


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
        matrix, sparsity_factor=lmbda_factor * lmbda, verbose=False
    )
    return sparse[:, matrix.shape[1] // 2]


def main() -> None:
    window_width = 15

    data = np.load("data/data.npy")
    # data = data[:, :100]
    rows, cols = data.shape

    denoised = sliding_columnwise_transform(
        data, window_width=window_width, function=denoise
    )

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True)
    axes[0].imshow(data, aspect="auto", cmap="gray")
    axes[1].imshow(denoised, aspect="auto", cmap="gray")

    y = np.zeros(cols)
    start = cols // 2 - window_width // 2
    y[start : start + window_width] = (
        1.0  # gaussian(window_width, window_width / 6, sym=True)
    )
    axes[2].plot(y, color="black")

    plt.show()


if __name__ == "__main__":
    main()
