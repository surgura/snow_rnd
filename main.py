import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pyrpca import rpca_pcp_ialm
import cvxpy as cp
from scipy.linalg import svd
from pyproximal import L1
import torch
from pathlib import Path
import os


def denoise(matrix: npt.ArrayLike) -> npt.ArrayLike:
    lmbda_factor = 1.5
    lmbda = 1.0 / np.sqrt(max(matrix.shape))
    low_rank, sparse = rpca_pcp_ialm(
        matrix,
        sparsity_factor=lmbda_factor * lmbda,
        verbose=True,
    )
    return sparse


def find_window(matrix: npt.ArrayLike) -> tuple[float, float]:
    # lambda_ = 1.0  # penalize window size
    # gamma = 10.0  # penalize transitions (TV)

    # w = cp.Variable(matrix.shape[0])
    # diag = cp.diag(1 - w)

    # nuclear = cp.norm(diag @ matrix, "nuc")
    # size_penalty = cp.sum(w)
    # # tv_penalty = cp.norm1(w[1:] - w[:-1])

    # objective = cp.Minimize(nuclear - lambda_ * size_penalty)
    # constraints = [w >= 0, w <= 1]

    # problem = cp.Problem(objective, constraints)
    # problem.solve(verbose=True)
    # # problem.solve(solver=cp.SCS, gp=False, enforce_dpp=False)

    # print(w.value)

    lambda_: float = 0.025
    lr: float = 0.1
    epochs: int = 30  # 300
    epsilon: float = 1e-3
    tv_weight: float = 0.1

    X = torch.tensor(matrix, dtype=torch.float32)
    n_rows = X.shape[0]

    # Initialize soft mask
    w = torch.ones(n_rows, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()

        # Apply soft mask
        Xw = torch.diag(w) @ X

        # Smoothed nuclear norm
        s = torch.linalg.svdvals(Xw)
        smooth_nuclear = torch.sum(torch.sqrt(s**2 + epsilon))

        # Total Variation on inverted mask (1 - w)
        inverted = 1 - w
        tv = torch.sum(torch.abs(inverted[1:] - inverted[:-1]))

        # Total loss: nuclear norm - lambda * sum(w) + TV on (1 - w)
        loss = smooth_nuclear - lambda_ * torch.sum(w) + tv_weight * tv
        print(loss.item())
        loss.backward()
        optimizer.step()

        # Clip w to [0, 1]
        with torch.no_grad():
            w.clamp_(0, 1)

    return 1.0 - w.detach().numpy() > 0.2


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    data = np.load("data/AWI_SR_array4.npy")
    mask = np.isnan(data) | (data < 1e-12)
    data = 10 * np.log10(data)
    data[mask] = np.nan
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    cutoff = 50
    data = data[:, :cutoff]
    mask = mask[:, :cutoff]

    if os.path.isfile("results/signal_mask.npy"):
        window_mask = np.load("results/signal_mask.npy")
    else:
        window_mask = np.zeros_like(data)

        width = 9
        pad = width // 2
        data_padded = np.pad(data, ((0, 0), (pad, pad)), mode="reflect")

        for i in range(data.shape[1]):
            print(f"Window {i}")
            window = find_window(np.nan_to_num(data_padded, nan=0.0)[:, i : i + width])
            window_mask[:, i] = window

        np.save("results/signal_mask.npy", window_mask)

    # mask min/max
    first = np.argmax(window_mask, axis=0)
    last = window_mask.shape[0] - 1 - np.argmax(window_mask[::-1], axis=0)
    signal_indices = np.stack([first - 5, last + 5], axis=1)
    for i, (start, end) in enumerate(signal_indices):
        window_mask[start:end, i] = 1

    red_overlay = np.zeros((*data.shape, 3))
    red_overlay[..., 0] = 1

    blue_overlay = np.zeros((*data.shape, 3))
    blue_overlay[..., 2] = 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(data, aspect="auto", cmap="gray", interpolation="none")
    axes[0, 0].imshow(
        red_overlay,
        alpha=mask.astype(np.float32),
        aspect="auto",
        zorder=1,
        interpolation="none",
    )
    axes[0, 0].imshow(
        blue_overlay,
        alpha=window_mask.astype(np.float32),
        aspect="auto",
        zorder=1,
        interpolation="none",
    )
    axes[1, 0].imshow(data, aspect="auto", cmap="gray", interpolation="none")

    idx = 40
    indices = np.where(window_mask[:, idx] == 1)[0]
    mindex = indices[0] - 5
    maxdex = indices[-1] + 5
    axes[0, 1].plot(data[mindex:maxdex, idx])
    axes[0, 1].set_title(f"signal at x={idx}")
    plt.show()


if __name__ == "__main__":
    main()
