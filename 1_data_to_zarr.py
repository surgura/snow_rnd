import h5py
import xarray as xr
import numpy as np
import numpy.typing as npt


def _load_file_into_xr(file: str) -> xr.Dataset:
    print(f"Reading file {file}")
    with h5py.File(file, "r") as f:
        power = f["Data"][()]
        time = f["Time"][()].squeeze()
        gps_time = f["GPS_time"][()].squeeze()

    return xr.Dataset(
        data_vars=dict(
            power=(["sample_number", "time"], power),
            gps_time=(["sample_number"], gps_time),
        ),
        coords=dict(time=("time", time)),
        attrs=dict(description=f"IceBird dataset {file}"),
    )


def _find_best_index(sorted_array: npt.NDArray, value: float, tol: float) -> int:
    """
    Find the index in `sorted_array` that best matches `value` within `tol`.
    Raises ValueError if no match is found.

    Parameters:
    - sorted_array: 1D array, must be sorted in ascending order.
    - value: target value to match.
    - tol: absolute tolerance for match.

    Returns:
    - index of the closest matching value.
    """
    i = np.searchsorted(sorted_array, value)

    candidates = []
    if i > 0 and abs(sorted_array[i - 1] - value) <= tol:
        candidates.append((abs(sorted_array[i - 1] - value), i - 1))
    if i < len(sorted_array) and abs(sorted_array[i] - value) <= tol:
        candidates.append((abs(sorted_array[i] - value), i))

    if not candidates:
        return None

    _, idx = min(candidates)
    return idx


def _concat_chunks(chunks: list[xr.Dataset], description: str) -> xr.Dataset:
    # Estimate dt from the first dataset
    dt = float(chunks[0].time[1] - chunks[0].time[0])

    # Get global time bounds
    t_min = min(ds.time.values[0] for ds in chunks)
    t_max = max(ds.time.values[-1] for ds in chunks)

    # Create time array
    truth_time = truth_time = np.arange(t_min, t_max + dt * 0.5, dt)

    # For each chunk, match it to the time array with mall tolerance, then pad it with nan.
    # Error if it doesn't align well.
    padded_dsets = []
    for ds in chunks:
        time = ds.time.values
        power = ds.power.values

        start_idx = _find_best_index(truth_time, time[0], tol=dt * 1e-5)
        if start_idx is None:
            raise ValueError(
                f"End time does not align with truth_time within tolerance. {ds.description}"
            )
        end_idx = start_idx + len(time)

        if not (abs(truth_time[end_idx - 1] - time[-1]) < dt * 1e-5):
            raise ValueError(
                f"End time does not align with truth_time within tolerance. {ds.description}"
            )

        padded_power = np.full((power.shape[0], len(truth_time)), np.nan)
        padded_power[:, start_idx:end_idx] = power

        padded_ds = xr.Dataset(
            data_vars=dict(
                power=(["sample_number", "time"], padded_power),
                gps_time=(["sample_number"], ds.gps_time.values),
            ),
            coords=dict(time=("time", truth_time)),
            attrs=ds.attrs,
        )
        padded_dsets.append(padded_ds)

    # Return concatenated datasets
    combined = xr.concat(padded_dsets, dim="sample_number")
    combined.attrs["description"] = description
    return combined


def load_data_into_datatree() -> xr.DataTree:
    return xr.DataTree(
        None,
        {
            f"transect_{transect:0>2}": xr.DataTree(
                _concat_chunks(
                    [
                        _load_file_into_xr(
                            f"data/raw/Data_img_{transect:0>2}_20170410_01_{chunk_i:0>3}.mat"
                        )
                        for chunk_i in range(1, 11)
                    ],
                    description=f"transect_{transect:0>2}",
                )
            )
            for transect in [1, 2, 4]
        },
    )


def main() -> None:
    tree = load_data_into_datatree()
    tree.to_zarr("results/data.zarr")


if __name__ == "__main__":
    main()
