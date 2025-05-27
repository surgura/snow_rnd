import xarray as xr
import numpy as np


def align(data: xr.Dataset) -> xr.Dataset:
    """
    Return a new Dataset with aligned waveforms based on maximum power.
    NaN-padded, skips alignment for all-NaN waveforms.
    Crops to fixed 201-sample window. Keeps all sample_number entries.
    """
    power = data["power_no_coh"]
    is_all_nan = power.isnull().all(dim="time")
    safe_power = power.where(~is_all_nan, 0)
    max_indices = safe_power.argmax(dim="time")
    center_index = power.sizes["time"] // 2
    shifts = center_index - max_indices

    def safe_shift(arr, shift, is_valid):
        if not is_valid:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        if shift > 0:
            result[shift:] = arr[:-shift]
        elif shift < 0:
            result[:shift] = arr[-shift:]
        else:
            result[:] = arr
        return result

    aligned = xr.apply_ufunc(
        safe_shift,
        power,
        shifts,
        ~is_all_nan,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[power.dtype],
    )

    # Centered 201-sample window
    window_radius = 100
    time = aligned["time"]
    center_index = time.size // 2
    start = center_index - window_radius
    end = center_index + window_radius + 1  # +1 to make it 201 samples
    aligned_window = aligned.isel(time=slice(start, end))

    # Construct output dataset with shared coords
    return xr.Dataset(
        {"power_aligned": aligned_window},
        coords={
            k: (v if "time" not in v.dims else v.isel(time=slice(start, end)))
            for k, v in data.coords.items()
        },
        attrs=dict(description=f"{data.description}_aligned"),
    )


# def align(data: xr.Dataset) -> xr.Dataset:
#     """
#     Return a new Dataset with aligned waveforms based on maximum power.
#     NaN-padded, skips all-NaN waveforms. No use of .copy().
#     """
#     power = data["power_no_coh"]
#     is_all_nan = power.isnull().all(dim="time")
#     safe_power = power.where(~is_all_nan, 0)
#     max_indices = safe_power.argmax(dim="time")
#     center_index = power.sizes["time"] // 2
#     shifts = center_index - max_indices

#     def safe_shift(arr, shift, is_valid):
#         if not is_valid:
#             return np.full_like(arr, np.nan)
#         result = np.full_like(arr, np.nan)
#         if shift > 0:
#             result[shift:] = arr[:-shift]
#         elif shift < 0:
#             result[:shift] = arr[-shift:]
#         else:
#             result[:] = arr
#         return result

#     aligned = xr.apply_ufunc(
#         safe_shift,
#         power,
#         shifts,
#         ~is_all_nan,
#         input_core_dims=[["time"], [], []],
#         output_core_dims=[["time"]],
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[power.dtype],
#     )

#     window_radius = 100  # for a total width of 201
#     time = aligned["time"]
#     center_index = time.size // 2
#     start = center_index - window_radius
#     end = center_index + window_radius + 1  # +1 to include endpoint, making width = 201

#     aligned_window = aligned.isel(time=slice(start, end))

#     # Construct new dataset with shared references + aligned output
#     return xr.Dataset(
#         {"power_aligned": aligned_window},
#         coords=data.coords,
#         attrs=dict(description=f"{data.description}_aligned"),
#     )


def main() -> None:
    for name in ["boxcar", "tukey"]:
        dt = xr.open_datatree(f"results/2_coh_{name}.zarr")
        aligned = xr.DataTree()
        for polarization_name, polarization in dt.items():
            aligned[f"{polarization_name}"] = align(polarization.dataset)
        aligned.to_zarr(f"results/3_aligned_coh_{name}.zarr")


if __name__ == "__main__":
    main()
