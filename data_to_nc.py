import h5py
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def _load_file_into_xr(file: str) -> xr.Dataset:
    with h5py.File(f"data/{file}", "r") as f:
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
    ).transpose()


def _concat_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
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

        start_idx = np.searchsorted(truth_time, time[0], side="left")
        end_idx = start_idx + len(time)

        # Check only first and last time alignment
        if not (
            abs(truth_time[start_idx] - time[0]) < dt / 2
            and abs(truth_time[end_idx - 1] - time[-1]) < dt / 2
        ):
            raise ValueError(
                "Start or end time does not align with truth_time within tolerance."
            )

        padded_power = np.full((len(truth_time), power.shape[1]), np.nan)
        padded_power[start_idx:end_idx, :] = power

        padded_ds = xr.Dataset(
            data_vars=dict(
                power=(["time", "sample_number"], padded_power),
                gps_time=(["sample_number"], ds.gps_time.values),
            ),
            coords=dict(time=("time", truth_time)),
            attrs=ds.attrs,
        )
        padded_dsets.append(padded_ds)

    # Return concatenated datasets
    return xr.concat(padded_dsets, dim="sample_number")


def load_data_into_datatree() -> xr.DataTree:
    return xr.DataTree(
        None,
        {
            f"day_{day:0>2}": xr.DataTree(
                _concat_chunks(
                    [
                        _load_file_into_xr(
                            f"Data_img_{day:0>2}_20170410_01_{chunk_i:0>3}.mat"
                        )
                        for chunk_i in range(1, 11)
                    ]
                )
            )
            for day in range(1, 2)
        },
    )


def main() -> None:
    tree = load_data_into_datatree()
    tree.to_netcdf("results/data.nc")


if __name__ == "__main__":
    main()
