import h5py
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def load_file_into_xr(file: str) -> xr.Dataset:
    with h5py.File(f"data/{file}", "r") as f:
        power = f["Data"][()]
        time = f["Time"][()].squeeze()
        gps_time = f["GPS_time"][()].squeeze()

    # Build dataset
    return xr.Dataset(
        data_vars=dict(
            power=(["sample_number", "time"], power),
            gps_time=(["sample_number"], gps_time),
        ),
        coords=dict(time=("time", time)),
        attrs=dict(description=f"IceBird dataset {file}"),
    ).transpose()


def find_common_time(datasets, atol=1e-12):
    times = [ds.coords["time"].values for ds in datasets]
    ref_time = times[0]

    for t in times[1:]:
        # Try to align ref_time and t by finding the first exact match
        for i in range(len(ref_time)):
            sub_ref = ref_time[i:]
            if len(sub_ref) > len(t):
                sub_ref = sub_ref[: len(t)]
            else:
                t = t[: len(sub_ref)]
            if np.allclose(sub_ref, t, atol=atol):
                ref_time = sub_ref
                break
        else:
            return None  # No alignment found
    return ref_time


def concat_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
    # Estimate dt from the first dataset
    dt = float(chunks[0].time[1] - chunks[0].time[0])

    # Get global time bounds
    t_min = min(ds.time.values[0] for ds in chunks)
    t_max = max(ds.time.values[-1] for ds in chunks)

    # 3. Create unified "truth" time array
    truth_time = truth_time = np.arange(t_min, t_max + dt * 0.5, dt)

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

    cc = xr.concat(padded_dsets, dim="sample_number")

    fix2, axs = plt.subplots(
        2,
    )
    cc.power.pipe(lambda da: 20 * np.log10(np.abs(da))).plot(ax=axs[0])
    # chunks[0].power.pipe(np.isnan).plot(ax=axs[0], add_colorbar=False)
    # chunks[1].power.pipe(np.isnan).plot(ax=axs[1], add_colorbar=False)
    # chunks[1].power.fillna(0).plot(ax=ax)
    # ((chunks[1].power == 0) | chunks[1].power.isnull()).plot(ax=ax)
    # chunks[1].power.where(chunks[1].power != 0).pipe(np.isnan).sum("time").plot(
    #     ax=axs[0]
    # )
    # chunks[1].power.where(chunks[1].power != 0).pipe(np.isnan).plot(
    #     ax=axs[1], add_colorbar=False
    # )
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()

    plt.show()
    # plt.show()

    return chunks[0]  # TODO


def load_dataset() -> xr.Dataset:
    xr.DataTree(
        None,
        {
            f"day_{day:0>2}": xr.DataTree(
                concat_chunks(
                    [
                        load_file_into_xr(
                            f"Data_img_{day:0>2}_20170410_01_{chunk_i:0>3}.mat"
                        )
                        for chunk_i in range(1, 4)
                    ]
                )
            )
            for day in range(1, 2)
        },
    )


load_dataset()
