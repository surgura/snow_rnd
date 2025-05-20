import numpy.typing as npt
import h5py
import numpy as np


def read_data() -> tuple[npt.NDArray, npt.NDArray]:
    base_name = "Data_img_01_20170410_01_"

    data = []
    gps_times = []
    for i in range(1, 3):
        with h5py.File(f"data/{base_name}{i:0>3}.mat", "r") as f:
            data.append(f["Data"][()])
            gps_times.append(f["GPS_time"][()])
    return np.concatenate(data), np.concatenate(gps_times)
