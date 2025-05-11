import numpy.typing as npt
import h5py


def read_data(name: str) -> tuple[npt.NDArray, npt.NDArray]:
    with h5py.File(f"data/{name}.mat", "r") as f:
        data = f["Data"][()]
        gps_time = f["GPS_time"][()]
    return data, gps_time
