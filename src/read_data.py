import numpy.typing as npt
import h5py


def read_data(name: str) -> npt.ArrayLike:
    with h5py.File(f"data/{name}.mat", "r") as f:
        data = f["Data"][()]
    return data
