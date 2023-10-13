import os
import json
import h5py
import pickle
import marshal
import numpy as np
from typing import Tuple
from pathlib import Path
from src.EquationLearning.Data import generator
from src.EquationLearning.Data.dclasses import DatasetDetails, Equation, GeneratorDetails


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def mse(imageA, imageB):
    """Calculate the 'Mean Squared Error' between the two images."""
    return np.mean((imageA - imageB) ** 2)


def normalize(trainx):
    """Normalize and returns the calculated means and stds for each feature"""
    trainxn = trainx.copy()
    if trainx.ndim > 1:
        means = np.zeros((trainx.shape[1], 1))
        stds = np.zeros((trainx.shape[1], 1))
        for n in range(trainx.shape[1]):
            means[n, ] = np.mean(trainxn[:, n])
            stds[n, ] = np.std(trainxn[:, n])
            trainxn[:, n] = (trainxn[:, n] - means[n, ]) / (stds[n, ])
    else:
        means = np.mean(trainxn)
        stds = np.std(trainxn)
        trainxn = (trainxn - means) / stds
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] - means[n, ]) / (stds[n, ])
    else:
        testxn = (testxn - means) / stds
    return testxn


def reversenormalize(testx, means, stds):
    """Reverse normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] * stds) + means
    else:
        testxn = (testxn * stds) + means

    return testxn


def minMaxScale(trainx):
    """Normalize and returns the calculated max and min of the output"""
    trainxn = trainx.copy()
    if np.std(trainxn) < 0.00001:
        trainxn = trainxn * 0
        maxs, mins = 0, 0
    else:
        maxs = np.max(trainxn)
        mins = np.min(trainxn)
        trainxn = (trainxn - mins) / (maxs - mins) * 1
    return trainxn, maxs, mins


def applyMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    testxn = (testxn - mins) / (maxs - mins) * 1
    return testxn


def reverseMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] * (maxs - mins) / 1) + mins
    else:
        testxn = (testxn * (maxs - mins) / 1) + mins

    return testxn


class H5FilesCreator:
    def __init__(self, base_path: Path = None, target_path: Path = None, metadata=None):
        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path

        self.base_path = base_path
        self.metadata = metadata

    def create_single_hd5_from_eqs(self, block):
        name_file, eqs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5"), 'w')
        for i, eq in enumerate(eqs):
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()

    def recreate_single_hd5_from_idx(self, block: Tuple):
        name_file, eq_idxs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5"), 'w')
        for i, eq_idx in enumerate(eq_idxs):
            eq = load_eq_raw(self.base_path, eq_idx, self.metadata.eqs_per_hdf)
            # curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=eq)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)


def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)


def load_eq_raw(path_folder, idx, num_eqs_per_set):
    index_file = str(int(idx / num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    f.close()
    return raw_metadata


def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx / num_eqs_per_set))
    # if 'train' in str(path_folder):
    #     f = h5py.File('src/EquationLearning/Data/data/training/' + f"{index_file}.h5", 'r')
    # else:
    #     f = h5py.File('src/EquationLearning/Data/data/validation/' + f"{index_file}.h5", 'r')
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata


def load_metadata_hdf5(path_folder: Path) -> DatasetDetails:
    # if 'train' in str(path_folder):
    #     f = h5py.File('src/EquationLearning/Data/data/training/' + "metadata.h5", 'r')
    # else:
    #     f = h5py.File('src/EquationLearning/Data/data/validation/' + "metadata.h5", 'r')
    f = h5py.File(os.path.join(path_folder, "metadata.h5"), 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata


def create_env(path):
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    env = generator.Generator(param)
    return env, param, d
