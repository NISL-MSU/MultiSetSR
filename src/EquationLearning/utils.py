import os
import io
import json
import h5py
import pickle
import marshal
import numpy as np
from typing import Tuple
from pathlib import Path
from sklearn.metrics import r2_score
from EquationLearning.Data import generator
from sklearn.linear_model import LinearRegression
from EquationLearning.Data.dclasses import DatasetDetails, Equation, GeneratorDetails


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def test_linearity(X, Y):
    """Fit linear regression model and calculate R2. If R2 is close to 1, it's a sign of linearity"""
    model = LinearRegression()
    model.fit(X[:, None], Y)
    Y_pred = model.predict(X[:, None])
    return r2_score(Y, Y_pred)


def calc_distance_curves(Ys):
    """Calculate the sum of all pairwise curve distances (mse) in Ys"""
    ncurves = len(Ys)
    total_d = 0
    for i in range(ncurves):
        for j in range(i + 1, ncurves):
            total_d += mse(Ys[i], Ys[j])
    return total_d


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


class RenamedModuleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('src.'):
            module = module[len('src.'):]
        return super().find_class(module, name)


def renamed_module_loads(dt):
    file_like_object = io.BytesIO(dt)
    return RenamedModuleUnpickler(file_like_object).load()


def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx / num_eqs_per_set))
    # if 'train' in str(path_folder):
    #     f = h5py.File('src/EquationLearning/Data/data/training/' + f"{index_file}.h5", 'r')
    # else:
    #     f = h5py.File('src/EquationLearning/Data/data/validation/' + f"{index_file}.h5", 'r')
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    try:
        metadata = pickle.loads(raw_metadata.tobytes())
    except ModuleNotFoundError:
        # Load the pickle data using the custom unpickler
        data_bytes = raw_metadata.tobytes()
        metadata = renamed_module_loads(data_bytes)
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
    try:
        metadata = pickle.loads(raw_metadata.tobytes())
    except ModuleNotFoundError:
        # Load the pickle data using the custom unpickler
        data_bytes = raw_metadata.tobytes()
        metadata = renamed_module_loads(data_bytes)
    return metadata


def create_env(path):
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    env = generator.Generator(param)
    return env, param, d


def tukeyLetters(pp, means=None, alpha=0.05):
    if len(pp.shape) == 1:
        # vector
        G = int(3 + np.sqrt(9 - 4 * (2 - len(pp)))) // 2
        ppp = .5 * np.eye(G)
        ppp[np.triu_indices(G, 1)] = pp
        pp = ppp + ppp.T
    conn = pp > alpha
    G = len(conn)
    if np.all(conn):
        return ['a' for _ in range(G)]
    conns = []
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            if conn[g1, g2]:
                conns.append((g1, g2))

    letters = [[] for _ in range(G)]
    nextletter = 0
    for g in range(G):
        if np.sum(conn[g, :]) == 1:
            letters[g].append(nextletter)
            nextletter += 1
    while len(conns):
        grp = set(conns.pop(0))
        for g in range(G):
            if all(conn[g, np.sort(list(grp))]):
                grp.add(g)
        for g in grp:
            letters[g].append(nextletter)
        for g in grp:
            for h in grp:
                if (g, h) in conns:
                    conns.remove((g, h))
        nextletter += 1

    if means is None:
        means = np.arange(G)
    means = np.array(means)
    groupmeans = []
    for k in range(nextletter):
        ingroup = [g for g in range(G) if k in letters[g]]
        groupmeans.append(means[np.array(ingroup)].mean())
    ordr = np.empty(nextletter, int)
    ordr[np.argsort(groupmeans)] = np.arange(nextletter)
    r = []
    for ltr in letters:
        lst = [chr(97 + ordr[x]) for x in ltr]
        lst.sort()
        r.append(''.join(lst))
    return r
