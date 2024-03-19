import os
import sys
import pickle
import random
import omegaconf
import sympy as sp
import numpy as np
from sympy import lambdify
from typing import List, Any
from src.utils import get_project_root
from dataclasses import dataclass, field
from src.EquationLearning.Data.FeynmanReader import FeynmanReader
from src.EquationLearning.Data.data_utils import bounded_operations
from src.EquationLearning.Transformers.GenerateTransformerData import skeleton2dataset, Dataset, modify_constants_avoidNaNs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_exclude(lo, hi, k, exclude_low, exclude_high):
    sample = []
    while len(sample) < k:
        x = random.uniform(lo, hi)
        if not exclude_low <= x <= exclude_high:
            sample.append(x)
    return np.array(sample)


@dataclass
class InputData:
    """Collects data in the format required by the MSSP solver"""
    X: np.array
    Y: np.array
    names: List[str]
    types: List[str]
    expr: str
    limits: Any = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self):
        # Shuffle dataset
        indexes = np.arange(len(self.X))
        np.random.seed(7)
        np.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.Y = self.Y[indexes]

        self.n_features = self.X.shape[1]
        # Calculate limits of each variable
        if self.X.ndim == 1:
            self.limits = (np.min(self.X), np.max(self.X))
        else:
            self.limits = [(np.min(self.X[:, xdim]), np.max(self.X[:, xdim])) for xdim in range(self.X.shape[1])]


class DataLoader:
    """Class used to load or generate datasets used for equation learning"""

    def __init__(self, name=None, extrapolation: bool = False):
        """
        Initialize DataLoader class
        :param name: Dataset name (If known, otherwise create a new temporal dataset)
        :param extrapolation: If True, generate extrapolation data
        """
        self.X, self.Y, self.names = np.zeros(0), np.zeros(0), None
        self.expr, self.cfg, self.types = None, None, []
        self.extrapolation = extrapolation

        if "U" in name or "E" in name or "CS" in name:
            self.modelType = "NN"
            if "CS" in name or name in ['E1', 'E4', 'E5', 'E9']:
                self.modelType = "NN3"
            if hasattr(self, f'{name}'):
                method = getattr(self, f'{name}')
                method()
            else:
                sys.exit('The provided dataset name does not exist')
        else:  # If not one of the previous names, it probably comes from the Feynman database
            try:
                Freader = FeynmanReader(name)
                self.X, self.Y, self.names, self.expr = Freader.X, Freader.Y, Freader.names, Freader.expr
                self.modelType = "NN"
            except FileNotFoundError:
                sys.exit('The provided dataset name does not exist')

        self.dataset = InputData(X=self.X, Y=self.Y, names=self.names, types=self.types,
                                 expr=self.expr)

    def E1(self, n=10000):  # S4 in "Informed Equation Learning" (Werner et. al, 2021)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = sample_exclude(-10, 10, n, -5, 5)
            x2 = sample_exclude(-10, 10, n, -5, 5)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = (3.0375 * x1 * x2 + 5.5 * np.sin(9/4 * (x1 - 2/3) * (x2 - 2/3))) / 5
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = (3.0375 * symb[0] * symb[1] + 5.5 * sp.sin(9/4 * (symb[0] - 2/3) * (symb[1] - 2/3))) / 5

    def E2(self, n=10000):  # Ours
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(-10, 10, size=n)
            x3 = np.random.uniform(-15, 15, size=n)
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
            x2 = sample_exclude(-20, 20, n, -10, 10)
            x3 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1, x2, x3]).T
        # Calculate output
        self.Y = 5.5 + (1 - x1 / 4) ** 2 + (np.sqrt(x2 + 10)) * np.sin(x3 / 5)
        self.names = ['x0', 'x1', 'x2']
        self.types = ['continuous', 'continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 3))
        self.expr = 5.5 + (1 - symb[0] / 4) ** 2 + (sp.sqrt(symb[1] + 10)) * sp.sin(symb[2] / 5)

    def E3(self, n=10000):  # A1 in "Informed Equation Learning" (Werner et. al, 2021)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-3, 3, size=n)
            x2 = np.random.uniform(-3, 3, size=n)
        else:
            x1 = sample_exclude(-8, 8, n, -5, 5)
            x2 = sample_exclude(-8, 8, n, -5, 5)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = (1.5 * np.exp(1.5 * x1) + 5 * np.cos(3 * x2))/10
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = (1.5 * sp.exp(1.5 * symb[0]) + 5 * sp.cos(3 * symb[1]))/10

    def E4(self, n=50000):  # Rosenbrock 4-D
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
            x3 = np.random.uniform(-5, 5, size=n)
            x4 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = sample_exclude(-10, 10, n, -5, 5)
            x2 = sample_exclude(-10, 10, n, -5, 5)
            x3 = sample_exclude(-10, 10, n, -5, 5)
            x4 = sample_exclude(-10, 10, n, -5, 5)
        self.X = np.array([x1, x2, x3, x4]).T
        # Calculate output
        self.Y = ((1 - x1) ** 2 + (1 - x3) ** 2 + 100 * (x2 - x1 ** 2) ** 2 + 100 * (x4 - x3 ** 2) ** 2) / 10000
        self.names = ['x0', 'x1', 'x2', 'x3']
        self.types = ['continuous', 'continuous', 'continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 4))
        self.expr = ((1 - symb[0]) ** 2 + (1 - symb[2]) ** 2 + 100 * (symb[1] - symb[0] ** 2) ** 2 +
                     100 * (symb[3] - symb[2] ** 2) ** 2) / 10000

    def E5(self, n=10000):  # Ours
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-3, 3, size=n)
            x3 = np.random.uniform(-3, 3, size=n)
            x4 = np.random.uniform(-3, 3, size=n)
        else:
            x1 = sample_exclude(-10, 10, n, -5, 5)
            x2 = sample_exclude(-6, 6, n, -3, 3)
            x3 = sample_exclude(-6, 6, n, -3, 3)
            x4 = sample_exclude(-6, 6, n, -3, 3)
        self.X = np.array([x1, x2, x3, x4]).T
        # Calculate output
        self.Y = np.sin(x1 + x2 * x3) + np.exp(1.2 * x4)
        self.names = ['x0', 'x1', 'x2', 'x3']
        self.types = ['continuous', 'continuous', 'continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 4))
        self.expr = sp.sin(symb[0] + symb[1] * symb[2]) + sp.exp(1.2 * symb[3])

    def E6(self, n=50000):  # Ours
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(-10, 10, size=n)
            x3 = np.random.uniform(-10, 10, size=n)
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
            x2 = sample_exclude(-20, 20, n, -10, 10)
            x3 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1, x2, x3]).T
        # Calculate output
        self.Y = np.tanh(x1 / 2) + (np.abs(x2)) * np.cos(x3 ** 2 / 5)
        self.names = ['x0', 'x1', 'x2']
        self.types = ['continuous', 'continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 3))
        self.expr = sp.tanh(symb[0] / 2) + (sp.Abs(symb[1])) * sp.cos(symb[2] ** 2 / 5)

    def E7(self, n=50000):  # S0 in "Informed Equation Learning" paper
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = sample_exclude(-10, 10, n, -5, 5)
            x2 = sample_exclude(-10, 10, n, -5, 5)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = (1 - x2**2) / (np.sin(2 * np.pi * x1) + 1.5)
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = (1 - symb[1]**2) / (sp.sin(2 * np.pi * symb[0]) + 1.5)

    def E8(self, n=50000):  # S5 in "Informed Equation Learning" (Werner et. al, 2021)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-25, 25, size=n)
            x2 = np.random.uniform(-25, 25, size=n)
        else:
            x1 = sample_exclude(-50, 50, n, -25, 25)
            x2 = sample_exclude(-50, 50, n, -25, 25)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = x1**4 / (x1**4 + 1) + x2**4 / (x2**4 + 1)
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = (symb[0])**4 / ((symb[0])**4 + 1) + (symb[1])**4 / ((symb[1])**4 + 1)

    def E9(self, n=50000):  # A2 in "Informed Equation Learning" (Werner et. al, 2021)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(0, 5, size=n)
        else:
            x1 = np.random.uniform(0, 10, size=n)
            x2 = np.random.uniform(0, 10, size=n)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = np.log(2 * x2 + 1) - np.log(4 * x1 ** 2 + 1)
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = sp.log(2 * symb[1] + 1) - sp.log(4 * symb[0] ** 2 + 1)

    def CS1(self, n=50000):  # CS1 in " The Metric is the Message" (Bertschinger et al., 2023)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-2, 2, size=n)
            range_values = np.linspace(-3, 3, 100)
            x2 = [np.random.choice(range_values) for _ in range(n)]
        else:
            x1 = np.random.uniform(-4, 4, size=n)
            range_values = np.linspace(-6, 6, 200)
            x2 = [np.random.choice(range_values) for _ in range(n)]
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = np.sin(x1 * np.exp(x2))
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'discrete']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = sp.sin(symb[0] * sp.exp(symb[1]))

    def CS2(self, n=50000):  # CS2 in " The Metric is the Message" (Bertschinger et al., 2023)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(-10, 10, size=n)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = x1 * np.log(x2 ** 4)
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = symb[0] * sp.log(symb[1] ** 4)

    def CS3(self, n=50000):  # CS3 in " The Metric is the Message" (Bertschinger et al., 2023)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(-10, 10, size=n)
        else:
            x1 = np.random.uniform(-20, 20, size=n)
            x2 = np.random.uniform(-20, 20, size=n)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = 1 + x1 * np.sin(1 / x2)
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = 1 + symb[0] * sp.sin(1 / symb[1])

    def CS4(self, n=50000):  # CS4 in " The Metric is the Message" (Bertschinger et al., 2023)
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(0, 20, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = np.random.uniform(0, 40, size=n)
            x2 = np.random.uniform(-10, 10, size=n)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = np.sqrt(x1) * np.log(x2 ** 2)
        self.names = ['x0', 'x1']
        self.types = ['continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = sp.sqrt(symb[0]) * sp.log(symb[1] ** 2)

    def EX1(self, n=50000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
            x3 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = sample_exclude(-10, 10, n, -5, 5)
            x2 = sample_exclude(-10, 10, n, -5, 5)
            x3 = sample_exclude(-10, 10, n, -5, 5)
        self.X = np.array([x1, x2, x3]).T
        # Calculate output
        self.Y = (np.abs(x3)) / x1 * np.exp(-1 / 2 * (x2 / x1) ** 2)
        self.names = ['x0', 'x1', 'x2']
        self.types = ['continuous', 'continuous', 'continuous']
        symb = sp.symbols("{}:{}".format('x', 3))
        self.expr = (sp.Abs(symb[2])) / symb[0] * sp.exp(-1 / 2 * (symb[1] / symb[0]) ** 2)

    def EX2(self, n=10000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1]).T
        # Calculate output
        self.Y = 3.2 / (np.sin(2 * np.pi * np.exp(x1 / 5)) - 1.5)
        self.names = ['x0']
        self.types = ['continuous']
        symb = sp.symbols("{}:{}".format('x', 1))
        self.expr = 3.2 / (sp.sin(2 * np.pi * sp.exp(symb[0] / 5)) - 1.5)

    #############################################################################################
    # LOAD DIFFICULT UNIVARIATE PROBLEMS
    #############################################################################################
    def read_cfg(self):
        # Read all equations
        self.cfg = omegaconf.OmegaConf.load(str(get_project_root()) + "/src/EquationLearning/Transformers/config.yaml")
        data_train_path = self.cfg.train_path
        dataset = Dataset(data_train_path, self.cfg.dataset_train, mode="train")
        word2id = dataset.word2id
        self.cfg = self.cfg.dataset_train
        return word2id

    def loadGeneratedData(self, filepath, skeleton, xrange):
        # If the file with the sampled data already exists, just load it
        if os.path.exists(filepath) and False:
            with open(filepath, 'rb') as file:
                self.X, self.Y, _, skeleton, expr = pickle.load(file)
        else:  # Else, generate the data from the target skeleton
            word2id = self.read_cfg()
            self.X, self.Y, _, skeleton, expr = skeleton2dataset(skeleton, xrange, self.cfg, word2id)
            self.X = self.X[:, None]
            # Save the data
            with open(filepath, 'wb') as f:
                pickle.dump([self.X, self.Y, _, skeleton, expr], f)
        self.expr = sp.sympify(str(expr).replace('x_1', 'x0'))
        self.names = ['x0']

    def U1(self):
        np.random.seed(7)
        filepath = str(get_project_root()) + '/src/EquationLearning/Data/univariate_datasets/U1.pkl'
        skeleton = 'c*x_1*log(c*x_1**2 + c) + c'
        self.loadGeneratedData(filepath, skeleton, xrange=[-10, 10])

    def U2(self):
        np.random.seed(1)
        filepath = str(get_project_root()) + '/src/EquationLearning/Data/univariate_datasets/U2.pkl'
        skeleton = 'c*(c*x_1 + c)**2*sin(c*x_1) + c'
        self.loadGeneratedData(filepath, skeleton, xrange=[-10, 10])

    def U3(self):
        np.random.seed(1)
        # filepath = str(get_project_root()) + '/src/EquationLearning/Data/univariate_datasets/U2.pkl'
        self.expr = sp.sympify('(x_1 + 1)**2/sin(x_1*x_2)')
        # Generate data by fixing values of x2
        xcount = 0
        xx = np.linspace(np.pi/2 - 1.3, np.pi/2 + 1.3, 30)
        xx2 = np.linspace(np.pi*3/2 - 1.3, np.pi*3/2 + 1.3, 30)
        xx3 = np.linspace(-np.pi/2 - 1.3, -np.pi/2 + 1.3, 30)
        xx = np.concatenate((xx, xx2, xx3))
        self.X, self.Y = np.zeros((1000 * len(xx), 2)), np.zeros((1000 * len(xx),))
        for v in xx:
            if np.abs(v) > 0.1:
                temp_expr = self.expr.subs(sp.sympify('x_2'), v)
                x0 = np.linspace(-10, 10, 1000)
                x, new_xp, _ = modify_constants_avoidNaNs(temp_expr, x0[None, :], bounded_operations(), 1000, [-10, 10],
                                                          [sp.sympify('x_1')], threshold=0.05)
                # Evaluate
                if (np.isnan(x)).any():
                    indices_to_remove = np.where(np.isnan(x[0, :]))[0]
                    x = np.delete(x, indices_to_remove)[None, :]
                function = lambdify(sp.sympify('x_1'), temp_expr)
                y = np.array(function(x))
                self.X[xcount:xcount + x.shape[1], 0] = x[0, :]
                self.X[xcount:xcount + x.shape[1], 1] = v
                self.Y[xcount:xcount + x.shape[1], ] = y[0, :]
                xcount += x.shape[1]
        self.X, self.Y = self.X[0:xcount, :], self.Y[0:xcount]
        self.expr = sp.sympify(str(self.expr).replace('x_1', 'x0'))
        self.expr = sp.sympify(str(self.expr).replace('x_2', 'x1'))
        self.names = ['x0', 'x1']


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    dataLoader = DataLoader(name='E1', extrapolation=False)
    X, Y, var_names = dataLoader.X, dataLoader.Y, dataLoader.names
    plt.scatter(X, Y)
