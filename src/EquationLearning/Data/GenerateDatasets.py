import random
import sympy as sp
import numpy as np
import pandas as pd
from src.utils import get_project_root
from src.EquationLearning.Data.FeynmanReader import FeynmanReader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_exclude(lo, hi, k, exclude_low, exclude_high):
    sample = []
    while len(sample) < k:
        x = random.uniform(lo, hi)
        if not exclude_low <= x <= exclude_high:
            sample.append(x)
    return np.array(sample)


class DataLoader:
    """Class used to load or generate datasets used for equation learning"""

    def __init__(self, name: str = 'S0', extrapolation: bool = False):
        """
        Initialize DataLoader class
        :param name: Dataset name
        :param extrapolation: If True, generate extrapolation data
        """
        self.X, self.Y, self.names = np.zeros(0), np.zeros(0), None
        self.expr = None
        self.extrapolation = extrapolation

        if name == "E1":
            self.E1()
            self.modelType = "NN"
        elif name == "E2":
            self.E2()
            self.modelType = "NN"
        elif name == "E3":
            self.E3()
            self.modelType = "NN"
        elif name == "E4":
            self.E4()
            self.modelType = "NN2"
        elif name == "E5":
            self.E5()
            self.modelType = "NN"
        elif name == "S1":
            self.S1()
            self.modelType = "NN3"
        elif name == "S2":
            self.S2()
            self.modelType = "NN"
        else:  # If not one of the previous names, it probably comes from the Feynman database
            Freader = FeynmanReader(name)
            self.X, self.Y, self.names, self.expr = Freader.X, Freader.Y, Freader.names, Freader.expr
            self.modelType = "NN"

        # Shuffle dataset
        indexes = np.arange(len(self.X))
        np.random.seed(7)
        np.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.Y = self.Y[indexes]

        # Calculate limits of each variable
        if self.X.ndim == 1:
            self.limits = (np.min(self.X), np.max(self.X))
        else:
            self.limits = [(np.min(self.X[:, xdim]), np.max(self.X[:, xdim])) for xdim in range(self.X.shape[1])]

    def S1(self, n=10000):
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
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = (3.0375 * symb[0] * symb[1] + 5.5 * sp.sin(9/4 * (symb[0] - 2/3) * (symb[1] - 2/3))) / 5

    def S2(self, n=10000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(-10, 10, size=n)  # x2 \in [-10, 10]
            x3 = np.random.uniform(-10, 10, size=n)  # x3 \in [-10, 10]
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
            x2 = sample_exclude(-20, 20, n, -10, 10)
            x3 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1, x2, x3]).T
        # Calculate output
        self.Y = 5.5 + (1 - x1 / 4) ** 2 + (np.sqrt(x2 + 10)) * np.sin(x3 / 5)
        self.names = ['x0', 'x1', 'x2']
        symb = sp.symbols("{}:{}".format('x', 3))
        self.expr = 5.5 + (1 - symb[0] / 4) ** 2 + (sp.sqrt(symb[1] + 10)) * sp.sin(symb[2] / 5)

    def E1(self, n=10000):
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
        symb = sp.symbols("{}:{}".format('x', 1))
        self.expr = 3.2 / (sp.sin(2 * np.pi * sp.exp(symb[0] / 5)) - 1.5)

    def E2(self, n=10000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(-10, 10, size=n)  # x2 \in [-10, 10]
            x3 = np.random.uniform(-10, 10, size=n)  # x3 \in [-10, 10]
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
            x2 = sample_exclude(-20, 20, n, -10, 10)
            x3 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1, x2, x3]).T
        # Calculate output
        self.Y = 5.5 + (1 - x1 / 4) ** 2 + (np.exp(x2 / 7)) / (np.sin(np.exp(x3 / 5)) - 1.5)
        self.names = ['x0', 'x1', 'x2']
        symb = sp.symbols("{}:{}".format('x', 3))
        self.expr = 5.5 + (1 - symb[0] / 4) ** 2 + (sp.exp(symb[1] / 7)) / (sp.sin(sp.exp(symb[2] / 5)) - 1.5)

    def E3(self, n=10000):
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
        self.Y = (1.5 * np.exp(1.5 * x1) + 5 * np.cos(3 * x2))/10
        self.names = ['x0', 'x1']
        symb = sp.symbols("{}:{}".format('x', 2))
        self.expr = (1.5 * sp.exp(1.5 * symb[0]) + 5 * sp.cos(3 * symb[1]))/10

    def E4(self, n=50000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-5, 5, size=n)
            x2 = np.random.uniform(-5, 5, size=n)
            x3 = np.random.uniform(-5, 5, size=n)
            x4 = np.random.uniform(-5, 5, size=n)
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
            x2 = sample_exclude(-20, 20, n, -10, 10)
            x3 = sample_exclude(-20, 20, n, -10, 10)
            x4 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1, x2, x3, x4]).T
        # Calculate output
        self.Y = ((1 - x1) ** 2 + (1 - x3) ** 2 + 100 * (x2 - x1 ** 2) ** 2 + 100 * (x4 - x3 ** 2) ** 2) / 10000
        self.names = ['x0', 'x1', 'x2', 'x3']
        symb = sp.symbols("{}:{}".format('x', 4))
        self.expr = ((1 - symb[0]) ** 2 + (1 - symb[2]) ** 2 + 100 * (symb[1] - symb[0] ** 2) ** 2 +
                     100 * (symb[3] - symb[2] ** 2) ** 2) / 10000

    def E5(self, n=10000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-2, 2, size=n)
            x2 = np.random.uniform(-2, 2, size=n)
            x3 = np.random.uniform(-2, 2, size=n)
            x4 = np.random.uniform(-2, 2, size=n)
        else:
            x1 = sample_exclude(-10, 10, n, -2, 2)
            x2 = sample_exclude(-10, 10, n, -2, 2)
            x3 = sample_exclude(-10, 10, n, -2, 2)
            x4 = sample_exclude(-10, 10, n, -2, 2)
        self.X = np.array([x1, x2, x3, x4]).T
        # Calculate output
        self.Y = np.sin(2 * x1 + x2 * x3) + np.exp(1.2 * x4)
        self.names = ['x0', 'x1', 'x2', 'x3']
        symb = sp.symbols("{}:{}".format('x', 4))
        self.expr = sp.sin(2 * symb[0] + symb[1] * symb[2]) + sp.exp(1.2 * symb[3])

    def E6(self, n=50000):
        np.random.seed(7)
        # Define features
        if not self.extrapolation:
            x1 = np.random.uniform(-10, 10, size=n)
            x2 = np.random.uniform(10, 10, size=n)
            x3 = np.random.uniform(10, 10, size=n)
        else:
            x1 = sample_exclude(-20, 20, n, -10, 10)
            x2 = sample_exclude(-20, 20, n, -10, 10)
            x3 = sample_exclude(-20, 20, n, -10, 10)
        self.X = np.array([x1, x2]).T
        # Calculate output
        self.Y = (np.abs(x3)) / x1 * np.exp(-1 / 10 * (x2 / x1) ** 2)
        self.names = ['x0', 'x1', 'x2']
        symb = sp.symbols("{}:{}".format('x', 3))
        self.expr = (sp.Abs(symb[2])) / symb[0] * sp.exp(-1 / 10 * (symb[1] / symb[0]) ** 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    dataLoader = DataLoader(name='I.6.2', extrapolation=False)
    X, Y, var_names = dataLoader.X, dataLoader.Y, dataLoader.names
    # plt.scatter(X[:, 3], Y)
