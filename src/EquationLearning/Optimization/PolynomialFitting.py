from scipy.stats import pearsonr
from src.EquationLearning.models.functions import *
from src.EquationLearning.models.NNModel import NNModel


class PolynomialFitting:
    """Define a polynomial fitting problem"""

    def __init__(self, gen_fun, variable, limits, seed=3, only_resample_1v=None):
        self.gen_fun = gen_fun
        self.variable, self.values = variable, np.zeros((len(limits), 1))
        self.limits = limits
        np.random.seed(seed)
        self.only_resample_1v = only_resample_1v  # If not None, it indicates the index of the variable (different than
        # self.variable) that will be resampled
        self.best_values, self.best_deg = None, None

    def _fit_coefficients(self, degree=None):
        """
        Fit polynomial coefficients
        :param degree: Polynomial degree. If None, the best degree between 1 and 6 will be selected
        """
        # Sample values of the variable that is being analyzed
        sample = np.random.uniform(self.limits[self.variable][0], self.limits[self.variable][1], 1000)
        # Fix the values of the other variables that act as constants
        if self.only_resample_1v is not None:
            self.values[self.only_resample_1v] = np.random.uniform(self.limits[self.only_resample_1v][0],
                                                                   self.limits[self.only_resample_1v][1])
        else:
            self.values = np.expand_dims(np.array([np.random.uniform(self.limits[v][0], self.limits[v][1])
                                         for v in range(len(self.values))]), axis=1)
        values = np.repeat(self.values, 1000, axis=1)
        values[self.variable] = sample

        # Obtain the estimated outputs using the generating function (e.g., the NN)
        if isinstance(self.gen_fun, NNModel):  # Used if gen_fun is a neural network
            y = np.array(self.gen_fun.evaluateFold(values.T, batch_size=len(values)))[:, 0]
        else:  # Used if gen_fun is a symbolic expressions
            y = self.gen_fun(*list(values))

        # Obtain coefficients with different polynomial degrees
        minC, deg, bestCoeff, best_deg = 0, 1, [], 0
        while deg < 5 and minC > -1:
            if degree is None:
                z = np.polyfit(sample, y, deg)
            else:
                z = np.polyfit(sample, y, degree)
            # Obtain estimated values
            z = np.polyfit(sample, y, deg)
            p = np.poly1d(z)
            y_pred = p(sample)
            corr = np.round(-np.abs(pearsonr(y, y_pred)[0]), 5)
            corr2 = np.round(-np.abs(pearsonr(y, 1 / y_pred)[0]), 5)
            bestC = np.minimum(corr, corr2)
            if bestC < minC:
                minC = bestC
                bestCoeff = z
                best_deg = deg
            deg += 1
            self.best_values = self.values.copy()
            if degree is not None:
                best_deg = degree
                break
        return bestCoeff, np.abs(minC), best_deg

    def fit(self, degree=None):
        # Fit coefficients three times and check if the polynomial degree is the same in all cases
        coeff1, C1, deg1 = self._fit_coefficients(degree)
        coeff2, C2, deg2 = self._fit_coefficients(degree=deg1)
        coeff3, C3, deg3 = self._fit_coefficients(degree=deg1)
        coeff4, C4, deg4 = self._fit_coefficients(degree=deg1)

        self.best_deg = deg1

        return coeff1, (C1 + C2 + C3 + C4) / 4
