import warnings
import sympy as sp
from scipy.stats import pearsonr
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Choice
from src.EquationLearning.models.functions import *
from src.EquationLearning.models.NNModel import NNModel

warnings.filterwarnings("ignore")


class CoefficientProblem(Problem):
    """Define an optimization problem that finds the best coefficients of an expression that maximizes correlation"""

    def __init__(self, operator, gen_fun, variable, values, limits, inv_set=None, only_resample_1v=None):
        self.operator = get_sym_function(operator)[1]  # Get numpy function of selected operator
        self.gen_fun = gen_fun
        self.variable, self.values = variable, values.copy()
        self.limits = limits
        self.best_values = None  # Saves the variable values that were used to find the best coefficients
        self.iteration = 1
        self.n_vars = 5  # 1 binary bit + 4 float variables
        self.y_temp, self.values_temp = None, None  # Used to store previous evaluations
        self.bestX, self.bestC, self.inv = None, 1, False
        self.only_resample_1v = only_resample_1v  # If not None, it indicates the index of the variable (different than
        # self.variable) that will be resampled every 20 epochs
        self.inv_set = inv_set  # In case we want to fix the inversion coefficient since the beginning
        np.random.seed(7)

        # Declare optimization variables
        opt_variables = dict()
        opt_variables[f"x{0:02}"] = Choice(options=[-1, 1])
        for k in range(1, self.n_vars):  # Consider all variables except the s-th variable
            if (operator == 'sin' or operator == 'cos') and (k == 2 or k == 3):
                opt_variables[f"x{k:02}"] = Real(bounds=(-np.pi, np.pi))
            elif (operator == 'sin' or operator == 'cos') and (k == 1):
                opt_variables[f"x{k:02}"] = Real(bounds=(0, 25))
            elif operator == 'square' and (k == 1):
                opt_variables[f"x{k:02}"] = Real(bounds=(0, 25))
            elif operator == 'id' and (k == 1 and k == 2):
                opt_variables[f"x{k:02}"] = Real(bounds=(1, 1))
            elif (operator == 'id' or operator == 'exp') and k == 3:
                opt_variables[f"x{k:02}"] = Real(bounds=(0, 0))
            else:
                opt_variables[f"x{k:02}"] = Real(bounds=(-25, 25))
        super().__init__(vars=opt_variables, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        r, MSE = [], []
        # Separate coefficients
        if self.iteration < 10:
            inv = [1] * len(x)
        else:
            inv = np.array([x[s][f"x{0:02}"] for s in range(len(x))])

        if self.inv_set is not None:
            inv[:] = self.inv_set[0]
        xc = np.array([np.array([x[s][f"x{k:02}"] for k in range(1, self.n_vars)]) for s in range(len(x))])

        # Sample values of the variable that is being analyzed
        sample = np.random.uniform(self.limits[self.variable][0], self.limits[self.variable][1], 1000)
        # After certain iterations, re-fix the values of the other variables that act as constants
        if (self.iteration - 1) % 20 == 0:
            if self.only_resample_1v is None:
                self.values = np.expand_dims(np.array([np.random.uniform(self.limits[v][0], self.limits[v][1])
                                             for v in range(len(self.values))]), axis=1)
            else:
                self.values[self.only_resample_1v] = np.random.uniform(self.limits[self.only_resample_1v][0],
                                                                       self.limits[self.only_resample_1v][1])
            values = np.repeat(self.values, 1000, axis=1)
            values[self.variable] = sample
            self.values_temp = values
            # Obtain the estimated outputs using the generating function (e.g., the NN)
            if isinstance(self.gen_fun, NNModel):  # Used if gen_fun is a neural network
                y = np.array(self.gen_fun.evaluateFold(values.T, batch_size=len(values)))[:, 0]
            else:  # Used if gen_fun is a symbolic expressions
                y = self.gen_fun(*list(values))
            self.y_temp = y
        else:
            values = self.values_temp
            y = self.y_temp

        # Evaluate the predicted output using the coefficients learned during this generation
        for s in range(x.shape[0]):
            if inv[s] == 1:
                xc[s, 1] = 0
            y_pred = self.operator((xc[s, 0] * values[self.variable] + xc[s, 1]) ** (inv[s]) + xc[s, 2]) + xc[s, 3]

            if len(np.argwhere((np.isinf(y_pred)) | (np.isnan(y_pred)) | (np.abs(y_pred) > 1000))) > 0:
                r.append(1000)
            else:
                corr, corr2 = 0, 0
                if self.inv_set is not None:
                    if self.inv_set[1]:
                        bestC = np.round(-np.abs(pearsonr(y, 1 / y_pred)[0]), 5)
                    else:
                        bestC = np.round(-np.abs(pearsonr(y, y_pred)[0]), 5)
                else:
                    if self.iteration > 60:
                        corr = np.round(-np.abs(pearsonr(y, y_pred)[0]), 5)
                        corr2 = np.round(-np.abs(pearsonr(y, 1 / y_pred)[0]), 5)
                        bestC = np.minimum(corr, corr2)
                    else:
                        corr = np.round(-np.abs(pearsonr(y, y_pred)[0]), 5)
                        corr2 = 0
                        bestC = corr

                r.append(bestC)
                if bestC < self.bestC:
                    self.bestC = bestC
                    self.best_values = self.values.copy()
                    if self.inv_set is not None:
                        self.inv = self.inv_set[1]
                    else:
                        if corr > corr2:
                            self.inv = True
                        else:
                            self.inv = False
                    if not self.inv:
                        xc[s, 3] = 0
                    self.bestX = np.insert(xc[s], 0, inv[s])
        r = np.array(r)
        r[np.isnan(r)] = 0
        out["F"] = r
        self.iteration += 1


