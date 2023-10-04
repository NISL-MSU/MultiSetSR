from scipy.stats import pearsonr
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Choice
from src.EquationLearning.models.functions import *
from src.EquationLearning.models.NNModel import NNModel


class DoubleCoefficientProblem(Problem):
    """Define an optimization problem that finds the best coefficients of a nested expression
    that maximizes correlation"""

    def __init__(self, operator, gen_fun, variable, values, limits, operators_list: list, only_resample_1v=None, inv_set=None):
        self.operator = get_sym_function(operator)[1]  # Get numpy function of selected operator
        self.gen_fun = gen_fun
        self.variable, self.values = variable, values.copy()
        self.limits = limits
        self.best_values = None  # Saves the variable values that were used to find the best coefficients
        self.iteration = 1
        self.n_vars = 11  # 2 binary bits + 8 float variables + 1 categorical variable
        self.operators_list = operators_list  # [get_sym_function(op)[1] for op in operators_list]
        self.bestX, self.bestC, self.inv = None, 1, False
        self.only_resample_1v = only_resample_1v  # If not None, it indicates the index of the variable (different than
        # self.variable) that will be resampled every 20 epochs
        self.inv_set = inv_set  # In case we want to fix the inversion coefficients since the beginning
        np.random.seed(7)

        # Prohibit some combination of operators such as sin(sin()), sqrt(sqrt()), exp(exp()), log(log())
        if (operator == 'sin') or (operator == 'cos'):
            if 'sin' in self.operators_list:
                self.operators_list.remove('sin')
            if 'cos' in self.operators_list:
                self.operators_list.remove('cos')
        elif operator == 'exp':
            if 'exp' in self.operators_list:
                self.operators_list.remove('exp')
            if 'log' in self.operators_list:
                self.operators_list.remove('log')
        elif operator == 'log':
            if 'exp' in self.operators_list:
                self.operators_list.remove('exp')
            if 'log' in self.operators_list:
                self.operators_list.remove('log')
        elif operator == 'sqrt':
            if 'sqrt' in self.operators_list:
                self.operators_list.remove('sqrt')

        # Declare optimization variables
        opt_variables = dict()
        opt_variables[f"x{0:02}"] = Choice(options=[-1, 1])
        opt_variables[f"x{0:02}"].prob = 0.1
        opt_variables[f"x{1:02}"] = Choice(options=[-1, 1])
        opt_variables[f"x{1:02}"].prob = 0.1
        for k in range(2, self.n_vars - 1):  # Consider all variables except the s-th variable
            if (operator == 'sin' or operator == 'cos') and (k == 3 or k == 4):
                opt_variables[f"x{k:02}"] = Real(bounds=(0, np.pi))
            elif (operator == 'exp') and (k == 2):
                opt_variables[f"x{k:02}"] = Real(bounds=(-10, 10))
            elif (operator == 'exp') and (k == 3) and (k == 4):
                opt_variables[f"x{k:02}"] = Real(bounds=(0, 0))
            elif operator == 'square' and (k == 2):
                opt_variables[f"x{k:02}"] = Real(bounds=(0, 30))
            else:
                opt_variables[f"x{k:02}"] = Real(bounds=(-30, 30))
        opt_variables[f"x{self.n_vars - 1:02}"] = Choice(options=np.arange(len(self.operators_list)))
        super().__init__(vars=opt_variables, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        r = []
        # Separate coefficients
        if self.iteration < 5:
            inv1, inv2 = [1] * len(x), [1] * len(x)
        else:
            inv1, inv2 = np.array([x[s][f"x{0:02}"] for s in range(len(x))]), np.array(
                [x[s][f"x{1:02}"] for s in range(len(x))])

        if self.inv_set is not None:
            inv1[:] = self.inv_set[0]
            inv2[:] = self.inv_set[1]

        x1 = np.array([np.array([x[s][f"x{k:02}"] for k in range(2, 6)]) for s in range(len(x))])
        x2 = np.array([np.array([x[s][f"x{k:02}"] for k in range(6, self.n_vars - 1)]) for s in range(len(x))])
        main_operator = np.array([x[s][f"x{self.n_vars - 1:02}"] for s in range(len(x))])

        # Sample values of the variable that is being analyzed
        sample = np.random.uniform(self.limits[self.variable][0], self.limits[self.variable][1], 5000)
        # After certain iterations, re-fix the values of the other variables that act as constants
        if (self.iteration - 1) % 20 == 0:
            if self.only_resample_1v is None:
                self.values = np.expand_dims(np.array([np.random.uniform(self.limits[v][0], self.limits[v][1])
                                             for v in range(len(self.values))]), axis=1)
            else:
                self.values[self.only_resample_1v] = np.random.uniform(self.limits[self.only_resample_1v][0],
                                                                       self.limits[self.only_resample_1v][1])
            values = np.repeat(self.values, 5000, axis=1)
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
            if inv1[s] == 1:
                x1[s, 1] = 0
            if inv2[s] == 1:
                x2[s, 1] = 0
            y_pred = self.operator((x1[s, 0] * values[self.variable] + x1[s, 1]) ** (inv1[s]) + x1[s, 2]) + x1[s, 3]
            y_pred = get_sym_function(self.operators_list[main_operator[s]])[1](
                (x2[s, 0] * y_pred + x2[s, 1]) ** (inv2[s]) + x2[s, 2]) + x2[s, 3]

            if len(np.argwhere((np.isinf(y_pred)) | (np.isnan(y_pred)) | (np.abs(y_pred) > 1000))) > 0:
                r.append(1000)
            else:
                corr, corr2 = 0, 0
                if self.inv_set is not None:
                    if self.inv_set[2]:
                        bestC = np.round(-np.abs(pearsonr(y, 1 / y_pred)[0]), 5)
                    else:
                        bestC = np.round(-np.abs(pearsonr(y, y_pred)[0]), 5)
                else:
                    corr = np.round(-np.abs(pearsonr(y, y_pred)[0]), 5)
                    corr2 = np.round(-np.abs(pearsonr(y, 1 / y_pred)[0]), 5)
                    bestC = np.minimum(corr, corr2)
                r.append(bestC)
                if bestC < self.bestC:
                    self.bestC = bestC
                    self.best_values = self.values
                    if corr > corr2:
                        self.inv = True
                    else:
                        self.inv = False
                    if not self.inv:
                        x2[s, 3] = 0
                    self.bestX = [inv1[s], inv2[s]] + list(x1[s, :]) + list(x2[s, :]) + [main_operator[s]]
        r = np.array(r)
        r[np.isnan(r)] = 0
        out["F"] = r
        self.iteration += 1
