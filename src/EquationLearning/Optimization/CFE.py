from pymoo.core.problem import Problem
from pymoo.core.variable import Real
from src.EquationLearning.models.utilities_expressions import *


#############################################################################
# OBJECTIVE FUNCTIONS
#############################################################################
def L1(x_orig, x_counter):
    """Objective function: Modify as few features as possible"""
    comp = 0
    for i in range(len(x_orig)):
        if np.abs((x_orig[i] - x_counter[i]) / x_orig[i]) > 0.03:
            comp += 1
    return comp


def L2(x_orig, x_counter):
    """Objective function: Minimize distance between original and counterfactual samples"""
    dist = []
    for i in range(len(x_orig)):
        dist.append((np.abs(x_orig[i] - x_counter[i])))
    dist = np.array(dist)
    return np.mean(dist)


#############################################################################
# MULTI-OBJECTIVE OPTIMIZATION
#############################################################################

class CFE(Problem):
    """Define an optimization problem that finds the minimum set of coefficients that need to be modified"""

    def __init__(self, xp_orig, list_symbols, gen_fun, values, limits, variable, resample_var):
        """
        :param xp_orig: Original expression
        :param list_symbols: List of Sympy symbols of the original expression
        :param gen_fun: Generative function. It can be an NN model or a sympy function.
        :param values: Initial values of all variables
        :param variable: Index of the variable being analyzed
        :param resample_var: Index of the variable that will be resampled to analyzed dependency w.r.t "variable"
        """
        self.xp_orig = xp_orig
        self.gen_fun = gen_fun
        self.args = get_args(self.xp_orig)
        self.variable, self.values = variable, values.copy()
        self.limits = limits
        self.list_symbols = list_symbols
        self.resample_var = resample_var
        self.iteration = 1
        self.y_temp, self.values_temp = None, self.values.copy()  # Used to store previous evaluations
        self.ref = self.values[self.resample_var].copy()  # Save original value of variable that will be resampled

        # Declare optimization variables
        self.n_vars = len(self.args)
        opt_variables = dict()
        # opt_variables[f"x{0:02}"] = Choice(options=[-1, 1])
        xl, xu = [], []
        for k in range(self.n_vars):
            # Check if this argument is inside a sin or cos operation
            ops = get_op(self.xp_orig, k)
            if len(ops) >= 2:
                if 'sin' in ops[-2] and 'Add' in ops[-1]:
                    opt_variables[f"x{k:02}"] = Real(bounds=(0, np.pi))
                    xl.append(-np.pi)
                    xu.append(np.pi)
                    continue
            if len(ops) == 3:
                if 'sin' in ops[0] and 'Add' in ops[1] and 'Mul' in ops[2]:
                    opt_variables[f"x{k:02}"] = Real(bounds=(0, 30))
                    xl.append(0)
                    xu.append(30)
                    continue
            opt_variables[f"x{k:02}"] = Real(bounds=(-30, 30))
            xl.append(-30)
            xu.append(30)

        super().__init__(vars=opt_variables, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, f3 = [], [], []

        # Sample values of the variable that is being analyzed
        sample = np.random.uniform(self.limits[self.variable][0], self.limits[self.variable][1], 1000)
        # After certain iterations, re-fix the values of the other variables that act as constants
        if (self.iteration - 1) % 40 == 0:
            res_value = np.random.uniform(self.limits[self.resample_var][0], self.limits[self.resample_var][1])
            while np.abs(res_value - self.ref) < 0.3 and np.abs(res_value - self.values[self.resample_var]) < 0.3:
                # If sampled value is close to the original, keep resampling
                res_value = np.random.uniform(self.limits[self.resample_var][0], self.limits[self.resample_var][1])
            self.values[self.resample_var] = res_value

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

        # Evaluate the predicted output using the arguments of this generation
        for s in range(x.shape[0]):
            # Update function with new args
            new_func = set_args(self.xp_orig, list(x[s, :]))
            # Lambdify to evaluate
            expr = sp.lambdify(flatten(self.list_symbols), new_func)
            # Evaluate new expression
            y_pred = expr(*list(values))

            # Calculate first objective
            f1.append(L1(self.args, x[s, :]))
            # Calculate first objective
            f2.append(L2(self.args, x[s, :]))
            # Calculate third objective
            if len(np.argwhere((np.isinf(y_pred)) | (np.isnan(y_pred)))) > 0:
                f3.append(1000)
            else:
                f3.append(np.round(-np.abs(pearsonr(y, y_pred)[0]), 4))
        out["F"] = np.array([f3, f1, f2]).T
        self.iteration += 1

        print("\tIteration " + str(self.iteration) + "/50...", end='\r')
