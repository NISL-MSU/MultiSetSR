import numpy as np
import warnings
from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.robust import RobustTermination
from src.EquationLearning.models.utilities_expressions import *
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

warnings.filterwarnings("ignore")


class CoefficientFitting(Problem):
    """
    Given a skeleton expression, fit its numerical coefficients to maximize correlation
    """

    def __init__(self, skeleton, x_values, values, y_est, climits):
        """
        :param skeleton: Symbolic expression
        :param x_values: Sampled values of the current analyzed variable
        :param values: Sampled values for the rest of the variables
        :param y_est: Corresponding estimated y values
        :param climits: Minimum and maximum coefficient bounds
        """
        # self.skeleton = sp.sympify('1 + sin(2*x1 + 3)')
        self.skeleton = skeleton
        self.x_values = x_values
        self.y_est = y_est
        self.args = get_args(self.skeleton)
        self.values = values.copy()
        self.limits = climits
        self.iteration = 0

        # Declare optimization variables
        self.n_coeff = len(self.args)
        opt_variables = dict()
        xl, xu = [], []
        count = 0
        self.is_offset = np.zeros(self.n_coeff)
        for k in range(self.n_coeff):
            for j in range(self.y_est.shape[1]):
                # Check if this argument is inside a sin or cos operation
                ops = get_op(self.skeleton, k)
                if len(ops) >= 2:
                    if (('sin' in ops[-2]) or ('cos' in ops[-2]) or ('tan' in ops[-2])) and 'Add' in ops[-1]:
                        opt_variables[f"x{count:02}"] = Real(bounds=(-2*np.pi, 2*np.pi))
                        xl.append(-2*np.pi)
                        xu.append(2*np.pi)
                        self.is_offset[k] = 1
                        count += 1
                        continue
                if len(ops) == 3:
                    if (('sin' in ops[0]) or ('cos' in ops[0]) or ('tan' in ops[0])) and 'Add' in ops[1] and 'Mul' in ops[2]:
                        opt_variables[f"x{count:02}"] = Real(bounds=(0, self.limits[1]))
                        xl.append(0)
                        xu.append(self.limits[1])
                        count += 1
                        continue
                opt_variables[f"x{count:02}"] = Real(bounds=(self.limits[0], self.limits[1]))
                xl.append(self.limits[0])
                xu.append(self.limits[1])
                count += 1

        super().__init__(vars=opt_variables, n_obj=1 + self.n_coeff, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

        # Retrieve coefficients
        c = x  # Rename to avoid confusion with input variable x
        outs = np.zeros((x.shape[0], 1 + self.n_coeff))

        for s in range(x.shape[0]):
            error = 0
            for j in range(self.y_est.shape[1]):
                # Get coefficients
                coefs = []
                for k in range(self.n_coeff):
                    coefs.append(c[s, j + self.y_est.shape[1] * k])
                # Replace the coefficients
                fs = set_args(self.skeleton, coefs)
                # Evaluate new expression
                fs_lambda = sp.lambdify(flatten(fs.free_symbols), fs)
                ys = fs_lambda(self.x_values)
                # Calculate correlation between the results of the new expression and the original estimated vector
                if len(np.argwhere((np.isinf(ys)) | (np.isnan(ys)) | (np.abs(ys) > 1000))) > 0:
                    error += 1000
                else:
                    error += np.mean((self.y_est[:, j] - ys)**2) * (2 - pearsonr(self.y_est[:, j], ys)[0])
            # Calculate first objective
            outs[s, 0] = error
            # Calculate second objective
            for k in range(self.n_coeff):
                ci = c[s, k * self.y_est.shape[1]: (k + 1) * self.y_est.shape[1]]
                # Check if the difference between the max and min is less than \epsilon. If True, all coefficients are
                # considered equivalent
                ci = np.array(ci)
                outs[s, k + 1] = - int((np.max(ci) - np.min(ci)) < 0.1)
                if self.is_offset[k] == 1 and outs[s, k + 1] != -1:
                    # Try changing the minimum offsets by 2\pi
                    min_ids = np.where(np.max(ci) - ci >= 6)[0]
                    ci2 = ci.copy()
                    ci2[min_ids] = ci2[min_ids] + 2 * np.pi
                    if (np.max(ci2) - np.min(ci2)) < 0.1:
                        outs[s, k + 1] = -1
                    else:
                        # Try changing the minimum offsets by \pi
                        min_ids = np.where(np.max(ci) - ci >= 0.1)[0]
                        ci2 = ci.copy()
                        ci2[min_ids] = np.pi - ci2[min_ids]
                        min_ids = np.where(np.max(ci2) - ci2 >= 6)[0]
                        ci2[min_ids] = ci2[min_ids] + 2 * np.pi
                        if (np.max(ci2) - np.min(ci2)) < 0.1:
                            outs[s, k + 1] = -1
                        else:
                            ci[min_ids] = -np.pi + ci[min_ids]
                            min_ids = np.where(np.max(ci2) - ci2 >= 6)[0]
                            ci2[min_ids] = ci2[min_ids] + 2 * np.pi
                            if (np.max(ci) - np.min(ci)) < 0.1:
                                outs[s, k + 1] = -1

        out["F"] = outs
        self.iteration += 1

        print("\tIteration " + str(self.iteration), end='\r')


class CheckDependency:

    def __init__(self, skeleton, t, list_vars, gen_func, v_limits, c_limits):
        """
        Given a univariate symbolic expression, find which coefficients are dependent on other functions
        :param skeleton: Symbolic skeleton generated for the t-th variable
        :param t: Index of the variable whose skeleton is currently being analyzed
        :param list_vars: List of Sympy symbols of the original expression
        :param gen_func: Generative function. It can be an NN model or a sympy function
        :param v_limits: Limits of the values that all variables can take
        :param c_limits: Limits of the values that all expression coefficients can take
        """
        self.skeleton = skeleton  # The skeleton is a function of variable x_t (the t-th variable in list_vars)
        self.t = t
        self.list_vars = list_vars
        self.gen_func = gen_func
        self.v_limits = v_limits
        self.c_limits = c_limits
        self.ns = 3
        self.nxs = 500

    def sample_values(self, v):
        """Sample values for the variables that are different than x_t.
        :param v: Index of the variable that will be allowed to vary. The other variables (except x_t will be fixed)
        """
        # Sample nxs values for variable x_t
        xt_values = np.random.uniform(limits[0], limits[1], size=self.nxs)

        # Sample values for variables that are different than x_t
        values_temp = np.zeros((len(self.list_vars), self.ns), dtype=object)
        ct, std, values, y = 0, 0, None, None
        while ct < 10:  # The sampling process is repeated 10 times, we'll keep the values with the highest variation
            y_temp = np.zeros((self.nxs, self.ns))
            for i in range(len(self.list_vars)):
                if i == v:  # Variable that is allowed to vary
                    values_temp[i, :] = np.random.uniform(limits[0], limits[1], size=self.ns)
                elif i == self.t:  # Multiple values are sampled
                    for si in range(self.ns):
                        values_temp[i, si] = list(xt_values)
                else:  # Fixed variables
                    values_temp[i, :] = np.random.uniform(limits[0], limits[1], size=1)
            gen_fun = sp.lambdify(flatten(self.list_vars), g_func(self.list_vars))
            for ix, xx in enumerate(xt_values):
                vals = values_temp.copy()
                vals[xt, :] = xt_values[ix]
                y_temp[ix, :] = gen_fun(*list(vals.astype(np.float)))
            stds = 0
            for i in range(self.nxs):
                stds += np.abs(np.std(y_temp[i, :]))
            # Keep the values that led to the highest sum of standard deviations
            if stds > std:
                y = y_temp.copy()
                values = values_temp.copy()
                std = stds
            ct += 1
        return xt_values, values, y

    def run(self):
        # Analyze the dependency of coefficients skeleton(x_t) on each variable x_v != x_t
        depend_vars = [i for i in range(len(self.list_vars)) if i != self.t]
        final_coeff = get_args(self.skeleton).copy()
        for v in depend_vars[0:]:
            print("Checking which coefficients depend on variable " + '\033[1m' + str(self.list_vars[v]) + '\033[0m')
            # Sample values
            xt_values, values, y = self.sample_values(v)
            # Fit coefficients
            termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-5), period=5)
            problem = CoefficientFitting(skeleton=self.skeleton, x_values=xt_values, values=values,
                                         y_est=y, climits=clim)
            algorithm = NSGA2(pop_size=200)
            res = minimize(problem, algorithm, termination, seed=1, verbose=False)
            # Select best solution
            best_error = np.argmin(res.F[:, 0])
            # Check each coefficient
            for cn, cf in enumerate(res.F[best_error, 1:]):
                if cf != -1:
                    new_var = sp.sympify(str(self.list_vars[v]).replace('x', 'f'))
                    if isinstance(final_coeff[cn], sp.Symbol) and str(final_coeff[cn]) != 'c':
                        final_coeff[cn] = [final_coeff[cn], new_var]
                    elif isinstance(final_coeff[cn], list):
                        (final_coeff[cn]).append(new_var)
                    else:
                        final_coeff[cn] = new_var
        for cf in range(len(final_coeff)):
            if isinstance(final_coeff[cf], list):
                final_coeff[cf] = sp.Symbol(str(final_coeff[cf]))

        # Apply modifications to final expression
        new_skeleton = set_args(self.skeleton, final_coeff)
        print('-----------------------------------------------------------')
        print('The resulting skeleton with dependencies is as follows: ' + str(new_skeleton) + '\n')
        return new_skeleton, final_coeff


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure()

    def g_func(symb):
        # return sp.sin(symb[0] + (symb[1]) * symb[2]) + sp.exp(0.2 * symb[3])
        return (sp.Abs(symb[2])) / symb[0] * sp.exp(-1/10 * (symb[1] / symb[0]) ** 2)


    def g_skeleton(symb):
        c = sp.sympify(str('c'))
        # return sp.sin(c + c * (symb[1])) + c
        return c / symb[0] * sp.exp(c / (symb[0] ** 2))


    # Parameters
    symbols = sp.symbols("{}:{}".format('x', 3))
    limits = [-10, 10]
    clim = [-10, 10]
    xt = 0  # Variable that is currently being analyzed

    # Execute program
    print('Initial skeleton: ' + str(g_skeleton(symbols)))
    print('-----------------------------------------------------------')
    check_dependency = CheckDependency(skeleton=g_skeleton(symbols), t=xt, list_vars=symbols, gen_func=g_func(symbols),
                                       v_limits=limits, c_limits=clim)
    new_sk, new_c = check_dependency.run()

    print()
