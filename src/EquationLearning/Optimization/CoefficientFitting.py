import numpy as np
import warnings
from sympy import flatten
from scipy.stats import pearsonr
from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination.robust import RobustTermination
from EquationLearning.models.utilities_expressions import *
from EquationLearning.Optimization.GP import norm_func, simplify
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
warnings.filterwarnings("ignore")


class CoefficientFitting(Problem):
    """
    Given a skeleton expression, fit its numerical coefficients to minimize error
    """

    def __init__(self, skeleton, x_values, y_est, climits, symbols_list, loss_MSE=True):
        """
        :param skeleton: Symbolic expression
        :param x_values: Sampled values of the current analyzed variable
        :param y_est: Corresponding estimated y values
        :param climits: Minimum and maximum coefficient bounds
        :param symbols_list: List of variable names
        :param loss_MSE: If True, we optimize the MSE, otherwise we optimize the correlation (PearsonR)
        """
        self.skeleton = skeleton
        self.x_values = x_values
        self.y_est = y_est
        self.args = [ar for ar in self.skeleton.free_symbols if 'c' in str(ar)]
        if len(self.args) == 0:  # If it doesn't have inner arguments it means the expression itself is the coefficient
            self.args = [skeleton]
        self.limits = climits
        self.iteration = 0
        self.loss_MSE = loss_MSE
        self.symbols_list = symbols_list

        # Declare optimization variables
        self.n_coeff = len(self.args)
        opt_variables = dict()
        xl, xu = [], []
        self.is_offset = np.zeros(self.n_coeff)
        for k in range(self.n_coeff):
            # Check if this argument is inside a sin or cos operation
            ops = get_op(self.skeleton, k)
            if len(ops) >= 2:
                if (('sin' in ops[-2]) or ('cos' in ops[-2]) or ('tan' in ops[-2])) and 'Add' in ops[-1]:
                    opt_variables[f"x{k:02}"] = Real(bounds=(-2*np.pi, 2*np.pi))
                    xl.append(-2*np.pi)
                    xu.append(2*np.pi)
                    continue
                # elif ('exp' in ops[-2]) and 'Mul' in ops[-1]:
                #     opt_variables[f"x{k:02}"] = Real(bounds=(-2.5, 2.5))
                #     xl.append(-2.5)
                #     xu.append(2.5)
                #     continue
            if len(ops) == 3:
                if (('sin' in ops[0]) or ('cos' in ops[0]) or ('tan' in ops[0])) and 'Add' in ops[1] and 'Mul' in ops[2]:
                    opt_variables[f"x{k:02}"] = Real(bounds=(0, self.limits[1]))
                    xl.append(0)
                    xu.append(self.limits[1])
                    continue
            opt_variables[f"x{k:02}"] = Real(bounds=(self.limits[0], self.limits[1]))
            xl.append(self.limits[0])
            xu.append(self.limits[1])

        super().__init__(vars=opt_variables, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Retrieve coefficients
        c = x  # Rename to avoid confusion with input variable x
        outs = np.zeros((x.shape[0], 1))

        for si in range(x.shape[0]):
            error = 0
            # Replace the coefficients
            csi = np.round(c[si, :], 6)
            if len(self.skeleton.args) > 0:
                fs = self.skeleton
                for ia, arg in enumerate(self.args):
                    fs = fs.subs({arg: c[si, ia]})
                if 'x' not in str(fs.free_symbols):
                    if fs.is_real:
                        ys = np.repeat(float(fs), len(self.x_values), axis=0)
                    else:
                        ys = np.repeat(1000, len(self.x_values), axis=0)  # Penalize these cases
                else:
                    # Evaluate new expression
                    fs_lambda = sp.lambdify(flatten(self.symbols_list), fs)
                    if self.x_values.ndim > 1:
                        ys = fs_lambda(*list(self.x_values.T))
                    else:
                        ys = fs_lambda(self.x_values)
            else:
                ys = np.repeat(c[si, :], len(self.x_values), axis=0)

            if len(np.argwhere((np.isinf(ys)) | (np.isnan(ys)) | (np.abs(ys) > 10**14))) > 0:
                error += 1000
            else:
                if self.loss_MSE:
                    er = np.mean(np.abs(self.y_est - ys))
                    penalty = -(np.sum(np.abs(csi) <= 0.0001) + np.sum(np.abs(csi) == 1))/1000
                    error += er + penalty
                else:
                    error -= abs(pearsonr(self.y_est, ys)[0])
            outs[si, 0] = error

        out["F"] = outs
        self.iteration += 1

        print("\t\t\tGA Iteration " + str(self.iteration) + "... Best error = " + str(np.min(outs)), end='\r')


class FitGA:

    def __init__(self, skeleton, Xs, Ys, v_limits, c_limits, max_it=None, loss_MSE=True, biased_sol=None, pop_size=200,
                 seed=1, symbols_list=None):
        """
        Given a univariate symbolic expression, find which coefficients are dependent on other functions
        :param skeleton: Symbolic skeleton generated for the t-th variable
        :param Xs: Support values
        :param Ys: Corresponding response values
        :param v_limits: Limits of the values that all variables can take
        :param c_limits: Limits of the values that all expression coefficients can take
        :param max_it: If not None, specify the maximum number of iterations for the GA
        :param loss_MSE: If True, we optimize the MSE, otherwise we optimize the correlation (PearsonR)
        :param biased_sol: Biased initial population
        :param pop_size: Population size
        """
        if 'c' not in str(skeleton):
            skeleton = sympy.sympify('c') * skeleton
        self.skeleton = add_constant_identifier(skeleton)[0]  # The skeleton is a function of variable x_t (the t-th variable in list_vars)
        self.Xs = Xs
        self.Ys = Ys
        self.v_limits = v_limits
        self.c_limits = c_limits
        self.loss_MSE = loss_MSE
        self.seed = seed
        if max_it is None:
            self.termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-6), period=40)
        else:
            self.termination = ('n_gen', max_it)
        self.biased_sol = biased_sol
        self.pop_size = pop_size
        if symbols_list is None:
            self.symbols_list = [arg for arg in self.skeleton.free_symbols if 'x' in str(arg)]
        else:
            self.symbols_list = symbols_list
        if len(self.symbols_list) > 1:  # Sort the symbols
            self.symbols_list = [sp.sympify(st) for st in
                                 sorted([str(sym) for sym in self.symbols_list], key=lambda s: int(s[1:]))]

    def run(self):
        # Fit coefficients
        problem = CoefficientFitting(skeleton=self.skeleton, x_values=self.Xs, y_est=self.Ys, climits=self.c_limits,
                                     loss_MSE=self.loss_MSE, symbols_list=self.symbols_list)
        np.random.seed(self.seed)
        if self.biased_sol is not None:
            # Create a population with the biased solution and the remaining random solutions
            Xin = np.random.uniform(low=self.c_limits[0], high=self.c_limits[1], size=(self.pop_size - 1, len(self.biased_sol)))
            initial_population = np.vstack([np.tile(self.biased_sol, (1, 1)), Xin])
            np.random.shuffle(initial_population)
            algorithm = GA(pop_size=self.pop_size, biased_initialization=self.biased_sol)
        else:
            algorithm = GA(pop_size=self.pop_size)
        res = minimize(problem, algorithm, self.termination, seed=self.seed, verbose=False)
        resX = np.round(res.X, 6)
        resX[np.abs(resX) <= 0.0016] = 0
        fs = None
        if len(self.skeleton.args) > 0:
            fs = self.skeleton
            args = [ar for ar in self.skeleton.free_symbols if 'c' in str(ar)]
            for ia, arg in enumerate(args):
                fs = fs.subs({arg: resX[ia]})
            if 'x' not in str(fs.free_symbols):
                if fs.is_real:
                    ys = np.repeat(float(fs), len(self.Xs), axis=0)
                else:
                    ys = np.repeat(1000, len(self.Xs), axis=0)  # Penalize these cases
            else:
                # Evaluate new expression
                fs_lambda = sp.lambdify(flatten(self.symbols_list), fs)
                if self.Xs.ndim > 1:
                    ys = fs_lambda(*list(self.Xs.T))
                else:
                    ys = fs_lambda(self.Xs)
        else:
            ys = np.repeat(resX, len(self.Xs), axis=0)

        if len(self.skeleton.args) > 0:
            if self.loss_MSE:
                fs = simplify(fs, all_var=True)[0]
                return fs, np.mean(np.abs(self.Ys - ys)), resX
            else:
                fs = simplify(norm_func(fs))[0]
                if len(np.argwhere((np.isinf(ys)) | (np.isnan(ys)) | (np.abs(ys) > 10 ** 14))) > 0:
                    metric = 1000
                else:
                    metric = -pearsonr(self.Ys, ys)[0]
                return fs, metric, resX
        else:
            return resX, res.F
