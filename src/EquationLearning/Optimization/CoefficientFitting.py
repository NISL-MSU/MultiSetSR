import numpy as np
import warnings
from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination.robust import RobustTermination
from src.EquationLearning.models.utilities_expressions import *
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

warnings.filterwarnings("ignore")


class CoefficientFitting(Problem):
    """
    Given a skeleton expression, fit its numerical coefficients to minimize error
    """

    def __init__(self, skeleton, x_values, y_est, climits):
        """
        :param skeleton: Symbolic expression
        :param x_values: Sampled values of the current analyzed variable
        :param y_est: Corresponding estimated y values
        :param climits: Minimum and maximum coefficient bounds
        """
        self.skeleton = skeleton
        self.x_values = x_values
        self.y_est = y_est
        self.args = get_args(self.skeleton)
        if len(self.args) == 0:  # If it doesn't have inner arguments it means the expression itself is the coefficient
            self.args = [skeleton]
        self.limits = climits
        self.iteration = 0

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
                elif ('exp' in ops[-2]) and 'Mul' in ops[-1]:
                    opt_variables[f"x{k:02}"] = Real(bounds=(-3, 3))
                    xl.append(-3)
                    xu.append(3)
                    continue
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
            if len(self.skeleton.args) > 0:
                # Replace the coefficients
                fs = set_args(self.skeleton, list(c[si, :]))
                # Evaluate new expression
                fs_lambda = sp.lambdify(flatten(fs.free_symbols), fs)
                ys = fs_lambda(self.x_values)
            else:
                ys = np.repeat(c[si, :], len(self.x_values), axis=0)
            # Calculate correlation between the results of the new expression and the original estimated vector
            if len(np.argwhere((np.isinf(ys)) | (np.isnan(ys)) | (np.abs(ys) > 1000000))) > 0:
                error += 1000
            else:
                # r = pearsonr(self.y_est, ys)[0]
                # if np.isnan(r):
                error += np.mean((self.y_est - ys)**2)
                # else:
                #     error += np.mean((self.y_est - ys)**2) * (2 - r)
            # Calculate first objective
            outs[si, 0] = error

        out["F"] = outs
        self.iteration += 1

        print("\t\t\tGA Iteration " + str(self.iteration) + "...", end='\r')


class FitGA:

    def __init__(self, skeleton, Xs, Ys, v_limits, c_limits):
        """
        Given a univariate symbolic expression, find which coefficients are dependent on other functions
        :param skeleton: Symbolic skeleton generated for the t-th variable
        :param Xs: Support values
        :param Ys: Corresponding response values
        :param v_limits: Limits of the values that all variables can take
        :param c_limits: Limits of the values that all expression coefficients can take
        """
        self.skeleton = skeleton  # The skeleton is a function of variable x_t (the t-th variable in list_vars)
        self.Xs = Xs
        self.Ys = Ys
        self.v_limits = v_limits
        self.c_limits = c_limits

    def run(self):
        # Fit coefficients
        termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-5), period=20)
        problem = CoefficientFitting(skeleton=self.skeleton, x_values=self.Xs, y_est=self.Ys, climits=self.c_limits)
        algorithm = GA(pop_size=400)
        res = minimize(problem, algorithm, termination, seed=1, verbose=False)

        if len(self.skeleton.args) > 0:
            return set_args(self.skeleton, list(res.X)), res.F
        else:
            return res.X, res.F