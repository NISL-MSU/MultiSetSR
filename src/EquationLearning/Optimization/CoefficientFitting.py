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
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
# from pymoo.operators.selection.tournament import TournamentSelection
warnings.filterwarnings("ignore")


class CoefficientFitting(Problem):
    """
    Given a skeleton expression, fit its numerical coefficients to minimize error
    """

    def __init__(self, skeleton, x_values, y_est, climits, loss_MSE=True):
        """
        :param skeleton: Symbolic expression
        :param x_values: Sampled values of the current analyzed variable
        :param y_est: Corresponding estimated y values
        :param climits: Minimum and maximum coefficient bounds
        :param loss_MSE: If True, we optimize the MSE, otherwise we optimize the correlation (PearsonR)
        """
        self.skeleton = skeleton
        self.x_values = x_values
        self.y_est = y_est
        self.args = get_args(self.skeleton)
        if len(self.args) == 0:  # If it doesn't have inner arguments it means the expression itself is the coefficient
            self.args = [skeleton]
        self.limits = climits
        self.iteration = 0
        self.loss_MSE = loss_MSE

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
            csi[np.abs(csi) <= 0.001] = 0
            if len(self.skeleton.args) > 0:
                fs = set_args(self.skeleton, list(c[si, :]))
                if 'x' not in str(fs.free_symbols):
                    if fs.is_real:
                        ys = np.repeat(float(fs), len(self.x_values), axis=0)
                    else:
                        ys = np.repeat(1000, len(self.x_values), axis=0)  # Penalize these cases
                else:
                    # Evaluate new expression
                    fs_lambda = sp.lambdify(flatten(fs.free_symbols), fs)
                    ys = fs_lambda(self.x_values)
            else:
                ys = np.repeat(c[si, :], len(self.x_values), axis=0)

            if len(np.argwhere((np.isinf(ys)) | (np.isnan(ys)) | (np.abs(ys) > 10**14))) > 0:
                error += 1000
            else:
                # r = pearsonr(self.y_est, ys)[0]
                # if np.isnan(r):
                if self.loss_MSE:
                    er = np.mean(np.abs(self.y_est - ys))
                    penalty = -(np.sum(csi == 0) + np.sum(csi == 1))/1000
                    error += er + penalty
                else:
                    error -= abs(pearsonr(self.y_est, ys)[0])
                # else:
                #     error += np.mean(np.abs(self.y_est - ys)) * (2 - r)
            outs[si, 0] = error

        out["F"] = outs
        self.iteration += 1

        # print("\t\t\tGA Iteration " + str(self.iteration) + "... Best error = " + str(np.min(outs)), end='\r')


class FitGA:

    def __init__(self, skeleton, Xs, Ys, v_limits, c_limits, max_it=None, loss_MSE=True, biased_sol=None):
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
        """
        self.skeleton = skeleton  # The skeleton is a function of variable x_t (the t-th variable in list_vars)
        self.Xs = Xs
        self.Ys = Ys
        self.v_limits = v_limits
        self.c_limits = c_limits
        self.loss_MSE = loss_MSE
        if max_it is None:
            self.termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-4), period=25)
        else:
            self.termination = ('n_gen', max_it)
        self.biased_sol = biased_sol

    def run(self):
        # def binary_tournament(pop, P, *args, **kwargs):
        #     n_tournaments, n_competitors = P.shape
        #     if n_competitors != 2:
        #         raise Exception("Only pressure=2 allowed for binary tournament!")
        #     S = np.full(n_tournaments, -1, dtype=int)
        #     for i in range(n_tournaments):
        #         a, b = P[i]
        #         if pop[a].F < pop[b].F:
        #             S[i] = a
        #         else:
        #             S[i] = b
        #     return S

        # Create the tournament selection
        # selection = TournamentSelection(pressure=2, func_comp=binary_tournament)

        # Fit coefficients
        problem = CoefficientFitting(skeleton=self.skeleton, x_values=self.Xs, y_est=self.Ys, climits=self.c_limits,
                                     loss_MSE=self.loss_MSE)
        pop_size = 200
        if self.biased_sol is not None:
            # Create a population with the biased solution and the remaining random solutions
            Xin = np.random.uniform(low=self.c_limits[0], high=self.c_limits[1], size=(pop_size - 1, len(self.biased_sol)))
            initial_population = np.vstack([np.tile(self.biased_sol, (1, 1)), Xin])
            np.random.shuffle(initial_population)
            algorithm = GA(pop_size=pop_size, biased_initialization=self.biased_sol)
        else:
            algorithm = GA(pop_size=200)
        res = minimize(problem, algorithm, self.termination, seed=1, verbose=False)
        resX = np.round(res.X, 6)
        # resX[np.abs(resX) <= 0.00001] = 0
        fs = None
        if len(self.skeleton.args) > 0:
            fs = set_args(self.skeleton, list(resX))
            if 'x' not in str(fs.free_symbols):
                if fs.is_real:
                    ys = np.repeat(float(fs), len(self.Xs), axis=0)
                else:
                    ys = np.repeat(1000, len(self.Xs), axis=0)  # Penalize these cases
            else:
                # Evaluate new expression
                fs_lambda = sp.lambdify(flatten(fs.free_symbols), fs)
                ys = fs_lambda(self.Xs)
        else:
            ys = np.repeat(resX, len(self.Xs), axis=0)

        if len(self.skeleton.args) > 0:
            if self.loss_MSE:
                return fs, np.mean(np.abs(self.Ys - ys)), resX
            else:
                return fs, -pearsonr(self.Ys, ys)[0], resX
        else:
            return resX, res.F
