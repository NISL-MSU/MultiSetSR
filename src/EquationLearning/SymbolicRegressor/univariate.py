import warnings
import sympy as sym
from scipy.stats import pearsonr
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from sympy.utilities.iterables import flatten
from pymoo.algorithms.soo.nonconvex.ga import GA
from src.EquationLearning.models.functions import *
from pymoo.termination.robust import RobustTermination
from src.EquationLearning.models.NNModel import NNModel
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from src.EquationLearning.models.symbolic_expression import round_expr
from src.EquationLearning.Optimization.CoefficientProblem import CoefficientProblem
from src.EquationLearning.Optimization.CFE import CFE, get_args, set_args, verify_dependency
from src.EquationLearning.Optimization.DoubleCoefficientProblem import DoubleCoefficientProblem
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination

warnings.filterwarnings("ignore")


class PolynomialFitting:
    """Define a polynomial fitting problem"""

    def __init__(self, gen_fun, variable, values, limits, seed=3, only_resample_1v=None, known_deg=None):
        self.gen_fun = gen_fun
        self.variable, self.values = variable, values.copy()
        self.limits = limits
        np.random.seed(seed)
        self.only_resample_1v = only_resample_1v  # If not None, it indicates the index of the variable (different than
        # self.variable) that will be resampled every 20 epochs
        self.best_values, self.best_deg = None, None
        self.known_deg = known_deg

    def _fit_coefficients(self):
        # Sample values of the variable that is being analyzed
        sample = np.random.uniform(self.limits[self.variable][0], self.limits[self.variable][1], 1000)
        # Fix the values of the other variables that act as constants
        if self.only_resample_1v is not None:
            self.values[self.only_resample_1v] = np.random.uniform(self.limits[self.only_resample_1v][0],
                                                                   self.limits[self.only_resample_1v][1])
        # else:
        #     self.values = np.expand_dims(np.array([np.random.uniform(self.limits[v][0], self.limits[v][1])
        #                                  for v in range(len(self.values))]), axis=1)
        values = np.repeat(self.values, 1000, axis=1)
        values[self.variable] = sample

        # Obtain the estimated outputs using the generating function (e.g., the NN)
        if isinstance(self.gen_fun, NNModel):  # Used if gen_fun is a neural network
            y = np.array(self.gen_fun.evaluateFold(values.T, batch_size=len(values)))[:, 0]
        else:  # Used if gen_fun is a symbolic expressions
            y = self.gen_fun(*list(values))

        # Obtain coefficients with different polynomial degrees
        minC, deg, bestCoeff, best_deg = 0, 1, [], 0
        while deg < 5 and minC > -1:  # TODO: Consider parsimony
            if self.known_deg is None:
                z = np.polyfit(sample, y, deg)
            else:
                z = np.polyfit(sample, y, self.known_deg)
            # Obtain estimated values
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
            if self.known_deg is not None:
                break
        return bestCoeff, np.abs(minC), best_deg

    def fit(self):
        # Fit coefficients three times and check if the polynomial degree is the same in all cases
        coeff1, C1, deg1 = self._fit_coefficients()
        coeff2, C2, deg2 = self._fit_coefficients()
        coeff3, C3, deg3 = self._fit_coefficients()

        ind = np.argmax([deg1, deg2, deg3])
        if ind == 0:
            self.best_deg = deg1
            return coeff1, C1
        elif ind == 1:
            self.best_deg = deg2
            return coeff2, C2
        else:
            self.best_deg = deg3
            return coeff3, C3


class FitCoefficients:
    def __init__(self, input_var, symbols_list, gen_fun, operators_list, limits, th_corr: float = 0.99998):
        """Initialize coefficient fitting object using gradient descent
        :param input_var: Symbolic expression of the variable that will be analyzed
        :param gen_fun: Generating function ($f(x) = \hat{y]$)
        :param operators_list: List of all the unary operators allowed
        :param limits: Lower and upper bounds (tuple) of each explanatory variable
        :param th_corr: Correlation threshold"""
        # Class variables
        self.th_corr = th_corr
        self.variable = input_var
        self.symbols_list = symbols_list
        self.var_ind = np.where(np.array(self.symbols_list) == self.variable)[0][0]
        self.operators_list = operators_list
        self.limits = limits

        if isinstance(gen_fun, NNModel):
            self.gen_fun = gen_fun
        else:  # If gen_fun is a symbolic expression, lambdify it to accelerate computations
            self.gen_fun = sym.lambdify(flatten(symbols_list), gen_fun)

        # Sample random values for all the variables
        self.dic_values = np.expand_dims(np.array([np.random.uniform(self.limits[v][0], self.limits[v][1])
                                                   for v in range(len(symbols_list))]), axis=1)

        self.termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-5), period=50)

    def get_polynomial(self, coeff, calc_complexity=True):
        """Retrieve polynomial symbolic expression"""
        xp = 0
        for d in range(len(coeff)):
            if isinstance(coeff[d], float) or isinstance(coeff[d], int):
                if np.abs(coeff[d]) < 0.001:
                    coeff[d] = 0
            xp += coeff[d] * self.variable ** (len(coeff) - d)

        if calc_complexity:  # Only calculate complexity when specified
            if len(coeff) <= 2:
                comp = np.sum(np.abs(coeff) >= 0.01)
            else:
                comp = 2 * np.sum(np.abs(coeff[:len(coeff) - 2]) >= 0.01) + np.sum(np.abs(coeff[len(coeff) - 2:]) >= 0.01)
            return round_expr(xp, 4), comp
        else:
            return round_expr(xp, 4)

    def get_expression(self, coeff, operator, inv, calc_complexity=True):
        """Retrieve simplified symbolic expression"""
        comp = 0
        if coeff[0] == 1:
            xp = get_sym_function(operator)[0]((coeff[1] * self.variable + coeff[2]) + coeff[3])
            if calc_complexity:  # Only calculate complexity when specified
                comp = get_sym_function(operator)[2] + (np.abs(coeff[2] + coeff[3]) > 0.001) * .5 + (
                            np.abs(coeff[4]) > 0.001)
        else:
            xp = get_sym_function(operator)[0](1 / (coeff[1] * self.variable + coeff[2]) + coeff[3])
            if calc_complexity:  # Only calculate complexity when specified
                comp = get_sym_function(operator)[2] + 1 + (np.abs(coeff[2]) > 0.001) * .5 + \
                       (np.abs(coeff[3]) > 0.001) * .5 + (np.abs(coeff[4]) > 0.001)
        # Divide by a constant so that the first coefficient is always 1
        if isinstance(xp + coeff[4], sym.Mul):
            xp = (xp + coeff[4]).args[1]
        elif isinstance(xp, sym.Mul):
            xp = xp.args[1] + coeff[4] / xp.args[0]
        else:
            xp = xp + coeff[4]
        if inv:
            xp = 1 / xp
            if calc_complexity:
                comp += 2

        if calc_complexity:
            return round_expr(xp, 4), comp
        else:
            return round_expr(xp, 4)

    def get_expression_nested(self, coeff, operator1, secondary_operators, inv, calc_complexity=True):
        """Retrieve simplified symbolic expression"""
        comp = 0
        operator2 = secondary_operators[int(coeff[-1])]
        inv1 = coeff[0]
        inv2 = coeff[1]
        x1 = [coeff[k] for k in range(2, 6)]
        x2 = [coeff[k] for k in range(6, 10)]

        # TODO: if operator 2 is sin, apply mod

        xp = get_sym_function(operator1)[0]((x1[0] * self.variable + x1[1]) ** inv1 + x1[2]) + x1[3]
        xp = get_sym_function(operator2)[0]((x2[0] * xp + x2[1]) ** inv2 + x2[2])

        if calc_complexity:  # Only calculate complexity when specified
            if inv1 == 1:
                comp = get_sym_function(operator1)[2] + (np.abs(x1[1] + x1[2]) > 0.001) * .5 + (np.abs(x1[3]) > 0.001)
            else:
                comp = get_sym_function(operator1)[2] + 1 + (np.abs(x1[1]) > 0.001) * .5 + (np.abs(x1[2]) > 0.001) * .5 + \
                       (np.abs(x1[3]) > 0.001)

            if inv2 == 1:
                comp += get_sym_function(operator2)[2] + (np.abs(x2[1] + x2[2]) > 0.001) * .5 + (np.abs(x2[3]) > 0.001)
            else:
                comp += get_sym_function(operator2)[2] + 1 + (np.abs(x2[1]) > 0.001) * .5 + (np.abs(x2[2]) > 0.001) * .5 + \
                        (np.abs(x2[3]) > 0.001)

        # Divide by a constant so that the first coefficient is always 1
        if isinstance(xp + x2[3], sym.Mul):
            xp = (xp + x2[3]).args[1]
        elif isinstance(xp, sym.Mul):
            xp = xp.args[1] + x2[3] / xp.args[0]
        else:
            xp = xp + x2[3]

        if inv:
            xp = 1 / xp
            if calc_complexity:
                comp += 2

        if calc_complexity:
            return round_expr(xp, 4), comp
        else:
            return round_expr(xp, 4)

    def fit(self):
        """Find the expression that yields the best correlation"""
        # best_values = None
        #############################################################################
        # Polynomial fitting
        #############################################################################
        print('Polynomial fitting:')
        problem = PolynomialFitting(gen_fun=self.gen_fun, variable=self.var_ind,
                                    values=self.dic_values, limits=self.limits)
        coeff, best_corr = problem.fit()
        best_values, best_deg = problem.best_values, problem.best_deg
        coeff = coeff[:-1]   # Get rid of last constant
        max_coeff = np.max(coeff)
        best_coeff = coeff / max_coeff
        best_coeff[np.abs(best_coeff) < 0.01] = 0
        if best_coeff[0] == 0:
            best_coeff = best_coeff[1:]
            best_deg -= 1
        best_exp, best_comp = self.get_polynomial(coeff=best_coeff)
        best_exp_type = 'poly'
        print('\t Resulting expression: ' + str(best_exp) + '\t r = ' + str(np.round(best_corr, 6)) +
              '\t Complexity = ' + str(best_comp))

        for operator in self.operators_list:
            algorithm = GA(pop_size=350, sampling=MixedVariableSampling(),
                           mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                           eliminate_duplicates=MixedVariableDuplicateElimination(), )
            print("Testing operator: " + operator)
            # Find coefficients using a GA-based optimization
            #############################################################################
            # Simple expression search
            #############################################################################
            problem = CoefficientProblem(operator=operator, gen_fun=self.gen_fun,
                                         variable=self.var_ind, values=self.dic_values, limits=self.limits)
            res = minimize(problem, algorithm, self.termination, seed=2, verbose=False)  # ('n_gen', 500)
            # Check if there was convergence
            if np.min(res.F) < -self.th_corr:
                # Select best solution
                coeff, corr = problem.bestX, np.abs(problem.bestC)
                xp, comp = self.get_expression(coeff=coeff, operator=operator, inv=problem.inv)
                print('\t Resulting expression: ' + str(xp) + '\t r = ' + str(np.round(corr, 6)) +
                      '\t Complexity = ' + str(comp))
                exp_type = 'simple'
            else:
                algorithm = GA(pop_size=450, sampling=MixedVariableSampling(),
                               mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                               eliminate_duplicates=MixedVariableDuplicateElimination(), )
                #############################################################################
                # Nested expression search
                #############################################################################
                # If there was no success using a simple subexpression, try using a nested expression
                print("\tWe couldn't find a suitable simple expression. Trying to find a nested expression...")
                problem = DoubleCoefficientProblem(operator=operator, gen_fun=self.gen_fun, variable=self.var_ind,
                                                   values=self.dic_values, limits=self.limits,
                                                   operators_list=self.operators_list.copy())
                _ = minimize(problem, algorithm, self.termination, seed=1, verbose=False)  # ('n_gen', 500)
                coeff, corr = problem.bestX, np.abs(problem.bestC)
                xp, comp = self.get_expression_nested(coeff=coeff, operator1=operator,
                                                      secondary_operators=problem.operators_list, inv=problem.inv)
                print('\t Resulting expression: ' + str(xp) + '\t r = ' + str(np.round(corr, 6)) +
                      '\t Complexity = ' + str(comp))
                exp_type = 'nested'

            # If performance improved w.r.t. previous operators, save
            if (np.floor(1000 * corr) > np.floor(1000 * best_corr)) or \
                    (np.floor(1000 * corr) >= np.floor(1000 * best_corr) and comp < best_comp):
                best_corr = corr
                best_exp = round_expr(xp, 4)
                best_comp = comp
                best_coeff = coeff
                best_exp_type = exp_type
                best_values = problem.best_values.copy()
        best_exp = sym.factor(best_exp)
        print('-----------------------------------------------------------')
        print('Best expression: ' + str(best_exp))
        print('-----------------------------------------------------------')

        #############################################################################
        # Detect dependency between variables
        #############################################################################
        functions = sym.symbols("{}:{}".format('f', len(self.symbols_list)))
        # Update list of the best coefficients (include only numerical coefficients of final expression)
        if best_exp_type != 'poly':
            best_coeff = get_args(best_exp)
        final_coeff = list(best_coeff)

        # Repeat for all variables different from the one being analyzed
        for vn, const_var in enumerate(self.symbols_list):
            if const_var != self.variable:
                print("Modifying values of variable " + '\033[1m' + str(const_var) + '\033[0m' +
                      " to check which coefficients of the estimated expression change")
                if verify_dependency(xp_orig=best_exp, list_symbols=self.symbols_list, gen_fun=self.gen_fun,
                                     values=best_values.copy(), variable=self.var_ind, resample_var=vn,
                                     r_orig=best_corr, limits=self.limits):
                    if best_exp_type == 'poly':
                        problem2 = PolynomialFitting(gen_fun=self.gen_fun, variable=self.var_ind,
                                                     values=self.dic_values, seed=3, only_resample_1v=vn,
                                                     known_deg=best_deg, limits=self.limits)
                        best_coeff2, best_corr = problem2.fit()
                        best_coeff2 = best_coeff2[:-1] / max_coeff  # Get rid of last constant
                        best_coeff2[np.abs(best_coeff2) < 0.01] = 0
                        best_exp2, _ = self.get_polynomial(coeff=best_coeff2)
                    else:
                        algorithm = NSGA2(pop_size=300)
                        problem2 = CFE(xp_orig=best_exp, list_symbols=self.symbols_list, gen_fun=self.gen_fun,
                                       values=best_values.copy(), variable=self.var_ind, resample_var=vn,
                                       limits=self.limits)
                        res = minimize(problem2, algorithm, ('n_gen', 159), seed=4, verbose=False)

                        # Select best solution
                        sols = res.F
                        sols[:, 0] = np.floor(1000 * sols[:, 0])
                        bestf1 = np.min(sols[:, 0])
                        f1sols = sols[(sols[:, 0] == bestf1)]  # Select solutions that produced the highest corr. values
                        bestf2 = np.min(f1sols[:, 1])  # The solution that required to change the fewer features
                        best_sol = np.where((sols[:, 0] == bestf1) & (sols[:, 1] == bestf2))[0][0]
                        best_coeff2 = list(res.X[best_sol, :])
                        best_exp2 = set_args(best_exp, best_coeff2.copy())
                else:
                    best_exp2 = best_exp
                    best_coeff2 = best_coeff

                print('\tNew estimated expression: ' + str(best_exp2))
                # Check each coefficient
                for cn, cf in enumerate(best_coeff):
                    if np.abs((cf - best_coeff2[cn]) / cf) > 0.05:  # If their difference is greater than a threshold
                        if isinstance(final_coeff[cn], sym.Symbol):
                            final_coeff[cn] = [final_coeff[cn], functions[vn]]
                        elif isinstance(final_coeff[cn], list):
                            (final_coeff[cn]).append(functions[vn])
                        else:
                            final_coeff[cn] = functions[vn]

        for cf in range(len(final_coeff)):
            if isinstance(final_coeff[cf], list):
                final_coeff[cf] = sym.Symbol(str(final_coeff[cf]))

        # Reconstruct expression using new coefficients
        if best_exp_type == 'poly':
            final_exp = self.get_polynomial(coeff=final_coeff, calc_complexity=False)
        else:
            final_exp = set_args(best_exp, final_coeff)
        print('-----------------------------------------------------------')
        print('Final estimated expression: ' + str(final_exp) + '\n')
        return final_exp


if __name__ == '__main__':
    def g_fun(symb):
        return (1 - symb[1] ** 2) / (sym.sin(2 * sym.pi * symb[0]) + 1.5)


    def g_fun0(symb):
        return (1 - symb[0]) ** 2 + (sym.exp(symb[1])) / (sym.sin(2 * sym.pi * sym.exp(symb[2])) - 1.5)


    def g_fun1(symb):
        return sym.sin(1 / (symb[0] * 3 + 1.2))


    def n_fun(symb):
        return sym.sin(2 * sym.pi * (symb[0]) ** 2) * 7 - 1.5


    def p_fun(symb):
        return (1 - symb[0]) ** 2 + 100 * (symb[1] - symb[0] ** 2) ** 2


    def rosenbrock4D(symb):
        return ((1 - symb[0]) ** 2 + (1 - symb[2]) ** 2 + 100 * (symb[1] - symb[0] ** 2) ** 2 +
                100 * (symb[3] - symb[2] ** 2) ** 2)

    def sin4D(symb):
        return sym.sin(4 * symb[0] + 2 * np.pi * (symb[1]) * symb[2]) + sym.exp(1.2 * symb[3])


    # # Define problem
    # dataset = 'U0'
    # dataLoader = DataLoader(name=dataset)
    # X, Y, var_names = dataLoader.X, dataLoader.Y, dataLoader.names
    # n_features = X.shape[1]
    #
    # # Define NN and load weights
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # nn_model = NNModel(device=device, n_features=n_features)
    # root = get_project_root()
    # folder = os.path.join(root, "output//CVResults//" + dataset + "//iEQL-GA")
    # filepath = folder + "//weights-iEQL-GA-" + dataset + "-" + str(1)
    # nn_model.loadModel(filepath)
    #
    # xx = np.arange(-2, 2, 0.001)
    # xx = np.expand_dims(xx, axis=1)
    # yy, maxs, mins = minMaxScale(1 / (np.sin(2 * np.pi * np.exp(xx)) - 1.5))
    # ypred, _, _ = minMaxScale(1 / (np.sin(6.2094 * np.exp(1.0095 * xx) - 31.3487) - 1.5145))
    # ynn = applyMinMaxScale(nn_model.evaluateFold(xx), maxs=maxs, mins=mins)
    # plt.figure()
    # plt.scatter(xx, yy)
    # plt.scatter(xx, ypred)
    # plt.scatter(xx, ynn)
    # plt.legend(['Original equation $1 / (\sin(2 * \pi * e^x) - 1.5)$',
    #             'Learned equation $1/(\sin(6.2094* e^{1.0095*x}) - 31.3487) - 1.5145)$', 'Neural Network'], fontsize=14)

    # Create symbols and analyze each variable
    symbols = sym.symbols("{}:{}".format('x', 4))
    # operators = ['square', 'exp', 'sin', 'log', 'sqrt']
    operators = ['sin']
    for va in symbols[1:]:
        print("********************************")
        print("Analyzing variable " + str(va))
        print("********************************")
        fitter = FitCoefficients(input_var=va, symbols_list=symbols,
                                 gen_fun=sin4D(symbols), operators_list=operators)  # g_fun(symbols)
        fxp = fitter.fit()
