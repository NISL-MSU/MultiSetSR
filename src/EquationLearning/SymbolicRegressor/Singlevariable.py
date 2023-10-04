import warnings
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination.robust import RobustTermination
from src.EquationLearning.models.utilities_expressions import *
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from src.EquationLearning.Optimization.PolynomialFitting import PolynomialFitting
from src.EquationLearning.Optimization.CoefficientProblem import CoefficientProblem
from src.EquationLearning.Optimization.CFE import CFE, get_args, set_args, verify_dependency
from src.EquationLearning.Optimization.DoubleCoefficientProblem import DoubleCoefficientProblem
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination


warnings.filterwarnings("ignore")


class SingleVariable:
    def __init__(self, input_var, symbols_list, gen_fun, operators_list, limits, th_corr: float = 0.99):
        """Class used to find univariate expressions
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
            self.gen_fun = sp.lambdify(flatten(symbols_list), gen_fun)

        self.termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-5), period=50)

    def fit(self):
        """Find the expression that yields the best correlation"""
        # best_values = None
        #############################################################################
        # Polynomial fitting
        #############################################################################
        print('Polynomial fitting:')
        problem = PolynomialFitting(gen_fun=self.gen_fun, variable=self.var_ind, limits=self.limits)
        coeff, best_corr = problem.fit()
        best_values, best_deg = problem.best_values, problem.best_deg
        coeff = coeff[:-1]   # Get rid of last constant
        max_coeff = np.max(coeff)
        best_coeff = coeff / max_coeff
        best_coeff[np.abs(best_coeff) < 0.01] = 0
        if best_coeff[0] == 0:
            best_coeff = best_coeff[1:]
            best_deg -= 1
        best_exp, best_comp = get_polynomial(variable=self.variable, coeff=best_coeff)
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
                xp, comp = get_expression(variable=self.variable, coeff=coeff, operator=operator, inv=problem.inv)
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
                xp, comp = get_expression_nested(variable=self.variable, coeff=coeff, operator1=operator,
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
        best_exp = sp.factor(best_exp)
        print('-----------------------------------------------------------')
        print('Best expression: ' + str(best_exp))
        print('-----------------------------------------------------------')

        #############################################################################
        # Detect dependency between variables
        #############################################################################
        functions = sp.symbols("{}:{}".format('f', len(self.symbols_list)))
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
                                                     seed=3, only_resample_1v=vn, limits=self.limits)
                        best_coeff2, best_corr = problem2.fit(degree=best_deg)
                        best_coeff2 = best_coeff2[:-1] / max_coeff  # Get rid of last constant
                        best_coeff2[np.abs(best_coeff2) < 0.01] = 0
                        best_exp2, _ = get_polynomial(variable=self.variable, coeff=best_coeff2)
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
                        if isinstance(final_coeff[cn], sp.Symbol):
                            final_coeff[cn] = [final_coeff[cn], functions[vn]]
                        elif isinstance(final_coeff[cn], list):
                            (final_coeff[cn]).append(functions[vn])
                        else:
                            final_coeff[cn] = functions[vn]

        for cf in range(len(final_coeff)):
            if isinstance(final_coeff[cf], list):
                final_coeff[cf] = sp.Symbol(str(final_coeff[cf]))

        # Reconstruct expression using new coefficients
        if best_exp_type == 'poly':
            final_exp = get_polynomial(variable=self.variable, coeff=final_coeff, calc_complexity=False)
        else:
            final_exp = set_args(best_exp, final_coeff)
        print('-----------------------------------------------------------')
        print('Final estimated expression: ' + str(final_exp) + '\n')
        return final_exp


if __name__ == '__main__':
    def g_fun(symb):
        return (1 - symb[1] ** 2) / (sp.sin(2 * sp.pi * symb[0]) + 1.5)


    def g_fun0(symb):
        return (1 - symb[0]) ** 2 + (sp.exp(symb[1])) / (sp.sin(2 * sp.pi * sp.exp(symb[2])) - 1.5)


    def g_fun1(symb):
        return sp.sin(1 / (symb[0] * 3 + 1.2))


    def n_fun(symb):
        return sp.sin(2 * sp.pi * (symb[0]) ** 2) * 7 - 1.5


    def p_fun(symb):
        return (1 - symb[0]) ** 2 + 100 * (symb[1] - symb[0] ** 2) ** 2


    def rosenbrock4D(symb):
        return ((1 - symb[0]) ** 2 + (1 - symb[2]) ** 2 + 100 * (symb[1] - symb[0] ** 2) ** 2 +
                100 * (symb[3] - symb[2] ** 2) ** 2)

    def sin4D(symb):
        return sp.sin(4 * symb[0] + 2 * np.pi * (symb[1]) * symb[2]) + sp.exp(1.2 * symb[3])


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
    symbols = sp.symbols("{}:{}".format('x', 4))
    # operators = ['square', 'exp', 'sin', 'log', 'sqrt']
    operators = ['sin']
    for va in symbols[1:]:
        print("********************************")
        print("Analyzing variable " + str(va))
        print("********************************")
        fitter = SingleVariable(input_var=va, symbols_list=symbols,
                                 gen_fun=sin4D(symbols), operators_list=operators)  # g_fun(symbols)
        fxp = fitter.fit()
