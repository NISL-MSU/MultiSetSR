import sympy as sp
from scipy.stats import pearsonr
from pymoo.core.problem import Problem
from pymoo.core.variable import Real
from sympy.utilities.iterables import flatten
from src.EquationLearning.models.functions import *
from src.EquationLearning.models.NNModel import NNModel
from src.EquationLearning.Optimization.CFE import get_args, set_args


#############################################################################
# HELPER FUNCTIONS
#############################################################################
def check_constant(xp):
    """Given an expression with coefficients a and b, check if b is added to another dependent function.
    E.g., get_c_args(sin(a*x + b + f0)) -> True
    :param xp: Symbolic expression"""
    args = xp.args
    check = False

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args = args[0]
        if args.is_number or isinstance(args, sp.Symbol):
            args = [args]
        else:
            args = args.args

    for arg in args:
        if isinstance(arg, sp.Symbol) and (str(arg) == 'b'):
            other_args = ['f' in str(oarg) for oarg in args]
            if True in other_args:
                return True
            else:
                return False
        else:  # If it's composed, explore a lower level of the tree
            check = check_constant(arg)
    return check


class SimpleFitProblem(Problem):
    """Fit the two constant coefficients that are added after merging two compatible expressions (MergeExpressions class)"""

    def __init__(self, xp_orig, list_symbols, gen_fun):
        """
        :param xp_orig: Original expression
        :param list_symbols: List of Sympy symbols of the original expression
        :param gen_fun: Generative function. It can be an NN model or a sympy function
        """
        self.xp_orig = xp_orig
        self.args = get_args(self.xp_orig, return_symbols=True)  # Return symbols and constants within an expression
        self.list_symbols = list_symbols
        self.iteration = 1
        self.y_temp = None  # Used to store previous evaluations

        if isinstance(gen_fun, NNModel):
            self.gen_fun = gen_fun
        else:  # If gen_fun is a symbolic expression, lambdify it to accelerate computations
            self.gen_fun = sp.lambdify(flatten(self.list_symbols), gen_fun)

        # Separate the variables that are explicitly shown in the expression
        self.xp_vars = sp.sympify(np.unique([str(arg) for arg in self.args if 'x' in str(arg)]))
        # Save positions where constants a or b appear
        self.a_pos = [i for i, arg in enumerate(self.args) if str(arg) == 'a'][0]
        self.b_pos = [i for i, arg in enumerate(self.args) if str(arg) == 'b'][0]
        self.f_pos = [i for i, arg in enumerate(self.args) if 'f' in str(arg)]

        # Declare optimization variables
        self.n_vars = 2 + len(self.f_pos)  # a and b plus any other dependencies that remain in the expression
        opt_variables = dict()
        xl, xu = [], []
        for k in range(self.n_vars):
            if k == 1 and check_constant(self.xp_orig):
                opt_variables[f"x{k:02}"] = Real(bounds=(0, 0))
                xl.append(0)
                xu.append(0)
            else:
                opt_variables[f"x{k:02}"] = Real(bounds=(-30, 30))
                xl.append(-30)
                xu.append(30)

        super().__init__(vars=opt_variables, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = []

        # After certain iterations, re-fix the values of the other variables that act as constants
        if (self.iteration - 1) % 20 == 0:
            values = np.zeros((len(self.list_symbols), 1000))
            for i in range(len(self.list_symbols)):
                if self.list_symbols[i] in self.xp_vars:
                    values[i, :] = np.random.uniform(-1, 1, size=1000)
                else:
                    values[i, :] = 2 * np.random.rand() - 1

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
        new_args = self.args.copy()
        for s in range(x.shape[0]):
            # Update function with new args
            new_args[self.a_pos] = x[s, 0]
            new_args[self.b_pos] = x[s, 1]
            for pF, f in enumerate(self.f_pos):
                new_args[f] = x[s, 2 + pF]
            new_func = set_args(self.xp_orig, list(new_args), return_symbols=True)
            # Lambdify to evaluate
            expr = sp.lambdify(flatten(self.list_symbols), new_func)
            # Evaluate new expression
            y_pred = expr(*list(values))

            # Calculate first objective
            f1.append(np.round(-np.abs(pearsonr(y, y_pred)[0]), 4))
        out["F"] = np.array(f1)
        self.iteration += 1

        print("\tIteration " + str(self.iteration) + "/30...", end='\r')
