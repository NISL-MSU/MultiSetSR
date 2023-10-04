import numpy as np
import sympy as sp
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from src.EquationLearning.Optimization.CFE import get_args, set_args
from src.EquationLearning.Optimization.SimpleFitProblem import SimpleFitProblem


def compatible(exp1, exp2):
    """Verify if exp1 and exp2 are compatible and could be merged"""
    # If neither expression is inside an operator. Ex. x1 * f2 and x2 ** 2 * x1
    if ('(' not in str(exp1)) and ('(' not in str(exp2)):
        return True

    # If their main operator is different, they're incompatible. Ex. x1 * f2 and sin(x2 ** 2 * x1)
    if exp1.func != exp2.func:
        return False

    # . Eg. x1 * f2 and x2 ** 2 * sin(x1)
    return True


def get_dependent_independent_vars(exp1):
    """Retrieve the main variable in the expression exp1 and what are the dependent variables present in it"""
    t = str(exp1).find("x")
    x_var = str(exp1)[t:t + 2]  # The name of the main variable in xp1 (e.g., 'x2')
    if '[' not in str(exp1) and ']' not in str(exp1):  # If there's no composition of dependent variables
        dep_vars = [str(exp1)[t:t + 2] for t, ch in enumerate(str(exp1)) if ch == 'f']
        dep_vars = np.unique(dep_vars)  # The list of dependent variables in xp1 (e.g. 'f1', 'f3')
    else:
        temp_exp1 = str(exp1)
        dep_vars = []
        # Find composition of dependent variables. Ex: Given [f1, f2]x0 + f2, it finds [f1, f2]
        while '[' in temp_exp1:
            t1, t2 = temp_exp1.find('['), temp_exp1.find(']')
            dep_vars.append(temp_exp1[t1:t2 + 1])
            temp_exp1 = temp_exp1.replace(temp_exp1[t1:t2 + 1], '')
        # Now consider remaining dependent variables
        dep_vars_alone = [str(temp_exp1)[t:t + 2] for t, ch in enumerate(str(temp_exp1)) if ch == 'f']
        dep_vars = dep_vars + dep_vars_alone
        dep_vars = np.unique(dep_vars)
    return x_var, dep_vars


def find_function(exp1, x1):
    """Find first function of x1 inside exp1 that is independent of other variables"""
    args1 = exp1.args
    f1 = None
    while len(args1) > 0:
        if x1 in str(args1[0]) and 'f' not in str(args1[0]):
            f1 = args1[0]
            break
        if len(args1) > 1:
            if x1 in str(args1[1]) and 'f' not in str(args1[1]):
                f1 = args1[1]
                break
        if x1 in str(args1[0]):
            exp1 = args1[0]
            args1 = args1[0].args
        elif len(args1) > 1:
            exp1 = args1[1]
            args1 = args1[1].args
    return f1, exp1


def merge_expressions(exp1, exp2, x1, x2):
    """Find coincidences in Sympy expressions and merge them"""
    # Parse the tree until functions of x1 and x2 are found
    fx1, args1 = find_function(exp1, x1)
    fx2, _ = find_function(exp2, x2)
    # Replace unknown functions using the functions that were found in the previous step
    comb = str(args1).replace('f' + x2[1], str(fx2))
    merged = str(exp1).replace(str(args1), str(sp.Symbol('a') * sp.sympify(comb) + sp.Symbol('b')))
    return sp.sympify(merged)


class MergeExpressions:

    def __init__(self, gen_fun, expressions: list, symbols_list: list, limits: list):
        """
        Class used to organize a list of expressions as independent expressions before they are fed into the GP.
        Expressions that are dependent on others are properly merged.
        :param gen_fun: Generative function. It can be an NN model or a sympy function.
        :param expressions: List of initial Sympy expressions. Original variables are denoted 'x'. Functions of other
        variables are denoted with 'f'.
        :param symbols_list: List of symbolic symbols present in the system
        :param limits: Lower and upper bounds (tuple) of each explanatory variable
        """
        self.limits = limits
        self.gen_fun = gen_fun
        self.symbols_list = symbols_list
        self.orig_expressions = expressions
        self.dependent_expressions, self.independent_expressions = self._separate_additive_terms()

    def _separate_additive_terms(self):
        dependent_expressions, independent_expressions = [], []
        for xp in self.orig_expressions:
            dependent_terms, independent_terms = [], []
            if isinstance(xp, sp.Add):
                for arg in xp.args:
                    if 'f' in str(arg):  # If there's an 'f' it's because that term is a function of another variable
                        dependent_expressions.append(arg)
                    else:
                        independent_terms.append(arg)
                independent_expressions.append(sp.Add(*independent_terms))
            else:
                if 'f' in str(xp):
                    dependent_expressions.append(xp)
                else:
                    independent_expressions.append(xp)
        dependent_expressions.sort(key=lambda x: len(str(x)))
        return dependent_expressions, independent_expressions

    def merge_dependent_expressions(self):
        algorithm = GA(pop_size=150)

        while len(self.dependent_expressions) > 0:  # Repeat until there are no more unmatched expressions
            # Pop the first expression of the list and merge
            exp1 = self.dependent_expressions.pop(0)

            # Identify what is the main variable in xp1 and with which variables it relates to
            x_var, dep_vars = get_dependent_independent_vars(exp1)

            # Find compatible candidates
            for candidate in self.dependent_expressions:
                # Verify if the main variable in this candidate is a dependent variable in exp1
                x_cand, dep_vars_cand = get_dependent_independent_vars(candidate)
                if ('f' + x_cand[1] not in str(dep_vars)) or ('f' + x_var[1] not in str(dep_vars_cand)):
                    continue
                # Verify if x_var and x_can have the same operator. Ex: sin(x1 + f2) and sin(exp(x2) + f1)
                if not compatible(exp1, candidate):
                    continue
                # Merge expressions
                merged_expression = merge_expressions(exp1=exp1, exp2=candidate, x1=x_var, x2=x_cand)
                # Fit added linear coefficients in the merged expression
                problem2 = SimpleFitProblem(xp_orig=merged_expression, list_symbols=self.symbols_list,
                                            gen_fun=self.gen_fun)
                res = minimize(problem2, algorithm, ('n_gen', 30), seed=4, verbose=False)
                # Select best coefficients
                if len(res.F) > 1:
                    sols = res.F
                    best_sol_ind = np.argmin(sols)
                    best_sol = res.X[best_sol_ind, :]
                else:
                    best_sol = res.X
                # Update merged expression with fitted coefficients
                args = get_args(merged_expression, return_symbols=True)
                a_pos = [i for i, arg in enumerate(args) if str(arg) == 'a'][0]
                b_pos = [i for i, arg in enumerate(args) if str(arg) == 'b'][0]
                args[a_pos] = best_sol[0]
                args[b_pos] = best_sol[1]
                merged_expression = set_args(merged_expression, list(args), return_symbols=True)

                # Remove the candidate from the stack
                self.dependent_expressions.remove(candidate)
                # If the merged expression is still dependent on other variables, add it to the stack
                if 'f' in str(merged_expression):
                    self.dependent_expressions = [merged_expression] + self.dependent_expressions
                else:
                    self.independent_expressions.append(merged_expression)
        return self.independent_expressions


if __name__ == '__main__':
    # # Example 1
    # symbols = sp.symbols("{}:{}".format('x', 4))
    # functions = sp.symbols("{}:{}".format('f', 4))
    #
    # xp1 = functions[1] * symbols[0] ** 2 + 100.0 * symbols[0] ** 4 - 2.0 * symbols[0]
    # xp2 = functions[0] * symbols[1] + 100.0 * symbols[1] ** 2
    # xp3 = functions[3] * symbols[2] ** 2 + 100.0 * symbols[2] ** 4 - 2.0 * symbols[2]
    # xp4 = functions[2] * symbols[3] + 100.0 * symbols[3] ** 2
    #
    # merger = MergeExpressions(expressions=[xp1, xp2, xp3, xp4])
    # Example 1
    # symbols = sp.symbols("{}:{}".format('x', 4))
    # functions = sp.symbols("{}:{}".format('f', 4))
    #
    # xp1 = functions[1] * symbols[0] ** 2 + 0.8122 * symbols[0] ** 4 - 0.0149 * symbols[0]
    # xp2 = functions[0] * symbols[1] + 100.0 * symbols[1] ** 2
    # xp3 = functions[3] * symbols[2] ** 2 + 100.0 * symbols[2] ** 4 - 2.0 * symbols[2]
    # xp4 = functions[2] * symbols[3] + 100.0 * symbols[3] ** 2
    #
    # merger = MergeExpressions(expressions=[xp1, xp2, xp3, xp4], symbols_list=symbols, gen_fun=None)

    # Example 2
    def sin4D(symb):
        return sp.sin(4 * symb[0] + 2 * np.pi * (symb[1]) * symb[2])
    symbols = sp.symbols("{}:{}".format('x', 3))
    functions = sp.symbols("{}:{}".format('f', 3))

    xp1 = sp.sin(sp.Symbol(str([functions[1], functions[2]])) + 4.0029 * symbols[0])
    xp2 = sp.sin(functions[0] + functions[2] * symbols[1])
    xp3 = sp.sin(functions[0] + functions[1] * symbols[2])

    merger = MergeExpressions(expressions=[xp1, xp2, xp3], symbols_list=symbols, gen_fun=sin4D(symbols))
    res = merger.merge_dependent_expressions()
    print("List of merged expressions = " + str(res))
