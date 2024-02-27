import os
import copy
import time
import types
import sympy
import warnings
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
from sympy import sympify, lambdify
from sympy.utilities.iterables import flatten
from src.EquationLearning.Data.dclasses import Equation
from src.EquationLearning.Data.generator import Generator
from src.EquationLearning.Data.dclasses import SimpleEquation
from src.utils import load_metadata_hdf5, load_eq, get_project_root
from src.EquationLearning.Data.sympy_utils import numeric_to_placeholder
from src.EquationLearning.Data.data_utils import sample_symbolic_constants, bounded_operations
from src.EquationLearning.models.utilities_expressions import add_constant_identifier, \
    avoid_operations_between_constants, get_op_constant
from src.EquationLearning.models.utilities_expressions import get_args, set_args

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Dataset:
    def __init__(
            self,
            data_path: Path,
            cfg,
            mode: str
    ):
        metadata = load_metadata_hdf5(Path(os.path.join(get_project_root(), "src/EquationLearning/Data", data_path)))
        cfg.total_variables = metadata.total_variables
        cfg.total_coefficients = metadata.total_coefficients
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        self.id2word = metadata.id2word
        self.data_path = Path(os.path.join(get_project_root(), "src/EquationLearning/Data", data_path))
        self.mode = mode
        self.cfg = cfg

    def __getitem__(self, index):
        eq_string, curr = '', None
        while 'x' not in str(eq_string):
            eq = load_eq(self.data_path, index, self.eqs_per_hdf)
            if 'cc' in eq.variables:
                eq.variables.remove('cc')  # In case the dummy variable was not removed during the eq generation process
            code = types.FunctionType(eq.code, globals=globals(), name="f")
            consts, initial_consts = sample_symbolic_constants(eq, self.cfg.constants)
            eq_string = str(eq.expr)

            for c in consts.keys():
                if self.cfg.predict_c:
                    eq_string = eq_string.replace(c, str(consts[c]))
                else:
                    eq_string = eq_string.replace(c, str(initial_consts[c]))

            eq_string = sympify(eq_string).evalf()
            # Simplify exponent of constant and constant multiplied by constants
            eq_string = avoid_operations_between_constants(eq_string)
            eq.expr = eq_string
            eq_sympy_infix = constants_to_placeholder(eq_string, eq.coeff_dict)
            eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)

            t = tokenize(eq_sympy_prefix, self.word2id)
            curr = Equation(code=code, expr=eq_string, coeff_dict=consts, variables=eq.variables,
                            support=eq.support, tokenized=t, valid=True)

        return curr

    def __len__(self):
        return self.len


def constants_to_placeholder(s, coeffs, symbol="c"):
    sympy_expr = s
    for si in set(coeffs.keys()):
        if "c" in si:
            sympy_expr = sympy_expr.subs(si, symbol)
            sympy_expr = sympy_expr.subs(si, symbol)
    return sympy_expr


def tokenize(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = [word2id["S"]]
    for i in prefix_expr:
        if 'x' in i and i != 'exp':
            i = 'x_1'
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    return tokenized_expr


def de_tokenize(tokenized_expr, id2word: dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[i]:
            break
        else:
            prefix_expr.append(id2word[i])
    return prefix_expr


def sample_support(curr_p, cfg, Xbounds, extrapolate=False):
    sym = []
    for sy in cfg.total_variables:
        if 'x' in sy:
            if not extrapolate:
                curr = np.random.uniform(Xbounds[0], Xbounds[1], size=int(curr_p))
            else:
                curr = np.random.uniform(Xbounds[0] * 2, Xbounds[1] * 2, size=int(curr_p))
        else:
            curr = np.zeros(int(curr_p))
        sym.append(curr)
    return np.stack(sym)


def sample_constants(eq, cfg):
    eq2 = copy.deepcopy(eq)
    exp = sympify(eq2.expr)
    # with sympy.evaluate(False):  # This prevents factorization
    for c in cfg.total_coefficients:
        if c[:2] == "cm":
            if c in str(exp):  # and not isinstance(eq.coeff_dict[c], int):  (if it's 0 or 1, the constant is not there)
                # Check if constant is inside an exponential  or hyperbolic function. Limit its values from -3 to 3
                op = get_op_constant(exp, c)
                if not isinstance(op, bool) and op is not None:
                    if ('exp' in op) or ('sinh' in op) or ('cosh' in op) or ('tanh' in op):
                        val = np.random.uniform(low=-3, high=3)
                        while np.abs(val) < 0.05:  # Avoid too small values
                            val = np.random.uniform(low=-3, high=3)
                        exp = exp.subs(c, val)
                    elif ('sin' in op) or ('cos' in op) or ('tan' in op):
                        val = np.random.uniform(low=-2, high=2)
                        while np.abs(val) < 0.1:  # Avoid too small values
                            val = np.random.uniform(low=-2, high=2)
                        exp = exp.subs(c, val)
                    elif isinstance(op, tuple):
                        if 'Pow' in op[0] and op[1] == 4:
                            val = np.random.uniform(low=-3, high=3)
                            exp = exp.subs(c, val)
                        else:
                            exp = exp.subs(c, np.random.uniform(low=cfg.constants.multiplicative.min,
                                                                high=cfg.constants.multiplicative.max))
                    else:
                        exp = exp.subs(c, np.random.uniform(low=cfg.constants.multiplicative.min,
                                                            high=cfg.constants.multiplicative.max))
                else:
                    exp = exp.subs(c, np.random.uniform(low=cfg.constants.multiplicative.min,
                                                        high=cfg.constants.multiplicative.max))
        elif c[:2] == "ca":
            if c in eq.coeff_dict and not isinstance(eq.coeff_dict[c], int):
                # Check if constant is inside an exponential  or hyperbolic function. Limit its values from -3 to 3
                op = get_op_constant(exp, c)
                if not isinstance(op, bool) and op is not None:
                    if ('exp' in op) or ('sinh' in op) or ('cosh' in op) or ('tanh' in op):
                        exp = exp.subs(c, np.random.uniform(low=-1, high=1))
                    else:
                        exp = exp.subs(c, np.random.uniform(low=cfg.constants.additive.min,
                                                            high=cfg.constants.additive.max))
                else:
                    exp = exp.subs(c, np.random.uniform(low=cfg.constants.additive.min,
                                                        high=cfg.constants.additive.max))
    eq2.expr = exp
    return eq2


def evaluate_and_wrap(eq, cfg, word2id, return_exprs=True, extrapolate=False, n_sets=None, xmin=None, xmax=None):
    """
    Given a list of skeleton equations, sample their inputs and corresponding constants and evaluate their results
    :param eq: List of skeleton equations
    :param cfg: configuration file
    :param word2id: Dictionary used to tokenize equations
    :param return_exprs: If True, return the expression that were sampeld during the process
    :param extrapolate: If True, sample support values that go beond the training domain
    :param n_sets: If not None, it explicitly specifies the number of sets to be generated
    :param xmin: If not None, it explicitly specifies the minimum support value to be used for generation
    :param xmax: If not None, it explicitly specifies the minimum support value to be used for generation
    """
    exprs = eq.expr
    curr_p = cfg.max_number_of_points
    # # Uncomment the code below if you have a specific skeleton from which you want to sample data as an example
    # sk = sympy.sympify('c*x_1 + c*sin(c*x_1 + c) + c')
    # sk, _, _ = add_constant_identifier(sk)
    # coeff_dict = dict()
    # var = None
    # for constant in sk.free_symbols:
    #     if 'c' in str(constant):
    #         coeff_dict[str(constant)] = constant
    #     if 'x' in str(constant):
    #         var = constant
    # eq = SimpleEquation(expr=sk, coeff_dict=coeff_dict, variables=[var])
    # exprs = eq.expr

    # Randomly stretch or shrink input domain. E.g., from [-10, 10] to [-5, 5] or [-2, 2]
    divider = 1  # np.random.randint(2, 10) / 2
    if xmin is None and xmax is None:
        minX, maxX = cfg.fun_support.min / divider, cfg.fun_support.max / divider
        Xbounds = [minX, maxX]
    else:
        minX, maxX = xmin, xmax
        Xbounds = [minX, maxX]

    # Check if there's any bounded operation in the expression
    bounded, double_bounded, op_with_singularities, trig_functions = bounded_operations()
    is_bounded = False
    if any([b in str(exprs) for b in list(bounded.keys())]) or any([b in str(exprs) for b in trig_functions]) or \
            any([b in str(exprs) for b in list(double_bounded.keys())]) or '/' in str(exprs) or \
            any([b in str(exprs) for b in list(op_with_singularities)]) or '**0.5' in str(exprs) or '**(-' in str(exprs):
        is_bounded = True

    # If the number of sets is explicitly specified
    Ns = cfg.number_of_sets
    if n_sets is not None:
        Ns = n_sets

    # Create "Ns" vectors to store the evaluations
    X = np.zeros((curr_p, Ns), dtype=np.float32)
    Y = np.zeros((curr_p, Ns), dtype=np.float32)
    # Sample constants "Ns" times
    support = sample_support(curr_p, cfg, Xbounds, extrapolate)
    tokenized = None
    n_set = 0
    sampled_exprs = [''] * Ns

    ###########################################################
    # Start Ns iterations
    ###########################################################
    tic_main = time.time()
    skeleton_eq, coeff_dict = None, None
    while n_set < Ns:  # for loop but using while because it may restart
        # If more than 20 seconds time has passed, skip this equation
        toc_main = time.time()
        if toc_main - tic_main > 50:
            return None
        restart = False

        if n_set > 0:
            consts = sample_constants(skeleton_eq, cfg)
        else:
            consts = sample_constants(eq, cfg)
            coeff_dict = eq.coeff_dict
        # Use consts.expr as the expression that will be used for evaluation
        new_expr = consts.expr
        # If there's a coefficient that is too big, clip it to 50
        if any(np.abs(get_args(new_expr)) > 50):
            args = get_args(new_expr)
            args = [arg if np.abs(arg) <= 50 else 50 * np.sign(arg) for arg in args]
            new_expr = set_args(new_expr, args)

        # If the expression contains a bounded operation, modify the constants to avoid NaN values
        # new_expr = sympify('-3.8967919334523*x_1*log(-0.183763376889317*x_1)/sin(0.124550801335169*x_1) + 0.238240379581665')
        if is_bounded:
            result = modify_constants_avoidNaNs(new_expr, support, bounded_operations(), curr_p, Xbounds,
                                                variable=eq.variables, extrapolate=extrapolate)
            if result is None:
                n_set = 0
                continue
            support, new_expr, _ = result
            # If the new expression needs coefficients that are too large, try again
            if any(np.abs(get_args(new_expr)) > 100):
                restart = True

        # Convert expression to prefix format and tokenize
        expr_with_placeholder = numeric_to_placeholder(new_expr)
        eq_sympy_infix = constants_to_placeholder(expr_with_placeholder, coeff_dict)
        eq_sympy_infix = sympify(str(eq_sympy_infix).replace('*(c + x_1)', '*(c*x_1 + c)'))
        eq_sympy_infix = sympify(str(eq_sympy_infix).replace('c*(c*x_1 + c)', 'c*(x_1 + c)'))
        # eq_sympy_infix = sympify(str(eq_sympy_infix2).replace('(x_1', '(c*x_1'))
        eq_sympy_infix = sympify(str(eq_sympy_infix).replace('+ (c*x_1 + c)**', '+ c*(x_1 + c)**'))
        eq_sympy_infix = sympify(str(eq_sympy_infix).replace('+ Abs(c*x_1)', '+ c*Abs(x_1)'))
        eq_sympy_infix = sympify(str(eq_sympy_infix).replace('c*Abs(c*x_1)', 'c*Abs(x_1)'))
        eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        t = tokenize(eq_sympy_prefix, word2id)
        skeleton, _, _ = add_constant_identifier(eq_sympy_infix)

        # Make sure that all sets have the same tokenized skeleton
        if n_set == 0:
            coeff_dict = dict()
            var = None
            for constant in skeleton.free_symbols:
                if 'c' in str(constant):
                    coeff_dict[str(constant)] = constant
                if 'x' in str(constant):
                    var = constant
            skeleton_eq = SimpleEquation(expr=skeleton, coeff_dict=coeff_dict, variables=[var])
            tokenized = t.copy()
            exprs = eq_sympy_infix
        else:
            tic = time.time()  # Start Time
            big_coef = False
            while big_coef or t != tokenized:
                consts = sample_constants(skeleton_eq, cfg)
                new_expr = consts.expr
                # If there's a coefficient that is too big, clip it to 10
                if any(np.abs(get_args(new_expr)) > 50):
                    args = get_args(new_expr)
                    args = [arg if np.abs(arg) <= 50 else 50 * np.sign(arg) for arg in args]
                    new_expr = set_args(new_expr, args)
                # If the expression contains a bounded operation, modify the constants to avoid NaN values
                if is_bounded:
                    support, new_expr, _ = modify_constants_avoidNaNs(new_expr, support, bounded_operations(), curr_p,
                                                                      Xbounds,
                                                                      variable=eq.variables, extrapolate=extrapolate)
                # If the new expression needs coefficients that are too large, try again
                if any(np.abs(get_args(new_expr)) > 100):
                    big_coef = True
                else:
                    big_coef = False
                expr_with_placeholder2 = numeric_to_placeholder(new_expr)
                eq_sympy_infix2 = constants_to_placeholder(expr_with_placeholder2, coeff_dict)
                eq_sympy_infix2 = sympify(str(eq_sympy_infix2).replace('*(c + x_1)', '*(c*x_1 + c)'))
                eq_sympy_infix2 = sympify(str(eq_sympy_infix2).replace('c*(c*x_1 + c)', 'c*(x_1 + c)'))
                # eq_sympy_infix2 = sympify(str(eq_sympy_infix2).replace('(x_1', '(c*x_1'))
                eq_sympy_infix2 = sympify(str(eq_sympy_infix2).replace('+ (c*x_1 + c)**', '+ c*(x_1 + c)**'))
                eq_sympy_infix2 = sympify(str(eq_sympy_infix2).replace('+ Abs(c*x_1)', '+ c*Abs(x_1)'))
                eq_sympy_infix2 = sympify(str(eq_sympy_infix2).replace('c*Abs(c*x_1)', 'c*Abs(x_1)'))
                eq_sympy_prefix2 = Generator.sympy_to_prefix(eq_sympy_infix2)
                t = tokenize(eq_sympy_prefix2, word2id)
                # Calculate how much time has passed
                toc = time.time()
                if toc - tic > 18:
                    restart = True
                    break

        # If there's a coefficient that is too small, replace it by zero
        if any(np.abs(get_args(new_expr)) < 0.0001):
            args = get_args(new_expr)
            args = [arg if np.abs(arg) >= 0.0001 else 0 for arg in args]
            new_expr2 = set_args(new_expr, args)
            # If after simplification of the new expression, the variable is removed, do not consider this equation
            if 'x' not in str(new_expr2):
                return None
            else:
                new_expr = new_expr2

        if restart:
            n_set = 0  # If too much time has passed trying to find an equation with the same skeleton, restart
        else:
            # Lambdify new function
            function = lambdify(flatten(eq.variables), new_expr)
            # Evaluate values from the support vector into the new expression
            vals = np.array(function(*list(support)))
            # # Drop input-output pairs containing NaNs and entries with an absolute value of y above 10000
            # isnans, indices_to_remove = is_nan(vals), []
            # if any(isnans):
            #     indices_to_remove = np.where(isnans)[0]
            # new_support = np.delete(np.array(support), indices_to_remove)
            # vals = np.delete(np.array(vals), indices_to_remove)
            #
            # # If some elements have been dropped, keep sampling until reaching a population of 'curr_p' size
            # tic = time.time()  # Start Time
            # while len(vals) < curr_p:
            #     missing = curr_p - len(vals)
            #     # Sample constants "Ns/2" times and remove NaNs from them
            #     extra_support = sample_support(eq, int(curr_p/2), cfg, extrapolate)
            #     extra_vals = np.array(function(*list(extra_support)))
            #     isnans, indices_to_remove = is_nan(extra_vals), []
            #     if any(isnans):
            #         indices_to_remove = np.where(isnans)[0]
            #     extra_support = np.delete(np.array(extra_support), indices_to_remove)
            #     extra_vals = np.delete(np.array(extra_vals), indices_to_remove)
            #     if len(extra_support) >= missing:
            #         new_support = np.append(new_support, extra_support[:missing])
            #         vals = np.append(vals, extra_vals[:missing])
            #     else:
            #         new_support = np.append(new_support, extra_support)
            #         vals = np.append(vals, extra_vals)
            #     # If more than 20 seconds time has passed, skip this equation
            #     toc = time.time()
            #     if toc - tic > 10:
            #         return None

            # if n_set is None:
            #     scaling_factor = 20 / (np.max(support) - np.min(support))
            #     support = (support - np.min(support)) * scaling_factor - 10
            X[:, n_set] = support
            Y[:, n_set] = vals
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X[:, n_set] = X[indices, n_set]
            Y[:, n_set] = Y[indices, n_set]
            sampled_exprs[n_set] = new_expr

            if np.std(vals) > 0.01 and not any(is_nan(vals)):  # If the result is too flat, try again
                if np.isnan(vals).any() or np.isinf(vals).any() or np.isnan(np.mean(vals)) or \
                        np.isinf(np.mean(vals)) or np.isnan(np.std(vals)) or np.isinf(np.std(vals)):
                    return None

                # If there's not enough samples between -3 and 3, try again (it avoids having functions with big gaps)
                selected_indices = np.where((support >= -4) & (support <= 4))[0]
                if len(selected_indices) < 3000 or np.std(support) < 3 or np.max(support) - np.min(support) < 15:
                    continue

                n_set += 1

            elif any(is_nan(vals)):  # If there's an undefined value, try again
                continue

            elif np.std(vals) == 0 or 'x_1' not in str(new_expr) or 'x_1' not in str(exprs):
                return None

    if return_exprs:
        return X, Y, tokenized, exprs, sampled_exprs
    else:
        return X, Y, tokenized, exprs


def skeleton2dataset(sk, Xbounds, cfg, word2id, extrapolate=False):
    """
    Given a symbolic skeleton, sample their inputs and corresponding constants and evaluate their results
    :param sk: List of skeleton equations
    :param Xbounds: Minimum and maximum support values
    :param cfg: configuration file
    :param word2id: Dictionary used to tokenize equations
    :param extrapolate: If True, sample support values that go beyond the training domain
    """
    # Read skeleton and identify numeric constants
    sk = sympy.sympify(sk)
    sk, _, _ = add_constant_identifier(sk)
    coeff_dict = dict()
    var = None
    for constant in sk.free_symbols:
        if 'c' in str(constant):
            coeff_dict[str(constant)] = constant
        if 'x' in str(constant):
            var = constant
    eq = SimpleEquation(expr=sk, coeff_dict=coeff_dict, variables=[var])
    exprs = eq.expr
    curr_p = cfg.max_number_of_points

    # Check if there's any bounded operation in the expression
    bounded, double_bounded, op_with_singularities, trig_functions = bounded_operations()
    is_bounded = False
    if any([b in str(exprs) for b in list(bounded.keys())]) or any([b in str(exprs) for b in trig_functions]) or \
            any([b in str(exprs) for b in list(double_bounded.keys())]) or '/' in str(exprs) or \
            any([b in str(exprs) for b in list(op_with_singularities)]) or '**0.5' in str(exprs) or '**(-' in str(exprs):
        is_bounded = True

    # Sample constants "cfg.number_of_sets" times
    support = sample_support(curr_p, cfg, Xbounds, extrapolate)

    ###########################################################
    # Start sampling process
    ###########################################################
    consts = sample_constants(eq, cfg)
    new_expr = consts.expr
    # If there's a coefficient that is too big, clip it to 50
    if any(np.abs(get_args(new_expr)) > 50):
        args = get_args(new_expr)
        args = [arg if np.abs(arg) <= 50 else 50 * np.sign(arg) for arg in args]
        new_expr = set_args(new_expr, args)

    # If the expression contains a bounded operation, modify the constants to avoid NaN values
    if is_bounded:
        result = modify_constants_avoidNaNs(new_expr, support, bounded_operations(), curr_p, Xbounds,
                                            variable=eq.variables, extrapolate=extrapolate)
        support, new_expr, _ = result

    # Convert expression to prefix format and tokenize
    expr_with_placeholder = numeric_to_placeholder(new_expr)
    eq_sympy_infix = constants_to_placeholder(expr_with_placeholder, coeff_dict)
    eq_sympy_infix = sympify(str(eq_sympy_infix).replace('*(c + x_1)', '*(c*x_1 + c)'))
    # eq_sympy_infix = sympify(str(eq_sympy_infix).replace('(x_1', '(c*x_1'))
    eq_sympy_infix = sympify(str(eq_sympy_infix).replace('+ (c*x_1 + c)**', '+ c*(c*x_1 + c)**'))
    eq_sympy_infix = sympify(str(eq_sympy_infix).replace('+ Abs(c*x_1)', '+ c*Abs(x_1)'))
    eq_sympy_infix = sympify(str(eq_sympy_infix).replace('c*Abs(c*x_1)', 'c*Abs(x_1)'))
    eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
    t = tokenize(eq_sympy_prefix, word2id)
    skeleton, _, _ = add_constant_identifier(eq_sympy_infix)
    exprs = eq_sympy_infix

    # If there's a coefficient that is too small, replace it by zero
    if any(np.abs(get_args(new_expr)) < 0.0001):
        args = get_args(new_expr)
        args = [arg if np.abs(arg) >= 0.0001 else 0 for arg in args]
        new_expr2 = set_args(new_expr, args)
        # If after simplification of the new expression, the variable is removed, do not consider this equation
        if 'x' not in str(new_expr2):
            return None
        else:
            new_expr = new_expr2

    # Lambdify new function
    function = lambdify(flatten(eq.variables), new_expr)
    # Evaluate values from the support vector into the new expression
    vals = np.array(function(*list(support)))

    return support[0, :], vals, t, exprs, new_expr


def is_nan(x, bound=None):
    """Encompasses NaN, Inf values, and outliers"""
    # mean = np.mean(x)
    # std = np.std(x)
    # threshold = 5
    # outliers = []
    # for xx in x:
    #     z_score = (xx - mean) / std
    #     try:
    #         if abs(z_score) > threshold:
    #             outliers.append(True)
    #         else:
    #             outliers.append(False)
    #     except:
    #         print()
    if bound is None:
        return np.iscomplex(x) + np.isnan(x) + np.isinf(x) + (np.abs(x) > 100000)  # + np.array(outliers)
    else:
        return np.iscomplex(x) + np.isnan(x) + np.isinf(x) + (np.abs(x) > 100000) + (np.abs(x) > bound)  # + np.array(outliers)


def modify_constants_avoidNaNs(expr, x, bounded_ops, npoints, Xbounds, variable=sympy.sympify('x_1'), extrapolate=False,
                               avoid_x=None, threshold=0.05):
    """Modifies the constants inside unary bounded operations to avoid generating NaN values
    :param expr: Original expression
    :param x: Support vector of variable x values
    :param bounded_ops: Dictionary including the admissible values inside each bounded operation
    :param npoints: Number of points used for support
    :param Xbounds: Minimimum and maximum support values
    :param variable: Name of the Sympy variable contained in the expression. Default: 'x_1'
    :param extrapolate: If True, sample support values that go beond the training domain
    :param avoid_x:List of points that the support should avoid
    :param threshold: Minimum distance w.r.t. a point that causes undefined values
    """
    args = expr.args
    new_args = []

    for arg in args:
        # Check if there's any unary and bounded function inside this argument
        contains_unary_bounded = False
        if any([f in str(arg) for f in list(bounded_ops[0].keys())]) or any([f in str(arg) for f in bounded_ops[-1]]) or \
                any([f in str(arg) for f in list(bounded_ops[1].keys())]) or '/' in str(arg) or \
                any([f in str(arg) for f in list(bounded_ops[2].keys())]) or '**0.5' in str(arg) or \
                '**(-' in str(arg):
            contains_unary_bounded = True

        if not contains_unary_bounded:
            new_args.append(arg)
        else:  # If there's a unary and bounded function inside this argument, explore a lower level of the tree
            # Check if the current function is sqrt = Pow and the second argument is 1/2
            is_sqrt, is_div = False, False
            if 'Pow' in str(arg.func):
                if np.abs(arg.args[1]) == 0.5:
                    is_sqrt = True
                if arg.args[1] < 0:  # If exponent is negative, is a division
                    is_div = True

            # Check if the function at the current level of the tree is unary and bounded
            if any([f in str(arg.func) for f in list(bounded_ops[0].keys())]) or \
                    any([f in str(arg.func) for f in list(bounded_ops[1].keys())]) or \
                    any([f in str(arg.func) for f in list(bounded_ops[2].keys())]) or \
                    any([f in str(arg.func) for f in bounded_ops[3]]) or is_sqrt or is_div:
                # Now check if there's any other unary and bounded function inside the current function
                # Note that inside this IF statement, it is true that arg is a unary and bounded function, so it has
                # only one argument
                arg_init = copy.deepcopy(arg)
                if any([f in str(arg.args[0]) for f in list(bounded_ops[0].keys())]) or \
                        any([f in str(arg.args[0]) for f in list(bounded_ops[1].keys())]) or \
                        any([f in str(arg.args[0]) for f in list(bounded_ops[2].keys())]) or \
                        any([f in str(arg.args[0]) for f in bounded_ops[3]]) or \
                        '**0.5' in str(arg.args[0]) or '**(-' in str(arg.args[0]) or '/' in str(arg.args[0]):
                    # If that's the case, go to a deeper level of the tree to obtain new function arguments
                    x, arg, avoid_x = modify_constants_avoidNaNs(arg, x, bounded_ops, npoints, Xbounds, variable,
                                                                 extrapolate=extrapolate, avoid_x=avoid_x,
                                                                 threshold=threshold)

                # Check if the operation function has changed. If yes, the new argument must be a number
                # For example, suppose the original expr is (4x + 7)**.5 and it changes to (4x + 16)**.5, then Sympy will
                # factorize it as 2(x + 4)**.5
                constant = 1
                change_detected = False
                if arg_init.func != arg.func:
                    # if arg.args[0].is_number:
                    change_detected = True
                    constant = arg.args[0]
                    args2 = arg.args[1].args[0]
                else:
                    args2 = arg.args[0]

                # Evaluate the support vector X to find NaNs
                function = lambdify(flatten(variable), arg)
                vals = np.array(function(*list(x)))

                if any(is_nan(vals)) or ('exp' in str(arg_init.func) and 0 in vals) or \
                        any([f in str(arg_init.func) for f in list(bounded_ops[2].keys())]) or \
                        any([f in str(arg_init.func) for f in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh']]):
                    # If NaN values were found, evaluate the values of the arguments inside the current function
                    arg_function = lambdify(flatten(variable), args2)
                    vals_arg = np.array(arg_function(*list(x)))

                    # If it's a single-bounded unary operation, add an additive constant to avoid NaNs
                    if any([f in str(arg_init.func) for f in list(bounded_ops[0].keys())]) or is_sqrt:
                        if is_sqrt:
                            bound_type, bound = bounded_ops[0].get('sqrt')
                        else:
                            find = \
                                np.where(np.array([f in str(arg_init.func) for f in list(bounded_ops[0].keys())]))[0][0]
                            current_function = list(bounded_ops[0].keys())[find]
                            bound_type, bound = bounded_ops[0].get(current_function)
                        if bound_type == 'min':
                            # If there are more NaNs than actual values, try multiplying vals_arg by -1
                            NaNs = np.sum(is_nan(vals))
                            if NaNs > (len(vals) - NaNs):
                                args2 = - args2
                                vals_arg = - vals_arg
                            # Check if after the previous modification, there's still values in vals_arg that are less than the minimum bound
                            if any(np.array(vals_arg) < bound):
                                # If that's the case, add a horizontal offset
                                minVal = np.min(vals_arg)
                                offset = bound - minVal
                                args2 = args2 + offset + 0.1
                                if offset > 10:
                                    args2 = args2 / offset * 10
                        else:
                            # Max-bounded operations are simply divided and multiplied
                            if np.any(np.abs(vals_arg) > bound):
                                args2 = args2 / np.max(np.abs(vals_arg)) * bound

                    elif any([f in str(arg_init.func) for f in list(bounded_ops[1].keys())]):
                        # Double-bounded ops: add multiplicative and additive constants to bound the values of vals_arg
                        find = np.where(np.array([f in str(arg_init.func) for f in list(bounded_ops[1].keys())]))[0][0]
                        current_function = list(bounded_ops[1].keys())[find]
                        [min_bound, max_bound] = bounded_ops[1].get(current_function)
                        # Min-max normalization
                        # with sympy.evaluate(False):  # This prevents factorization, which would create more constants
                        args2 = (args2 - np.min(vals_arg)) / (
                                np.max(vals_arg) - np.min(vals_arg))  # Scaled between 0 and 1
                        args2 = args2 * (max_bound - min_bound) + min_bound  # Scaled between min_bound and max_bound

                    elif str(arg_init.func) not in ['sin', 'cos']:
                        # For operations with singularities such as TAN, resample the support vector to avoid singularities
                        # Drop input-output pairs containing NaNs and entries with an absolute value of y above 10000
                        if extrapolate:
                            minb, maxb = Xbounds[0] * 2, Xbounds[1] * 2
                        else:
                            minb, maxb = Xbounds[0], Xbounds[1]

                        new_support, avoid_x = handle_singularities(expr=arg, variable=variable, n_points=npoints,
                                                                    minb=minb, maxb=maxb, pre_solutions=avoid_x,
                                                                    threshold=threshold)
                        x = new_support
                        x = x[None, :]

                    if any([f in str(arg_init.func) for f in ['sin', 'cos', 'tan']]):
                        # Restrict the range of values that can be encountered inside trig functions
                        arg_range = np.max(vals_arg) - np.min(vals_arg)
                        if arg_range > 50:
                            args2 = args2 / arg_range * 30

                # Update function
                if is_sqrt or is_div:
                    arg = arg_init.func(*[args2, arg_init.args[-1]])
                else:
                    arg = arg_init.func(*[args2])

                if change_detected:
                    arg = constant * arg
            else:
                # Go to a deeper level of the tree to obtain new function arguments
                result = modify_constants_avoidNaNs(arg, x, bounded_ops, npoints, Xbounds, variable,
                                                    extrapolate=extrapolate, avoid_x=avoid_x, threshold=threshold)
                if result is None:
                    return None
                x, arg, avoid_x = result

            new_args.append(arg)

    new_xp = expr.func(*new_args)
    return x, new_xp, avoid_x


def handle_singularities(expr, variable, n_points, minb, maxb, pre_solutions=None, threshold=0.05):
    """Sample a support that avoid generating undefined or too extreme values for the case o division or tangent"""
    arg = 1 / expr
    if isinstance(arg, sympy.Mul):
        if arg.args[0].is_number:
            arg = arg.args[1]
        else:
            arg = arg.args[0]
    expr_fun = lambdify(flatten(variable), arg)

    x_range = np.linspace(minb - 0.1, maxb + 0.1, 1500)  # Declare initial potential solutions
    y_range = expr_fun(x_range)
    solutions = []
    for i in range(1, len(x_range) - 1):
        if np.sign(expr_fun(x_range[i])) != np.sign(
                y_range[i - 1]) or np.abs(expr_fun(x_range[i])) < 0.001:  # If the sign change, it means it's crossed the x axis
            solutions.append(x_range[i])
    # Find the values of x where the equation equals 0
    valid_solutions = [np.round(x, 3) for x in solutions if minb <= x <= maxb]
    if pre_solutions is not None:
        valid_solutions = valid_solutions + list(pre_solutions)
    unique_solutions = np.sort(np.unique(valid_solutions))

    # Define a function to check if a point is too close to any number that causes singularities
    def is_close_to_sg(number, resp, fb_list, thr):
        if 1 / np.abs(resp) >= 10:
            return True
        for fb_number in fb_list:
            if abs(number - fb_number) < thr:
                return True
        return False

    # Identify valid ranges
    x_init = np.linspace(minb + 0.1, maxb - 0.1, 2000)
    y_init = expr_fun(x_init)
    y_init = [is_close_to_sg(x, y, unique_solutions, thr=threshold) for x, y in zip(x_init, y_init)]
    list_pairs = []
    temp_pair = []
    check_left = True
    for count, (x, y) in enumerate(zip(x_init, y_init)):
        if check_left:
            if y is False:
                temp_pair.append(x)
                check_left = False
        else:
            if y:
                temp_pair.append(x_init[count - 1])
                check_left = True
                list_pairs.append(tuple(temp_pair))
                temp_pair = []
    if len(temp_pair) == 1:
        list_pairs.append((temp_pair[0], maxb - 0.1))

    # Calculate the total length of all ranges
    total_length = sum(end - start for start, end in list_pairs)
    # Initialize an empty list to store the generated points
    generated_points = []
    # Generate points proportional to the length of each range
    for start, end in list_pairs:
        # Calculate the number of points to be generated in the current range
        points_in_range = int(n_points * (end - start) / total_length)
        # Generate equidistant points within the current range
        step = (end - start) / (points_in_range - 1)
        points_in_current_range = [start + j * step for j in range(points_in_range)]
        # Extend the generated points to the list
        generated_points.extend(points_in_current_range)

    # If the generated points are less than required, sample from one of the calculated ranges
    while len(generated_points) < n_points:
        selected_range = np.random.choice(np.arange(0, len(list_pairs)))
        random_point = np.random.uniform(list_pairs[selected_range][0], list_pairs[selected_range][1])
        if not is_close_to_sg(random_point, expr_fun(random_point), unique_solutions, thr=threshold):
            generated_points.append(random_point)

    return np.array(generated_points), unique_solutions


def remove_outliers(support, vals):
    """Detect and remove outliers"""
    # Sort based on the values of the support
    support = support[0, :]
    ind = np.argsort(support)
    support = support[ind]
    vals = vals[ind]
    # Calculate mean and std distance between each point and its closest neighbors
    dist_neigh = []
    for i in range(1, len(support) - 1):
        # Consider only the maximum distance
        dist_neigh.append(np.minimum(np.abs(vals[i] - vals[i - 1]), np.abs(vals[i] - vals[i + 1])))
    dist_neigh = np.array(dist_neigh)
    mean_dist, std_dist = np.mean(dist_neigh), np.std(dist_neigh)
    # Find outliers and remove them
    indices_outliers = np.where(np.abs(dist_neigh - mean_dist) > 3 * std_dist)[0]
    support = np.delete(np.array(support), indices_outliers)
    vals = np.delete(np.array(vals), indices_outliers)
    return support[None, :], vals
