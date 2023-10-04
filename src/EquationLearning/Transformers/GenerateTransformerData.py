import os
import copy
import time
# import math
import types
import sympy
# import torch
import warnings
import numpy as np
# from typing import List
from pathlib import Path
# from torch.utils import data
# import matplotlib.pyplot as plt
from sympy import sympify, lambdify
from wrapt_timeout_decorator import *
from sympy.utilities.iterables import flatten
# from torch.distributions.uniform import Uniform
from src.EquationLearning.Data.dclasses import Equation
from src.EquationLearning.Data.generator import Generator
from src.utils import load_metadata_hdf5, load_eq, get_project_root
from src.EquationLearning.Data.sympy_utils import numeric_to_placeholder
from src.EquationLearning.Data.data_utils import sample_symbolic_constants, bounded_operations
from src.EquationLearning.models.utilities_expressions import avoid_operations_between_constants, get_op_constant
from src.EquationLearning.models.utilities_expressions import get_args, set_args

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Dataset:
    def __init__(
            self,
            data_path: Path,
            cfg,
            mode: str
    ):
        metadata = load_metadata_hdf5(Path(os.path.join(get_project_root(), "src\\EquationLearning\\Data", data_path)))
        cfg.total_variables = metadata.total_variables
        cfg.total_coefficients = metadata.total_coefficients
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        self.id2word = metadata.id2word
        self.data_path = Path(os.path.join(get_project_root(), "src\\EquationLearning\\Data", data_path))
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
        if 'x' in i:
            i = 'x_1'
        try:
            tokenized_expr.append(word2id[i])
        except:
            print('b')
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


# def tokens_padding(tokens):
#     max_len = max([len(y) for y in tokens])
#     p_tokens = torch.zeros(len(tokens), max_len)
#     for i, y in enumerate(tokens):
#         y = torch.tensor(y).long()
#         p_tokens[i, :] = torch.cat([y, torch.zeros(max_len - y.shape[0]).long()])
#     return p_tokens


# def number_of_support_points(p, type_of_sampling_points):
#     if type_of_sampling_points == "constant":
#         curr_p = p
#     elif type_of_sampling_points == "logarithm":
#         curr_p = int(10 ** Uniform(1, math.log10(p)).sample())
#     else:
#         raise NameError
#     return curr_p


def sample_support(eq, curr_p, cfg, extrapolate=False):
    sym = []
    # if not eq.support:
    #     distribution = torch.distributions.Uniform(cfg.fun_support.min, cfg.fun_support.max)
    # else:
    #     raise NotImplementedError

    for sy in cfg.total_variables:
        if 'x' in sy:
            # curr = distribution.sample([int(curr_p)])
            if not extrapolate:
                curr = np.random.uniform(cfg.fun_support.min, cfg.fun_support.max, size=int(curr_p))
            else:
                curr = np.random.uniform(cfg.fun_support.min * 2, cfg.fun_support.max * 2, size=int(curr_p))
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
                if not isinstance(op, bool):
                    if ('exp' in op) or ('sinh' in op) or ('cosh' in op) or ('tanh' in op):
                        exp = exp.subs(c, np.random.uniform(low=-3,
                                                            high=3))
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
                if not isinstance(op, bool):
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


@timeout(18)  # Set a 40-second timeout
def evaluate_and_wrap(eq, cfg, word2id, return_exprs=False, extrapolate=False):
    """
    Given a list of skeleton equations, sample their inputs and corresponding constants and evaluate their results
    :param eq: List of skeleton equations
    :param cfg: configuration file
    :param word2id: Dictionary used to tokenize equations
    :param return_exprs: If True, return the expression that were sampeld during the process
    :param extrapolate: If True, sample support values that go beond the training domain
    """
    exprs = eq.expr
    curr_p = cfg.max_number_of_points

    # Check if there's any bounded operation in the expression
    bounded, double_bounded, op_with_singularities = bounded_operations()
    is_bounded = False
    if any([b in str(eq) for b in list(bounded.keys())]) or \
            any([b in str(eq) for b in list(double_bounded.keys())]) or \
            any([b in str(eq) for b in list(op_with_singularities)]) or '**0.5' in str(eq):
        is_bounded = True

    # Create "cfg.number_of_sets" vectors to store the evaluations
    X = np.zeros((curr_p, cfg.number_of_sets), dtype=np.float32)
    Y = np.zeros((curr_p, cfg.number_of_sets), dtype=np.float32)
    # Sample constants "cfg.number_of_sets" times
    support = sample_support(eq, curr_p, cfg, extrapolate)
    tokenized = None
    n_set = 0
    sampled_exprs = [''] * cfg.number_of_sets
    tic_main = time.time()
    while n_set < cfg.number_of_sets:  # for loop but using while because it may restart
        # If more than 20 seconds time has passed, skip this equation
        toc_main = time.time()
        if toc_main - tic_main > 15:
            return None

        restart = False
        consts = sample_constants(eq, cfg)
        # Use consts.expr as the expression that will be used for evaluation
        new_expr = consts.expr
        # If there's a coefficient that is too big, clip it to 10
        if any(np.abs(get_args(new_expr)) > 10):
            args = get_args(new_expr)
            args = [arg if np.abs(arg) <= 10 else 10 * np.sign(arg) for arg in args]
            new_expr = set_args(new_expr, args)
        # If the expression contains a bounded operation, modify the constants to avoid NaN values
        if is_bounded:
            support, new_expr = modify_constants_avoidNaNs(new_expr, support, bounded_operations(),
                                                           cfg, variable=eq.variables, extrapolate=extrapolate)
            # If the new expression needs coefficients that are too large, try again
            if any(np.abs(get_args(new_expr)) > 100):
                restart = True

        # Convert expression to prefix format and tokenize
        expr_with_placeholder = numeric_to_placeholder(new_expr)
        eq_sympy_infix = constants_to_placeholder(expr_with_placeholder, eq.coeff_dict)
        eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        t = tokenize(eq_sympy_prefix, word2id)

        # Make sure that all sets have the same tokenized skeleton
        if n_set == 0:
            tokenized = t
            exprs = expr_with_placeholder
        else:
            tic = time.time()  # Start Time
            big_coef = True
            while not (t == tokenized and not big_coef):
                consts = sample_constants(eq, cfg)
                # Use consts.expr as the expression that will be used for evaluation
                new_expr = consts.expr
                # If there's a coefficient that is too big, clip it to 10
                if any(np.abs(get_args(new_expr)) > 10):
                    args = get_args(new_expr)
                    args = [arg if np.abs(arg) <= 10 else 10 * np.sign(arg) for arg in args]
                    new_expr = set_args(new_expr, args)
                # If the expression contains a bounded operation, modify the constants to avoid NaN values
                if is_bounded:
                    support, new_expr = modify_constants_avoidNaNs(new_expr, support, bounded_operations(), cfg,
                                                                   variable=eq.variables, extrapolate=extrapolate)
                # If the new expression needs coefficients that are too large, try again
                if any(np.abs(get_args(new_expr)) > 100):
                    big_coef = True
                else:
                    big_coef = False
                # Convert expression to prefix format and tokenize
                expr_with_placeholder = numeric_to_placeholder(new_expr)
                eq_sympy_infix = constants_to_placeholder(expr_with_placeholder, eq.coeff_dict)
                eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
                t = tokenize(eq_sympy_prefix, word2id)
                # Calculate how much time has passed
                toc = time.time()
                if toc - tic > 10:
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
            # Drop input-output pairs containing NaNs and entries with an absolute value of y above 10000
            isnans, indices_to_remove = is_nan(vals), []
            if any(isnans):
                indices_to_remove = np.where(isnans)[0]
            new_support = np.delete(np.array(support), indices_to_remove)
            vals = np.delete(np.array(vals), indices_to_remove)

            # If some elements have been dropped, keep sampling until reaching a population of 'curr_p' size
            tic = time.time()  # Start Time
            while len(vals) < curr_p:
                missing = curr_p - len(vals)
                # Sample constants "cfg.number_of_sets/2" times and remove NaNs from them
                extra_support = sample_support(eq, int(curr_p/2), cfg, extrapolate)
                extra_vals = np.array(function(*list(extra_support)))
                isnans, indices_to_remove = is_nan(extra_vals), []
                if any(isnans):
                    indices_to_remove = np.where(isnans)[0]
                extra_support = np.delete(np.array(extra_support), indices_to_remove)
                extra_vals = np.delete(np.array(extra_vals), indices_to_remove)
                if len(extra_support) >= missing:
                    new_support = np.append(new_support, extra_support[:missing])
                    vals = np.append(vals, extra_vals[:missing])
                else:
                    new_support = np.append(new_support, extra_support)
                    vals = np.append(vals, extra_vals)
                # If more than 20 seconds time has passed, skip this equation
                toc = time.time()
                if toc - tic > 10:
                    return None

            X[:, n_set] = new_support
            Y[:, n_set] = vals
            sampled_exprs[n_set] = new_expr
            n_set += 1

        # plt.figure()
        # plt.scatter(np.array(new_support), vals)
    # print(exprs)
    if return_exprs:
        return X, Y, tokenized, exprs, sampled_exprs
    else:
        return X, Y, tokenized, exprs


def is_nan(x, bound=None):
    """Encompasses NaN, Inf values, and outliers"""
    mean = np.mean(x)
    std = np.std(x)
    threshold = 5
    outliers = []
    try:
        for xx in x:
            z_score = (xx - mean) / std
            if abs(z_score) > threshold:
                outliers.append(True)
            else:
                outliers.append(False)
    except:
        print('a')

    if bound is None:
        return np.iscomplex(x) + np.isnan(x) + np.isinf(x) + (np.abs(x) > 20000) + np.array(outliers)
    else:
        return np.iscomplex(x) + np.isnan(x) + np.isinf(x) + (np.abs(x) > 20000) + (np.abs(x) > bound) + np.array(outliers)


@timeout(8)
def modify_constants_avoidNaNs(expr, x, bounded_ops, cfg, variable=sympy.sympify('x_1'), extrapolate=False):
    """Modifies the constants inside unary bounded operations to avoid generating NaN values
    :param expr: Original expression
    :param x: Support vector of variable x values
    :param bounded_ops: Dictionary including the admissible values inside each bounded operation
    :param cfg: configuration yaml
    :param variable: Name of the Sympy variable contained in the expression. Default: 'x_1'
    :param extrapolate: If True, sample support values that go beond the training domain
    """

    args = expr.args
    new_args = []

    for arg in args:
        # Check if there's any unary and bounded function inside this argument
        contains_unary_bounded = False
        if any([f in str(arg) for f in list(bounded_ops[0].keys())]) or \
                any([f in str(arg) for f in list(bounded_ops[1].keys())]) or \
                any([f in str(arg) for f in list(bounded_ops[2].keys())]) or '**0.5' in str(arg) or '**(-0.5)' in str(arg):
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
                    any([f in str(arg.func) for f in list(bounded_ops[2].keys())]) or is_sqrt or is_div:
                # Now check if there's any other unary and bounded function inside the current function
                # Note that inside this IF statement, it is true that arg is a unary and bounded function, so it has
                # only one argument
                arg_init = copy.deepcopy(arg)
                if any([f in str(arg.args[0]) for f in list(bounded_ops[0].keys())]) or \
                        any([f in str(arg.args[0]) for f in list(bounded_ops[1].keys())]) or\
                        any([f in str(arg.args[0]) for f in list(bounded_ops[2].keys())]) or \
                        '**0.5' in str(arg.args[0]) or '**(-' in str(arg.args[0]) or '/' in str(arg.args[0]):
                    # If that's the case, go to a deeper level of the tree to obtain new function arguments
                    x, arg = modify_constants_avoidNaNs(arg, x, bounded_ops, cfg, variable, extrapolate=extrapolate)

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
                        any([f in str(arg_init.func) for f in list(bounded_ops[2].keys())]):
                    # If NaN values were found, evaluate the values of the arguments inside the current function
                    arg_function = lambdify(flatten(variable), args2)
                    vals_arg = np.array(arg_function(*list(x)))

                    # If it's a single-bounded unary operation, add an additive constant to avoid NaNs
                    if any([f in str(arg_init.func) for f in list(bounded_ops[0].keys())]) or is_sqrt:
                        if is_sqrt:
                            bound_type, bound = bounded_ops[0].get('sqrt')
                        else:
                            find = np.where(np.array([f in str(arg_init.func) for f in list(bounded_ops[0].keys())]))[0][0]
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
                                args2 = args2 + (bound - minVal)
                                if offset > 10:
                                    args2 = args2 / offset * 10
                        else:
                            # Max-bounded operations are simply divided and multiplied
                            args2 = args2 / np.max(np.abs(vals_arg)) * bound

                    elif any([f in str(arg_init.func) for f in list(bounded_ops[1].keys())]):
                        # Double-bounded ops: add multiplicative and additive constants to bound the values of vals_arg
                        find = np.where(np.array([f in str(arg_init.func) for f in list(bounded_ops[1].keys())]))[0][0]
                        current_function = list(bounded_ops[1].keys())[find]
                        [min_bound, max_bound] = bounded_ops[1].get(current_function)
                        # Min-max normalization
                        # with sympy.evaluate(False):  # This prevents factorization, which would create more constants
                        args2 = (args2 - np.min(vals_arg)) / (np.max(vals_arg) - np.min(vals_arg))  # Scaled between 0 and 1
                        args2 = args2 * (max_bound - min_bound) + min_bound  # Scaled between min_bound and max_bound

                    else:
                        # For operations with singularities such as TAN, resample the support vector to avoid singularities
                        # Drop input-output pairs containing NaNs and entries with an absolute value of y above 10000
                        if is_div:
                            bound = bounded_ops[2].get('div')
                        else:
                            find = np.where(np.array([f in str(arg_init.func) for f in list(bounded_ops[2].keys())]))[0][0]
                            current_function = list(bounded_ops[2].keys())[find]
                            bound = bounded_ops[2].get(current_function)
                        isnans, indices_to_remove = is_nan(vals, bound), []
                        if any(isnans):
                            indices_to_remove = np.where(isnans)[0]
                        new_support = np.delete(np.array(x), indices_to_remove)
                        vals = np.delete(np.array(vals), indices_to_remove)
                        curr_p = cfg.max_number_of_points
                        # distribution = torch.distributions.Uniform(cfg.fun_support.min, cfg.fun_support.max)
                        while len(vals) < curr_p:
                            missing = curr_p - len(vals)
                            # Sample constants "cfg.number_of_sets/2" times and remove NaNs from them
                            if extrapolate:
                                extra_support = np.random.uniform(cfg.fun_support.min * 2, cfg.fun_support.max * 2,
                                                                  size=int(curr_p / 2))
                            else:
                                extra_support = np.random.uniform(cfg.fun_support.min, cfg.fun_support.max, size=int(curr_p / 2))
                            extra_support = extra_support[None, :]  # Add extra dimension
                            extra_vals = np.array(function(*list(extra_support)))
                            isnans, indices_to_remove = is_nan(extra_vals, bound), []
                            if any(isnans):
                                indices_to_remove = np.where(isnans)[0]
                            extra_support = np.delete(np.array(extra_support), indices_to_remove)
                            extra_vals = np.delete(np.array(extra_vals), indices_to_remove)
                            if len(extra_support) >= missing:
                                new_support = np.append(new_support, extra_support[:missing])
                                vals = np.append(vals, extra_vals[:missing])
                            else:
                                new_support = np.append(new_support, extra_support)
                                vals = np.append(vals, extra_vals)
                        x = new_support
                        x = x[None, :]

                # Update function
                if is_sqrt or is_div:
                    arg = arg_init.func(*[args2, arg_init.args[1]])
                else:
                    arg = arg_init.func(*[args2])

                if change_detected:
                    arg = constant * arg
            else:
                # Go to a deeper level of the tree to obtain new function arguments
                x, arg = modify_constants_avoidNaNs(arg, x, bounded_ops, cfg, variable, extrapolate=extrapolate)

            new_args.append(arg)

    new_xp = expr.func(*new_args)
    return x, new_xp
