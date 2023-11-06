# Code adapted from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales
# The "add_additive_constants" was removed, bugs were fixed, and the method was improved

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def simplify(ff, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0

    def _simplify(fff):
        try:
            f2 = sp.simplify(fff)
            return f2
        except TimeoutError:
            return fff

    return _simplify(ff)


def count_occurrences(expr):
    """
    Count atom occurrences in an expression.
    """
    if expr.is_Atom:
        return {expr: 1}
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:
        assert len(expr.args) >= 2
        result = {}
        for arg in expr.args:
            sub_count = count_occurrences(arg)
            for k, v in sub_count.items():
                result[k] = result.get(k, 0) + v
        return result
    else:
        assert len(expr.args) == 1, expr
        return count_occurrences(expr.args[0])


def count_occurrences2(expr):
    """
    Count atom occurrences in an expression.
    """
    result = {}
    for sub_expr in sp.preorder_traversal(expr):
        if sub_expr.is_Atom:
            result[sub_expr] = result.get(sub_expr, 0) + 1
    return result


def remove_root_constant_terms_t(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    variables = [str(xx) for xx in variables]
    assert mode in ["add", "mul", "pow"]
    if not any(str(xx) in variables for xx in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add:
        args = [
            arg
            for arg in expr.args
            if any(str(x) in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "mul" and expr.is_Mul:
        args = [
            arg
            for arg in expr.args
            if any(x in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "pow" and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr

    return expr


def remove_root_constant_terms(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    assert mode in ["add", "mul", "pow"]
    if not any(x in variables for x in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add or mode == "mul" and expr.is_Mul:
        args = [
            arg
            for arg in expr.args
            if any(x in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "pow" and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr
    return expr


def remove_mul_const(ff, variables):
    """
    Remove the multiplicative factor of an expression, and return it.
    """
    if not ff.is_Mul:
        return ff, 1
    variables = variables if type(variables) is list else [variables]
    var_args = []
    cst_args = []
    for arg in ff.args:
        if any(var in arg.free_symbols for var in variables):
            var_args.append(arg)
        else:
            cst_args.append(arg)
    return sp.Mul(*var_args), sp.Mul(*cst_args)


def extract_non_constant_subtree(expr, variables):
    """
    Extract a non-constant sub-tree from an equation.
    """
    while True:
        last = expr
        expr = remove_root_constant_terms(expr, variables, "mul")
        n_arg = len(expr.args)
        for i in range(n_arg):
            expr = expr.subs(
                expr.args[i], extract_non_constant_subtree(expr.args[i], variables)
            )

        if str(expr) == str(last):
            return expr


def check_additive_constants(expr, variables):
    """
    Extract a non-constant sub-tree from an equation.
    """
    while True:
        last = expr
        expr = remove_root_constant_terms_t(expr, variables, "add")
        if expr != last:
            return True
        n_arg = len(expr.args)
        for i in range(n_arg):
            if check_additive_constants(expr.args[i], variables):
                return True
            else:
                continue

        if str(expr) == str(last):
            return False


def add_multiplicative_constants(expr, multiplicative_placeholder, unary_operators=None):
    """
    Traverse the tree in post-order fashion and add multiplicative placeholders
    """

    if unary_operators is None:
        unary_operators = []

    if not expr.args:
        if type(expr) == sp.core.numbers.NegativeOne:
            return expr
        elif isinstance(expr, sp.Float) or isinstance(expr, sp.Integer):
            return 1
        else:
            return multiplicative_placeholder * expr
    for sub_expr in expr.args:
        expr = expr.subs(sub_expr, add_multiplicative_constants(sub_expr, multiplicative_placeholder,
                                                                unary_operators=unary_operators))

    if str(type(expr)) in unary_operators:
        expr = multiplicative_placeholder * expr

    # If a constant appears after Sympy simplifies the expression, get rid of it
    
    return expr


def remove_numeric_constants(expr):
    if not expr.args:
        if expr.is_number:
            return 1

    for iarg, sub_expr in enumerate(expr.args):
        if iarg == len(expr.args) - 1:
            if isinstance(expr, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
                continue
        expr = expr.subs(sub_expr, remove_numeric_constants(sub_expr))

    return expr


def add_constants(expr, placeholders, prev_expr=None):
    if not expr.args:
        if type(expr) == sp.core.numbers.NegativeOne or str(expr) == str(placeholders["cm"]):
            return expr
        elif expr.is_number:
            # Add dummy constant that will be removed once each constant is assigned a different identifier
            return sp.sympify('cc')
        else:
            if 'exp' in prev_expr:  # If it's an exponential function, only add a multiplicative constant
                return expr * placeholders["cm"]
            else:
                return placeholders["ca"] + expr * placeholders["cm"]

    new_args = []
    for iarg, sub_expr in enumerate(expr.args):
        if iarg == len(expr.args) - 1:
            if isinstance(expr, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
                continue
        new_args.append(add_constants(sub_expr, placeholders, str(expr.func)))

    if isinstance(expr, sp.Pow):
        new_args.append(expr.args[1])

    new_xp = expr.func(*new_args)
    # if len(new_xp.args) == 1:
    #     if str(new_xp.args[0]) != 'ca + cm*x_1':
    #         new_arg = placeholders["ca"] + new_xp.args[0] * placeholders["cm"]
    #         new_xp = new_xp.func(new_arg)
    return new_xp


def numeric_to_placeholder(expr, var=None):
    """Given an expression with numeric constants, replace all constants with placeholder 'c'
   If var is not None, consider any variable different from 'var' a numeric constant"""
    if not expr.args:
        if expr.is_number or (var is not None and expr.is_Symbol and str(expr) != var):
            return sp.sympify('c')
        else:
            return expr

    new_args = []
    for iarg, sub_expr in enumerate(expr.args):
        if iarg == len(expr.args) - 1:
            if isinstance(expr, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
                continue
        new_args.append(numeric_to_placeholder(sub_expr, var=var))

    if isinstance(expr, sp.Pow):
        new_args.append(expr.args[1])

    new_xp = expr.func(*new_args)
    return new_xp


def remove_dummy_constants(expr, prev_expr=None):

    if not expr.args:
        if str(expr) == 'cc':
            if 'Add' in prev_expr:
                return 0
            else:
                return 1
        else:
            return expr
    else:
        if 'cc' in str(expr.args) and 'x' not in str(expr.args):
            if 'Add' in prev_expr:
                return 0
            else:
                return 1

    new_args = []
    for iarg, sub_expr in enumerate(expr.args):
        if iarg == len(expr.args) - 1:
            if isinstance(expr, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
                continue
        replaced_value = remove_dummy_constants(sub_expr, str(expr.func))
        if not (replaced_value == 0 or replaced_value == 1):
            new_args.append(remove_dummy_constants(sub_expr, str(expr.func)))

    if isinstance(expr, sp.Pow):
        new_args.append(expr.args[1])

    new_xp = expr.func(*new_args)
    if 're' in str(new_xp):
        n_exp = str(new_xp)
        n_exp = n_exp.replace('re', '')
        new_xp = sp.sympify(n_exp)
    return new_xp


def reindex_coefficients(expr, coefficients):
    """
    Re-index coefficients (i.e. if a1 is there and not a0, replace a1 by a0, and recursively).
    """
    coeffs = sorted(
        [x for x in expr.free_symbols if x in coefficients], key=lambda x: x.name
    )
    for idx, coeff in enumerate(coefficients):
        if idx >= len(coeffs):
            break
        if coeff != coeffs[idx]:
            expr = expr.subs(coeffs[idx], coeff)
    return expr


def reduce_coefficients(expr, variables, coefficients):
    """
    Reduce coefficients in an expression.
    `sqrt(x)*y*sqrt(1/a0)` -> `a0*sqrt(x)*y`
    `x**(-cos(a0))*y**cos(a0)` -> `x**(-a0)*y**a0`
    """
    temp = sp.Symbol("temp")
    while True:
        last = expr
        for a in coefficients:
            if a not in expr.free_symbols:
                continue
            for subexp in sp.preorder_traversal(expr):
                if a in subexp.free_symbols and not any(
                        var in subexp.free_symbols for var in variables
                ):
                    p = expr.subs(subexp, temp)
                    if a in p.free_symbols:
                        continue
                    else:
                        expr = p.subs(temp, a)
                        break
        if last == expr:
            break
    return expr


def simplify_const_with_coeff(expr, coeff):
    """
    Simplify expressions with constants and coefficients.
    `sqrt(10) * a0 * x` -> `a0 * x`
    `sin(a0 + x + 9/7)` -> `sin(a0 + x)`
    `a0 + x + 9` -> `a0 + x`
    """
    assert coeff.is_Atom
    for parent in sp.preorder_traversal(expr):
        if any(coeff == arg for arg in parent.args):
            break
    if not (parent.is_Add or parent.is_Mul):
        return expr
    removed = [arg for arg in parent.args if len(arg.free_symbols) == 0]
    if len(removed) > 0:
        removed = parent.func(*removed)
        new_coeff = (coeff - removed) if parent.is_Add else (coeff / removed)
        expr = expr.subs(coeff, new_coeff)
    return expr


def simplify_equa_diff(_eq, required=None):
    """
    Simplify a differential equation by removing non-zero factors.
    """
    eq = sp.factor(_eq)
    if not eq.is_Mul:
        return _eq
    args = []
    for arg in eq.args:
        if arg.is_nonzero:
            continue
        if required is None or arg.has(required):
            args.append(arg)
    assert len(args) >= 1
    return args[0] if len(args) == 1 else eq.func(*args)


def smallest_with_symbols(expr, symbols):
    """
    Return the smallest sub-tree in an expression that contains all given symbols.
    """
    assert all(x in expr.free_symbols for x in symbols)
    if len(expr.args) == 1:
        return smallest_with_symbols(expr.args[0], symbols)
    candidates = [
        arg for arg in expr.args if any(x in arg.free_symbols for x in symbols)
    ]
    return (
        smallest_with_symbols(candidates[0], symbols) if len(candidates) == 1 else expr
    )


def smallest_with(expr, symbol):
    """
    Return the smallest sub-tree in an expression that contains a given symbol.
    """
    assert symbol in expr.free_symbols
    candidates = [arg for arg in expr.args if symbol in arg.free_symbols]
    if len(candidates) > 1 or candidates[0] == symbol:
        return expr
    else:
        return smallest_with(candidates[0], symbol)


def clean_degree2_solution(expr, x, a88, a99):
    """
    Clean solutions of second order differential equations.
    """
    last = expr
    while True:
        for a in [a88, a99]:
            if a not in expr.free_symbols:
                return expr
            small = smallest_with(expr, a)
            if small.is_Add or small.is_Mul:
                counts = count_occurrences2(small)
                if counts[a] == 1 and a in small.args:
                    if x in small.free_symbols:
                        expr = expr.subs(
                            small,
                            small.func(
                                *[
                                    arg
                                    for arg in small.args
                                    if arg == a or x in arg.free_symbols
                                ]
                            ),
                        )
                    else:
                        expr = expr.subs(small, a)
        if expr == last:
            break
        last = expr
    return expr


def has_inf_nan(*args):
    """
    Detect whether some expressions contain a NaN / Infinity symbol.
    """
    for ff in args:
        if ff.has(sp.nan) or ff.has(sp.oo) or ff.has(-sp.oo) or ff.has(sp.zoo):
            return True
    return False


def has_I(*args):
    """
    Detect whether some expressions contain complex numbers.
    """
    for ff in args:
        if ff.has(sp.I):
            return True
    return False
