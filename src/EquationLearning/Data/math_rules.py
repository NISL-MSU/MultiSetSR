import sympy as sp
from EquationLearning.models.utilities_expressions import multi_div


def sk_equivalence(expr):
    """If possible, transform the skeleton into a mathematically equivalent skeleton"""
    xvar = str([v for v in expr.free_symbols if 'x' in str(v)][0])
    expr = str(expr)

    ################################################
    # Account for different types of equivalency
    ################################################
    # 1. Transform cos expressions into sin
    sin_expr = 'sin(c*x + c)'.replace('x', xvar)
    cos_expr1, cos_expr2 = 'cos(c*x + c)'.replace('x', xvar), 'cos(c*x)'.replace('x', xvar)
    cos_expr3, cos_expr4 = 'cos(c + c*x)'.replace('x', xvar), 'cos(c + x)'.replace('x', xvar)
    if cos_expr1 in expr:
        expr = expr.replace(cos_expr1, sin_expr)
    if cos_expr3 in expr:
        expr = expr.replace(cos_expr3, sin_expr)
    if cos_expr4 in expr:
        expr = expr.replace(cos_expr4, sin_expr)
    if cos_expr2 in expr:
        expr = expr.replace(cos_expr2, sin_expr)
    expr = expr.replace('cos(', 'sin(')

    expr = str(multi_div(sp.sympify(expr)))

    expr = expr.replace('+ sin(', '+ c*sin(')
    if 'c/(c + c*sin(c*x + c))'.replace('x', xvar) in expr:
        expr = expr.replace('c/(c + c*sin(c*x + c))'.replace('x', xvar), 'c/(c + sin(c*x + c))'.replace('x', xvar))
    if '/(c*sin(c*x + c) + c)'.replace('x', xvar) in expr:
        expr = expr.replace('/(c*sin(c*x + c) + c)'.replace('x', xvar), '/(sin(c*x + c) + c)'.replace('x', xvar))

    # 2. Redundant sums
    symexpr = sp.sympify(expr)
    if isinstance(symexpr, sp.Add):
        ct = 0
        for arg in symexpr.args:
            if 'c*' + xvar == str(arg) or xvar == str(arg):
                ct += 1
        if ct >= 2:
            expr = 0
            for arg in symexpr.args:
                if xvar != str(arg):
                    expr += arg
        expr = str(expr)

    # 3. Log equivalence
    if 'log(c*Abs(x) + c)'.replace('x', xvar) in expr:
        expr = expr.replace('log(c*Abs(x) + c)'.replace('x', xvar), 'log(c + Abs(x))'.replace('x', xvar))

    # 4 exp equivalence
    # if 'c*exp(c*x)'.replace('*x', '*' + xvar) in expr:
    #     expr = expr.replace('c*exp(c*x)'.replace('*x', '*' + xvar), 'c*exp(x)'.replace('(x', '(' + xvar))

    # 2. ... TODO: Include other type of equivalences

    return sp.sympify(expr)
