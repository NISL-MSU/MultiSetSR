import re
import sympy as sp
from EquationLearning.models.utilities_expressions import multi_div


def sk_equivalence(expr, alts=False):
    """If possible, transform the skeleton into a mathematically equivalent skeleton"""
    if 'c*' not in str(expr)[0:3]:
        expr = sp.sympify('c') * expr
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

    # 4. Manage division
    if ('c/(c*x**2 + c)'.replace('x', xvar) in expr) and alts:
        expr = expr.replace('c/(c*x**2 + c)'.replace('x', xvar), '(c*x + c)/(c*x**2 + c)'.replace('x', xvar))
    if 'c/(c*x**2 + c*x + c)'.replace('x', xvar) in expr:
        expr = expr.replace('c/(c*x**2 + c*x + c)'.replace('x', xvar), '(c*x + c)/(c*x**2 + c*x + c)'.replace('x', xvar))
        expr2 = expr.replace('(c*x + c)/(c*x**2 + c*x + c)'.replace('x', xvar), 'c/(x + c)'.replace('x', xvar))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if 'c/(c*x**3 + c)'.replace('x', xvar) in expr:
        expr = expr.replace('c/(c*x**3 + c)'.replace('x', xvar), '(c*x + c)/(c*x**3 + c*x + c)'.replace('x', xvar))
    if 'c/(c*x**4 + c)'.replace('x', xvar) in expr:
        expr = expr.replace('c/(c*x**4 + c)'.replace('x', xvar), '(c*x + c)/(c*x**4 + c*x + c)'.replace('x', xvar))
    if re.sub(r'\bx\b', xvar, 'x/(c*x**2 + c*x + c)') in expr and alts:
        expr2 = expr.replace(re.sub(r'\bx\b', xvar, 'x/(c*x**2 + c*x + c)'), re.sub(r'\bx\b', xvar, '1/(x**2 + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]

    # 4. Distributive multiplication
    if 'c*(c*x' in expr:
        expr = expr.replace('c*(c*x', '(c*x')

    expr2 = expr
    if (re.sub(r'\bx\b', xvar, 'sin((c*x + c)**2)') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'sin((c*x + c)**2)'),
                              re.sub(r'\bx\b', xvar, 'sin(c*x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, '(c*sin(c*x1 + c) + c)**2') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, '(c*sin(c*x1 + c) + c)**2'),
                              re.sub(r'\bx\b', xvar, 'c*sin(c*x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, 'sin((c*x + c)**2 + c)') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'sin((c*x + c)**2 + c)'),
                              re.sub(r'\bx\b', xvar, 'sin(c*x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, 'c*x*sin(c*x + c)') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*x*sin(c*x + c)'),
                              re.sub(r'\bx\b', xvar, 'c*sin(c*x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]

    # 5. If exp(cx) is considered, exp(cx^2) should also be considered
    expr2 = expr
    if re.sub(r'\bx\b', xvar, 'exp(c*x)') in expr and alts:
        expr2 = expr.replace(re.sub(r'\bx\b', xvar, 'exp(c*x)'), re.sub(r'\bx\b', xvar, 'exp(c*x**2)'))
        expr3 = expr.replace(re.sub(r'\bx\b', xvar, 'exp(c*x)'), re.sub(r'\bx\b', xvar, 'c*x**2 + c*x'))
        return [sp.sympify(expr), sp.sympify(expr2), sp.sympify(expr3)]

    # 6. If abs(cx) is considered, sqrt(cx^2) should also be considered
    expr2 = expr
    if ((re.sub(r'\bx\b', xvar, 'c*Abs(c + x)') in expr) and (
            re.sub(r'\bx\b', xvar, '(c*Abs(c + x)') not in expr)) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*Abs(c + x)'),
                              re.sub(r'\bx\b', xvar, 'c*sqrt(c + (c + c*x)**2)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, 'c*sqrt(c*Abs(c + x) + c)') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*sqrt(c*Abs(c + x) + c)'),
                              re.sub(r'\bx\b', xvar, 'c*sqrt(c + (c + c*x)**2)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, 'c*sqrt(c*x + c)') == expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*sqrt(c*x + c)'),
                              re.sub(r'\bx\b', xvar, 'c*log(c*x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, 'c*x + c*sqrt(c + x)') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*x + c*sqrt(c + x)'),
                              re.sub(r'\bx\b', xvar, 'c*sqrt(x**2 + c*x + c)'))
        expr3 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*sqrt(x**2 + c*x + c)'), re.sub(r'\bx\b', xvar, 'c*Abs(x + c)'))
        expr4 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*sqrt(x**2 + c*x + c)'), re.sub(r'\bx\b', xvar, 'c*(x + c)**2'))
        return [sp.sympify(expr), sp.sympify(expr2), sp.sympify(expr3), sp.sympify(expr4)]
    if (re.sub(r'\bx\b', xvar, 'c*sqrt(c*Abs(x) + c)') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*sqrt(c*Abs(x) + c)'),
                              re.sub(r'\bx\b', xvar, 'c*sqrt(c + (c + c*x)**2)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if (re.sub(r'\bx\b', xvar, '(c*log(c*x + c) + c)**2') in expr) and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, '(c*log(c*x + c) + c)**2'),
                              re.sub(r'\bx\b', xvar, 'c*sqrt(c + (c + c*x)**2)'))
        return [sp.sympify(expr), sp.sympify(expr2)]

    # 7. Division
    expr2 = expr
    if re.sub(r'\bx\b', xvar, '/(c*sqrt(c + x) + c)') in expr and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, '/(c*sqrt(c + x) + c)'), re.sub(r'\bx\b', xvar, '/(x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]
    if re.sub(r'\bx\b', xvar, '/(c*sqrt(c*x + c) + c)') in expr and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, '/(c*sqrt(c*x + c) + c)'), re.sub(r'\bx\b', xvar, '/(x + c)'))
        if re.sub(r'\bx\b', xvar, 'c*x/(x + c)') in expr2 and alts:
            expr3 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*x/(x + c)'), re.sub(r'\bx\b', xvar, 'c/(x + c)'))
            return [sp.sympify(expr), sp.sympify(expr2), sp.sympify(expr3)]
        return [sp.sympify(expr), sp.sympify(expr2)]
    if re.sub(r'\bx\b', xvar, 'c*sqrt(c*log(c*x + c) + c)') in expr and alts:
        expr2 = expr2.replace(re.sub(r'\bx\b', xvar, 'c*sqrt(c*log(c*x + c) + c)'), re.sub(r'\bx\b', xvar, 'c/(x + c)'))
        return [sp.sympify(expr), sp.sympify(expr2)]

    # If there's a sum c*x + f(x), consider only "c*x" as well
    if re.sub(r'\bx\b', xvar, 'c*x +') in expr[0:6] and alts:
        return [sp.sympify(expr), sp.sympify(re.sub(r'\bx\b', xvar, 'c*x'))]

    if alts:
        return [sp.sympify(expr)]
    else:
        return sp.sympify(expr)
