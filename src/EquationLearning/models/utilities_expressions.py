import sympy
import sympy as sp
from scipy.stats import pearsonr
from sympy.utilities.iterables import flatten
# from src.EquationLearning.models.functions import *
# from src.EquationLearning.models.NNModel import NNModel
# from src.EquationLearning.models.symbolic_expression import round_expr


def get_args(xp, return_symbols=False):
    """Extract all numeric arguments of an expression
    :param xp: Symbolic expression
    :param return_symbols: If True, it also returns the symbols in the expression"""
    args = xp.args
    num_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args = args[0]
        if args.is_number or isinstance(args, sp.Symbol):
            args = [args]
        else:
            args = args.args

    for arg in args:
        if arg.is_number or (isinstance(arg, sp.Symbol) and str(arg) == 'c'):  # If it's a number, add it to the list
            num_args.append(arg)
        elif isinstance(arg, sp.Symbol) and return_symbols:
            num_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            num_args = num_args + get_args(arg, return_symbols=return_symbols)
    return num_args


def set_args(xp, target_args, return_symbols=False):
    """Set all numeric arguments of an expression"""
    args = xp.args
    new_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        if xp.args[0].is_number or (isinstance(xp.args[0], sp.Symbol) and str(xp.args[0]) == 'c'):  # If it's a number, add it to the list
            new_args.append(target_args.pop(0))
        elif isinstance(xp.args[0], sp.Symbol):
            new_args.append(xp.args[0])
        else:  # If it's composed, explore a lower level of the tree
            new_args.append(set_args(xp.args[0], target_args))
        new_args.append(xp.args[1])
    else:
        for arg in args:
            if arg.is_number or (isinstance(arg, sp.Symbol) and str(arg) == 'c'):  # If it's a number, add it to the list
                new_args.append(target_args.pop(0))
            elif isinstance(arg, sp.Symbol) and not return_symbols:
                new_args.append(arg)
            elif isinstance(arg, sp.Symbol) and return_symbols:
                new_args.append(target_args.pop(0))
            else:  # If it's composed, explore a lower level of the tree
                new_args.append(set_args(arg, target_args, return_symbols=return_symbols))
    new_xp = xp.func(*new_args)
    return new_xp


def get_op(xp, n):
    """Extract the operator in which the n-th argument is inside of
    :param xp: Symbolic expression
    :param n: The n-th numeric argument"""
    args = xp.args

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args = args[0]
        if args.is_number or isinstance(args, sp.Symbol) or (isinstance(args, sp.Symbol) and str(args) == 'c'):
            args = [args]
        else:
            args = args.args

    ops = []
    for arg in args:
        if arg.is_number or (isinstance(arg, sp.Symbol) and str(arg) == 'c'):  # If it's a number, add it to the list
            if n == 0:
                return str(xp.func)
            else:
                n -= 1
        else:  # If it's composed, explore a lower level of the tree
            op = get_op(arg, n)
            if len(op) == 3:
                ops = op
            else:
                if isinstance(op, list):
                    ops = [str(xp.func)] + op
                else:
                    ops = [str(xp.func)] + [op]
    return ops


def get_op_constant(xp, c):
    """Extract the operator in which the specified constant name is inside (ignoring mult or add)
    :param xp: Symbolic expression
    :param c: Constant name"""
    args = xp.args

    for arg in args:
        if c in str(arg):
            if arg.is_Symbol and str(arg) == c:
                return True  # str(xp.func)
            else:  # If it's composed, explore a lower level of the tree
                op = get_op_constant(arg, c)
                if op is True:
                    if len(arg.args) == 1:  # It will return the first unary operation the constant is in
                        return str(arg.func)
                    else:
                        return op
                else:
                    return op
        else:
            continue


def add_constant_identifier(sk, cm_counter=1, ca_counter=1):
    """Label each coefficient of a skeleton expression"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol) and str(arg) == 'c':
            if isinstance(sk, sp.Add):
                new_args.append(sp.sympify('ca_' + str(ca_counter)))
                ca_counter += 1
            else:
                new_args.append(sp.sympify('cm_' + str(cm_counter)))
                cm_counter += 1
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            deep_xp = add_constant_identifier(arg, cm_counter=cm_counter, ca_counter=ca_counter)
            new_args.append(deep_xp)

    new_xp = sk.func(*new_args)
    return new_xp


def check_forbidden_combination(xp):
    args = xp.args
    res = []

    forbidden_group1 = [sp.exp, sp.log]
    forbidden_group2 = [sp.exp, sp.sinh, sp.cosh, sp.tanh, sp.tan]
    forbidden_group3 = [sp.sin, sp.cos]
    forbidden_group4 = [sp.asin, sp.acos, sp.atan]
    unary_ops = [sp.exp, sp.sinh, sp.cosh, sp.tanh, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sqrt]

    for arg in args:
        if arg.is_number or isinstance(arg, sp.Symbol):
            res.append(False)
        else:
            g1 = any([arg.func == op for op in forbidden_group1])
            if g1:
                args2 = arg.args
                g12 = any([str(op)+'(' in str(args2) for op in forbidden_group1])
                if g12:
                    return True
            g2 = any([(arg.func == op) or (arg.func == sp.Pow and (arg.args[1] > 2)) for op in forbidden_group2])
            if g2:
                args2 = arg.args
                g22 = any([(str(op)+'(' in str(args2)) or ('**3' in str(args2)) or ('**4' in str(args2)) or
                           ('**5' in str(args2)) for op in forbidden_group2])
                if g22:
                    return True
            g3 = any([arg.func == op for op in forbidden_group3])
            if g3:
                args2 = arg.args
                g32 = 'tan' in str(args2)
                if g32:
                    return True
            if arg.func == sp.tan:
                args2 = arg.args
                g32 = any([str(op)+'(' in str(args2) for op in forbidden_group3])
                if g32:
                    return True
            g4 = any([arg.func == op for op in forbidden_group4])
            if g4:
                args2 = arg.args
                g42 = any([str(op)+'(' in str(args2) for op in forbidden_group4])
                if g42:
                    return True
            # Lastly, check if there are more than 3 nested unary operations
            g5 = any([arg.func == op for op in unary_ops])
            if g5:
                args2 = arg.args
                g52 = sum([str(op)+'(' in str(args2) for op in unary_ops])
                if g52 >= 2:
                    return True

            res.append(check_forbidden_combination(arg))

    return any(res)







# def verify_dependency(xp_orig, list_symbols, gen_fun, values, limits, variable, resample_var, r_orig):
#     """Randomly change value of variable "resample_var" and check if the correlation value of the overall equation
#     changes"""
#     # Sample values of the variable that is being analyzed
#     sample = np.random.uniform(limits[variable][0], limits[variable][1], 1000)
#     ref = values[resample_var]  # Save original value of variable that will be resampled
#     res_value = np.random.uniform(limits[resample_var][0], limits[resample_var][1])
#     while np.abs(res_value - ref) < 0.3:
#         # If sampled value is close to the original, keep resampling
#         res_value = np.random.uniform(limits[resample_var][0], limits[resample_var][1])
#     values[resample_var] = res_value
#     values = np.repeat(values, 1000, axis=1)
#     values[variable] = sample
#
#     # Obtain the estimated outputs using the generating function (e.g., the NN)
#     if isinstance(gen_fun, NNModel):  # Used if gen_fun is a neural network
#         y = np.array(gen_fun.evaluateFold(values.T, batch_size=len(values)))[:, 0]
#     else:  # Used if gen_fun is a symbolic expressions
#         y = gen_fun(*list(values))
#
#     # Lambdify to evaluate
#     expr = sp.lambdify(flatten(list_symbols), xp_orig)
#     # Evaluate original expression
#     y_pred = expr(*list(values))
#     return (r_orig - np.round(np.abs(pearsonr(y, y_pred)[0]), 5)) > 0.0003


# def get_polynomial(variable, coeff, calc_complexity=True):
#     """Retrieve polynomial symbolic expression"""
#     xp = 0
#     for d in range(len(coeff)):
#         if isinstance(coeff[d], float) or isinstance(coeff[d], int):
#             if np.abs(coeff[d]) < 0.001:
#                 coeff[d] = 0
#         xp += coeff[d] * variable ** (len(coeff) - d)
#
#     if calc_complexity:  # Only calculate complexity when specified
#         if len(coeff) <= 2:
#             comp = np.sum(np.abs(coeff) >= 0.01)
#         else:
#             comp = 2 * np.sum(np.abs(coeff[:len(coeff) - 2]) >= 0.01) + np.sum(np.abs(coeff[len(coeff) - 2:]) >= 0.01)
#         return round_expr(xp, 4), comp
#     else:
#         return round_expr(xp, 4)


# def get_expression(variable, coeff, operator, inv, calc_complexity=True):
#     """Retrieve simplified symbolic expression"""
#     comp = 0
#     if coeff[0] == 1:
#         xp = get_sym_function(operator)[0]((coeff[1] * variable + coeff[2]) + coeff[3])
#         if calc_complexity:  # Only calculate complexity when specified
#             comp = get_sym_function(operator)[2] + (np.abs(coeff[2] + coeff[3]) > 0.001) * .5 + (
#                     np.abs(coeff[4]) > 0.001)
#     else:
#         xp = get_sym_function(operator)[0](1 / (coeff[1] * variable + coeff[2]) + coeff[3])
#         if calc_complexity:  # Only calculate complexity when specified
#             comp = get_sym_function(operator)[2] + 1 + (np.abs(coeff[2]) > 0.001) * .5 + \
#                    (np.abs(coeff[3]) > 0.001) * .5 + (np.abs(coeff[4]) > 0.001)
#     # Divide by a constant so that the first coefficient is always 1
#     if isinstance(xp + coeff[4], sp.Mul):
#         xp = (xp + coeff[4]).args[1]
#     elif isinstance(xp, sp.Mul):
#         xp = xp.args[1] + coeff[4] / xp.args[0]
#     else:
#         xp = xp + coeff[4]
#     if inv:
#         xp = 1 / xp
#         if calc_complexity:
#             comp += 2
#
#     if calc_complexity:
#         return round_expr(xp, 4), comp
#     else:
#         return round_expr(xp, 4)
#
#
# def get_expression_nested(variable, coeff, operator1, secondary_operators, inv, calc_complexity=True):
#     """Retrieve simplified symbolic expression"""
#     comp = 0
#     operator2 = secondary_operators[int(coeff[-1])]
#     inv1 = coeff[0]
#     inv2 = coeff[1]
#     x1 = [coeff[k] for k in range(2, 6)]
#     x2 = [coeff[k] for k in range(6, 10)]
#
#     # TODO: if operator 2 is sin, apply mod
#
#     xp = get_sym_function(operator1)[0]((x1[0] * variable + x1[1]) ** inv1 + x1[2]) + x1[3]
#     xp = get_sym_function(operator2)[0]((x2[0] * xp + x2[1]) ** inv2 + x2[2])
#
#     if calc_complexity:  # Only calculate complexity when specified
#         if inv1 == 1:
#             comp = get_sym_function(operator1)[2] + (np.abs(x1[1] + x1[2]) > 0.001) * .5 + (np.abs(x1[3]) > 0.001)
#         else:
#             comp = get_sym_function(operator1)[2] + 1 + (np.abs(x1[1]) > 0.001) * .5 + (np.abs(x1[2]) > 0.001) * .5 + \
#                    (np.abs(x1[3]) > 0.001)
#
#         if inv2 == 1:
#             comp += get_sym_function(operator2)[2] + (np.abs(x2[1] + x2[2]) > 0.001) * .5 + (np.abs(x2[3]) > 0.001)
#         else:
#             comp += get_sym_function(operator2)[2] + 1 + (np.abs(x2[1]) > 0.001) * .5 + (np.abs(x2[2]) > 0.001) * .5 + \
#                     (np.abs(x2[3]) > 0.001)
#
#     # Divide by a constant so that the first coefficient is always 1
#     if isinstance(xp + x2[3], sp.Mul):
#         xp = (xp + x2[3]).args[1]
#     elif isinstance(xp, sp.Mul):
#         xp = xp.args[1] + x2[3] / xp.args[0]
#     else:
#         xp = xp + x2[3]
#
#     if inv:
#         xp = 1 / xp
#         if calc_complexity:
#             comp += 2
#
#     if calc_complexity:
#         return round_expr(xp, 4), comp
#     else:
#         return round_expr(xp, 4)


def avoid_operations_between_constants(xp):
    """ Simplify exponent of constants inside symbolic expression and constant multiplied by constants.
    E.g., constant ** 3 = constant or cons1 * cons2 * cons3 = cons1
    :param xp: Symbolic expression"""
    args = xp.args
    t_args = args
    new_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args1 = args[0]
        args2 = args[1]
        if (args1.is_number or (isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)))) and \
                (args2.is_number or (isinstance(args2, sp.Symbol) and ("cm" in str(args2) or "ca" in str(args2) or "c" in str(args2)))):
            va = None
            if isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)):
                va = args1
            elif isinstance(args2, sp.Symbol) and ("cm" in str(args2) or "ca" in str(args2) or "c" in str(args2)):
                va = args2
            t_args = [va, sympy.sympify("1")]

    if isinstance(xp, sp.Mul):
        t_args = []
        # Check if two or more of the arguments are constants and replace them for just one constant
        flag = False  # This flag turns to True when a constant was already found
        for arg in args:
            if isinstance(arg, sp.Symbol) and ("cm" in str(arg) or "ca" in str(arg) or "c" in str(arg)):
                if not flag:
                    flag = True
                else:
                    arg = sympy.sympify("1")
            t_args.append(arg)

    for arg in t_args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            new_args.append(avoid_operations_between_constants(arg))

    if len(new_args) > 0:
        new_xp = xp.func(*new_args)
    else:
        new_xp = xp

    if len(new_args) == 1 and new_args[0] == sp.sympify('c'):  # If it's a unary operation and it's argument is a constant
        new_xp = sp.sympify('c')

    return new_xp


if __name__ == '__main__':
    s = sp.sympify('cm_0*cosh(ca_1 + cm_1*x_1 + exp(cm_2*x_1))')
    rt = check_forbidden_combination(s)
    print()
