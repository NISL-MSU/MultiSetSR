import sympy
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from src.EquationLearning.Data.sympy_utils import numeric_to_placeholder


def get_args(xp, return_symbols=False):
    """Extract all numeric arguments and coefficients of an expression
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
        if arg.is_number or (isinstance(arg, sp.Symbol) and 'c' in str(arg)):  # If it's a number or coeff, add it
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
        if xp.args[0].is_number or (
                isinstance(xp.args[0], sp.Symbol) and str(xp.args[0]) == 'c'):  # If it's a number, add it to the list
            new_args.append(target_args.pop(0))
        elif isinstance(xp.args[0], sp.Symbol):
            new_args.append(xp.args[0])
        else:  # If it's composed, explore a lower level of the tree
            new_args.append(set_args(xp.args[0], target_args))
        new_args.append(xp.args[1])
    else:
        for arg in args:
            if arg.is_number or (
                    isinstance(arg, sp.Symbol) and 'c' in str(arg)):  # If it's a number or coeff, add it to the list
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
        if args.is_number or isinstance(args, sp.Symbol) or (isinstance(args, sp.Symbol) and 'c' in str(args)):
            args = [args]
        else:
            args = args.args

    ops = []
    for arg in args:
        if arg.is_number or (isinstance(arg, sp.Symbol) and 'c' in str(arg)):  # If it's a number, add it to the list
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
                    elif isinstance(arg, sp.Pow):  # In case it's Pow, return the exponent too
                        return str(arg.func), arg.args[1]
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
            deep_xp, cm_counter, ca_counter = add_constant_identifier(arg, cm_counter=cm_counter, ca_counter=ca_counter)
            new_args.append(deep_xp)

    new_xp = sk.func(*new_args)
    return new_xp, cm_counter, ca_counter


def check_forbidden_combination(xp):
    args = xp.args
    res = []

    forbidden_group1 = [sp.Abs]
    forbidden_group2 = [sp.exp, sp.sinh, sp.cosh, sp.tanh, sp.tan, sp.log]
    forbidden_group3 = [sp.sin, sp.cos]
    forbidden_group4 = [sp.asin, sp.acos, sp.atan]
    unary_ops = [sp.Pow, sp.exp, sp.log, sp.sinh, sp.cosh, sp.tanh, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan,
                 sp.sqrt, 'sqrt']

    for arg in args:
        if arg.is_number or isinstance(arg, sp.Symbol):
            res.append(False)
        else:
            g1 = any([arg.func == op for op in forbidden_group1])
            if g1:
                args = arg.args
                g12 = any([(str(op) + '(' in str(args)) or ('sqrt' in str(args)) or ('**2' in str(args)) or
                           ('**4' in str(args)) for op in forbidden_group1])
                if g12:
                    return True
            g2 = any([(arg.func == op) or (arg.func == sp.Pow and (arg.args[1] > 2)) for op in forbidden_group2])
            if g2:
                args2 = arg.args
                g22 = any([(str(op) + '(' in str(args2)) or ('**3' in str(args2)) or ('**4' in str(args2)) or
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
                g32 = any([str(op) + '(' in str(args2) for op in forbidden_group3])
                if g32:
                    return True
            g4 = any([arg.func == op for op in forbidden_group4])
            if g4:
                args2 = arg.args
                g42 = any([str(op) + '(' in str(args2) for op in forbidden_group4])
                if g42:
                    return True
            # Lastly, check if there are more than 3 nested unary operations
            g5 = any([arg.func == op for op in unary_ops])
            if g5:
                args2 = arg.args
                g52 = sum([str(op) + '(' in str(args2) for op in unary_ops])
                g52 += str(args2).count('**')
                if g52 > 1:
                    return True

            res.append(check_forbidden_combination(arg))

    return any(res)


def _avoid_operations_between_constants(xp):
    """ Simplify exponent of constants inside symbolic expression and constant multiplied by constants.
    E.g., constant ** 3 = constant or cons1 * cons2 * cons3 = cons1
    :param xp: Symbolic expression"""
    args = xp.args
    t_args = args
    new_args = []

    if isinstance(xp, sp.Pow):  # If it's a power function, ignore the power and focus only on the base
        args1 = args[0]
        args2 = args[1]
        if (args1.is_number or (
                isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)))) and \
                (args2.is_number or (isinstance(args2, sp.Symbol) and (
                        "cm" in str(args2) or "ca" in str(args2) or "c" in str(args2)))):
            va = None
            if isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)):
                va = args1
            elif isinstance(args2, sp.Symbol) and ("cm" in str(args2) or "ca" in str(args2) or "c" in str(args2)):
                va = args2
            t_args = [va, sympy.sympify("1")]

    if isinstance(xp, sp.Mul) or isinstance(xp, sp.Add):
        t_args = []
        # Check if two or more of the arguments are constants and replace them for just one constant
        flag = False  # This flag turns to True when a constant was already found
        for ia, arg in enumerate(args):
            if arg.is_number and any([(isinstance(co, sp.Symbol) and 'c' in str(co)) for co in (args[0:ia] + args[ia + 1:])]):
                # If the current number is summed or multiplied by other constants, replace it by 0 or one
                if isinstance(xp, sp.Mul):
                    arg = sympy.sympify("1")
                else:
                    arg = sympy.sympify("0")
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
            new_args.append(_avoid_operations_between_constants(arg))

    if len(new_args) > 0:
        new_xp = xp.func(*new_args)
    else:
        new_xp = xp

    if len(new_args) == 1 and 'c' == str(new_args[0]):  # If it's a unary operation and it's argument is a constant
        new_xp = sp.sympify('c')

    return new_xp


def avoid_operations_between_constants(xp):
    """Repeat the _avoid_operations_between_constants function until no change is found"""
    xp2 = None
    while xp != xp2:
        xp2 = xp
        xp = _avoid_operations_between_constants(xp)
    return xp


def get_skeletons(expr, var_names):
    # Get a skeleton for each variable present in the expression
    skeletons = []
    for var in var_names:
        skeleton = get_skeleton_var(expr, var, var_names, expand=True)
        # skeleton2 = get_skeleton_var(expr, var, var_names, expand=False)
        # if count_placeholders(skeleton) > count_placeholders(skeleton2):
        #     skeleton = skeleton2  # Choose the skeleton form with the fewest coefficient placeholders
        skeletons.append(skeleton)
    return skeletons


def get_skeleton_var(expr, var, var_names, expand=True):
    skeleton = numeric_to_placeholder(expr, var=var)
    for v in var_names:
        if v != var:
            skeleton = skeleton.subs(sp.sympify(v), sp.sympify('c'))
    skeleton2 = None
    if expand:
        skeleton = sp.expand(skeleton)
    while skeleton != skeleton2:
        skeleton2 = skeleton
        skeleton = avoid_operations_between_constants(skeleton)
        skeleton = numeric_to_placeholder(skeleton, var=var)
    if 'x' in str(skeleton):  # If there are no variables in the expression, skip adding identifiers
        skeleton = add_constant_identifier(skeleton)[0]
    return skeleton


def count_placeholders(expr):
    expr = parse_expr(str(expr))
    count = 0
    for atom in expr.atoms():
        if str(atom).startswith('ca_') or str(atom).startswith('cm_'):
            count += 1
    return count
