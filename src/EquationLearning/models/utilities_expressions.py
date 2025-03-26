import sympy
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from EquationLearning.Data.sympy_utils import numeric_to_placeholder


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

    return True


def get_all_op_constant(xp, c, ops=None):
    """Extract all the operators in which the specified constant name is inside
    :param xp: Symbolic expression
    :param c: Constant name
    :param ops: Current identified ops"""
    args = xp.args
    if ops is None:
        ops = [str(xp.func)]

    for arg in args:
        if str(c) in str(arg):
            if arg.is_Symbol and str(arg) == str(c):
                return ops
            else:  # If it's composed, explore a lower level of the tree
                if isinstance(arg, sp.Pow):  # In case it's Pow, return the exponent too
                    ops.append(str(arg.func) + str(arg.args[1]))
                else:  # It will return the first unary operation the constant is in
                    ops.append(str(arg.func))
                return get_all_op_constant(arg, c, ops=ops)
        else:
            continue


def find_node_in_ops(xp, ops, prev_ops=None, sols=None):
    """Find a node in a expression tree that is inside the specified list of operations
    :param xp: Symbolic expression
    :param ops: List of operations
    :param prev_ops: Previous operations
    :param sols: Current identified node solutions"""
    if prev_ops is None:
        sols = []
        if str(xp.func) == ops[0]:
            if ops == [str(xp.func)]:
                return [xp]
            else:
                return find_node_in_ops(xp, ops, prev_ops=[str(xp.func)], sols=sols)
        else:
            return None

    args = xp.args
    for arg in args:
        if str(arg.func) == ops[len(prev_ops)]:
            prev_ops_temp = prev_ops.copy()
            prev_ops_temp.append(str(arg.func))
            if ops == prev_ops_temp:
                sols.append(arg)
            elif len(ops) > len(prev_ops_temp):
                sols = find_node_in_ops(arg, ops, prev_ops=prev_ops_temp, sols=sols)
        else:
            continue
    return sols


def add_constant_identifier(sk, cm_counter=1, ca_counter=1):
    """Label each coefficient of a skeleton expression"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol) and 'c' in str(arg):
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


def multi_div(sk):
    """Multiply elements from the dividend by a constant"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Pow):
            base = arg.args[0]
            if arg.args[1] < 0 and isinstance(arg.args[0], sp.Add):
                base = 0
                for argsum in arg.args[0].args:
                    base += avoid_operations_between_constants(sp.sympify('c') * argsum)
            new_args.append(base ** arg.args[1])
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            new_args.append(multi_div(arg))

    new_xp = sk.func(*new_args)
    return new_xp


def change_sign_k_constant(sk, k, c_counter=1):
    """Change sign of the k-th constant or symbol of expr"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
            c_counter += 1
        elif isinstance(arg, sp.Symbol) and 'x' in str(arg):
            if k == c_counter:
                arg = -arg
            new_args.append(arg)
            c_counter += 1
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
            c_counter += 1
        else:  # If it's composed, explore a lower level of the tree
            deep_xp, c_counter = change_sign_k_constant(arg, k=k, c_counter=c_counter)
            new_args.append(deep_xp)

    new_xp = sk.func(*new_args)
    return new_xp, c_counter


def remove_constant_identifier(sk):
    """Remove the coefficient labels of a skeleton expression and replace them with a placeholder"""
    args = sk.args
    new_args = []

    for arg in args:
        if arg.is_number:  # If it's a number, add it to the list
            new_args.append(arg)
        elif isinstance(arg, sp.Symbol) and (('ca_' in str(arg)) or ('cm_' in str(arg))):
            new_args.append(sp.sympify('c'))
        elif isinstance(arg, sp.Symbol):
            new_args.append(arg)
        else:  # If it's composed, explore a lower level of the tree
            deep_xp = remove_constant_identifier(arg)
            new_args.append(deep_xp)

    new_xp = sk.func(*new_args)
    return new_xp


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

    if isinstance(xp, sp.exp):  # If it's an exp function, ignore the power and focus only on the base
        args1 = args[0]
        if (args1.is_number or (
                isinstance(args1, sp.Symbol) and ("cm" in str(args1) or "ca" in str(args1) or "c" in str(args1)))):
            t_args = tuple([sympy.sympify("0")])

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


def count_variables(expr):
    expr = parse_expr(str(expr))
    count = 0
    for atom in expr.atoms():
        if str(atom).startswith('x') or str(atom).startswith('cm_'):
            count += 1
    return count


def expr2skeleton(expr):
    """Compare the coefficients of a skeleton to a known expression and remove useless coefficients"""
    # args_expr = get_args(expr)
    # args_expr = [e if abs(e) > 0.001 else 0 for e in args_expr]
    # expr = set_args(expr, args_expr)
    return numeric_to_placeholder(expr)


def count_nodes0(expr):
    node_count = 0
    for _ in sp.preorder_traversal(expr):
        node_count += 1
    return node_count


def count_nodes(exprr):
    unary_ops = 0
    binary_ops = 0

    # Define a helper function to traverse the expression
    def traverse(expr):
        nonlocal unary_ops, binary_ops

        # Check if the expression is an operation
        if isinstance(expr, sp.Basic):
            if isinstance(expr, sp.Add):
                binary_ops += 1
            if isinstance(expr, sp.Mul) or isinstance(expr, sp.Pow):
                if not(isinstance(expr, sp.Mul) and any([('c' in str(arg) and arg.is_Symbol and len(expr.args) == 2) for arg in expr.args])):
                    binary_ops += 1
                    if isinstance(expr, sp.Mul):
                        binary_ops += len(expr.args)
                for arg in expr.args:
                    traverse(arg)
            elif isinstance(expr, sp.Function):
                unary_ops += 1
                for arg in expr.args:
                    traverse(arg)
            elif isinstance(expr, sp.Derivative):
                unary_ops += 1
                traverse(expr.args[0])
                for sym in expr.args[1:]:
                    if isinstance(sym, tuple):
                        unary_ops += len(sym)
                    else:
                        unary_ops += 1
            # elif isinstance(expr, sp.Rational):
            #     # Rational is a binary operation (numerator/denominator)
            #     binary_ops += 1
            # elif isinstance(expr, sp.Number):
            #     # Numbers are not operations
            #     pass
            else:
                for arg in expr.args:
                    traverse(arg)

    traverse(exprr)
    return unary_ops + binary_ops


def remove_coeffs(skeleton):
    args = skeleton.args
    if isinstance(skeleton, sp.Add):
        terms = 0
        for arg in args:
            if isinstance(arg, sp.Symbol) and 'c' in str(arg):
                terms += 0
            else:
                terms += arg
        skeleton = terms

    # args = skeleton.args
    # if isinstance(skeleton, sp.Add) and all([isinstance(arg, sp.Mul) for arg in args]) and all(['c' in str(arg)[:2] for arg in args]):
    #     terms = 0
    #     for ia, arg in enumerate(args):
    #         if ia == 0:
    #             terms += sp.sympify(arg.args[1])
    #         else:
    #             terms += arg
    #     skeleton = terms

    args = skeleton.args
    if isinstance(skeleton, sp.Mul) and len(args) == 2:
        terms = 1
        for arg in args:
            if isinstance(arg, sp.Symbol) and 'c' in str(arg):
                terms *= 1
            else:
                terms *= arg
        skeleton = terms

    return skeleton


SYMPY_OPERATORS = {
        # Elementary functions
        sp.Pow: "pow",
        sp.exp: "exp",
        sp.log: "ln",
        sp.Abs: 'abs',
        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        # Trigonometric Inverses
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",
        # Hyperbolic Functions
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",
    }


def check_if_inside_unary_ops(depend_var, skeleton, ops=None):
    """
    Determine inside which operators of the given skeleton there is a coefficient that depends on the variable depend_var
    """
    if ops is None:
        ops = []

    for arg in skeleton.args:
        if arg.is_number or isinstance(arg, sp.Symbol):
            continue
        elif depend_var in str(arg):
            if arg.func in list(SYMPY_OPERATORS.keys()):
                if len(arg.args) == 1:
                    if depend_var in str(arg.args):
                        # If the unary operator contains a coefficient that depends on the variable under analysis, add it
                        ops.append(SYMPY_OPERATORS[arg.func])
                        # And keep going deeper and check if there's an inner unary operator that contains the same dependency
                        ops = check_if_inside_unary_ops(depend_var, arg, ops=ops)
                if len(arg.args) == 2:
                    if depend_var in str(arg.args[0]):
                        ops.append(SYMPY_OPERATORS[arg.func])
                        ops = check_if_inside_unary_ops(depend_var, arg, ops=ops)
                    if depend_var in str(arg.args[1]):
                        ops.append(SYMPY_OPERATORS[arg.func])
                        ops = check_if_inside_unary_ops(depend_var, arg, ops=ops)
            else:
                ops = check_if_inside_unary_ops(depend_var, arg, ops=ops)

    return ops


# oops = check_if_inside_unary_ops('f1', sp.sympify('cm_2*sin(f1*x0 + f1) + f1*x0 + f1'))

# Example usage
# x, y = sp.symbols('x y')
# expr = sp.sin(x) + x**2 - sp.log(y)
# expr0 = sp.sympify('cm_1*x0 + cm_2*x2 + cm_3*x0**2 + cm_4*x2**2 + cm_5*x0**4 + cm_6*x2**4')
# print(count_nodes(expr0))
# expr1 = sp.sympify('cm_1*x0 + cm_2*x2**2 + cm_3*x0**4 + cm_4*x2**4 + cm_5*x0**2*x2')
# print(count_nodes(expr1))
