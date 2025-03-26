import sympy
import random
import numpy as np
from EquationLearning.Optimization.GP import GP
from EquationLearning.Data.math_rules import sk_equivalence
from EquationLearning.models.utilities_expressions import add_constant_identifier, get_args, get_all_op_constant, \
    find_node_in_ops, avoid_operations_between_constants, get_skeletons, expr2skeleton, remove_constant_identifier, set_args


def add_constants(expr):
    if not isinstance(expr, sympy.Mul) and not isinstance(expr, sympy.Add):
        expr = sympy.sympify('cm') * expr + sympy.sympify('ca')
    elif isinstance(expr, sympy.Add) and not any([isinstance(arg, sympy.Mul) for arg in expr.args]):
        expr = avoid_operations_between_constants(sympy.sympify('cm') * expr)
        expr = sympy.expand(expr)
    else:
        expr = expr + sympy.sympify('ca')

    return avoid_operations_between_constants(expr)


class MergeExpressions:

    def __init__(self, sk1, sk2, merging_variables):
        """Merge and find the best combination of two univariate symbolic skeletons
        :param sk1: Skeleton 1
        :param sk2: Skeleton 2
        """
        self.merging_variables = merging_variables
        self.sk1, cm1, ca1 = add_constant_identifier(add_constants(sk1))
        self.sk2, cm2, ca2 = add_constant_identifier(add_constants(sk2))
        num_pl1, num_pl2 = cm1 + ca1 - 2, cm2 + ca2 - 2
        [self.skBase, self.skAlt] = [self.sk1, self.sk2] if num_pl1 <= num_pl2 else [self.sk2, self.sk1]
        # Get the list of original univariate skeleton
        self.univ_sks = get_skeletons(sk1, var_names=[str(arg) for arg in sk1.free_symbols if 'x' in str(arg)])
        self.univ_sks += get_skeletons(sk2, var_names=[str(arg) for arg in sk2.free_symbols if 'x' in str(arg)])

    def merge(self):
        return self.merge_base(self.skBase, self.skAlt)

    def merge_base(self, exp1, exp2, inside_sum_with_const=False):
        """Generate a random combination of input skeletons"""

        if isinstance(exp1, sympy.Add) and isinstance(exp2, sympy.Add):
            # Check which sum has fewer arguments
            exp1_args = [sympy.sympify('cm_0') * op if not isinstance(op, sympy.Mul) and 'ca_' not in str(op) else op for op in exp1.args]
            exp2_args = [sympy.sympify('cm_0') * op if not isinstance(op, sympy.Mul) and 'ca_' not in str(op) else op for op in exp2.args]
            [exp_short, exp_long] = [exp1_args, exp2_args] if len(exp2_args) >= len(exp1_args) else [exp2_args, exp1_args]
            random.shuffle(exp_short)
            random.shuffle(exp_long)
            const = [arg.is_symbol and 'c' in str(arg) for arg in exp_short]
            flag = False
            if any(const):
                temp = exp_short.copy()
                temp[const.index(True)] = exp_short[-1]
                temp[-1] = exp_short[const.index(True)]
                exp_short = temp
                flag = True

            for i, xp in enumerate(exp_short):
                if (i == len(exp_short) - 1) and 'c' in str(xp) and len(exp_long) > 0:
                    # If there are elements from exp_long that have not been merged, merge them with the constant term
                    exp_short[i] = avoid_operations_between_constants(exp_short[i] + sum(exp_long))
                else:
                    # For each argument in exp_short, find compatible args in exp_long and choose one for merging
                    compatible_args = [op for op in exp_long if op.func == xp.func]
                    if len(compatible_args) == 0:
                        continue
                    selected_args = random.sample(compatible_args, k=random.randint(0, len(compatible_args)))
                    if len(selected_args) == 0:
                        continue
                    [exp_long.remove(s_arg) for s_arg in selected_args]
                    if len(selected_args) == 1:
                        exp_short[i] = self.merge_base(exp_short[i], selected_args[0], inside_sum_with_const=flag)
                    else:
                        exp_short[i] = exp_short[i] * sum(selected_args)
            exp1 = sum(exp_short)
        else:
            # Check if both expressions coincide in some operators that can be used for merging
            if exp1.is_symbol or exp2.is_symbol:
                exp1 = avoid_operations_between_constants(exp1 * exp2)
            elif (exp1.func == exp2.func) and not isinstance(exp1, sympy.Mul):
                # If the operators are exactly equal, just merge the arguments
                if len(exp1.args) == 1:
                    exp1 = exp1.func(*[self.merge_base(exp1.args[0], exp2.args[0])])
                elif exp1.args[1] == exp2.args[1]:  # e.g. Pow(sin(c*x1+c),2) and Pow(sin(c*x2+c),2)->Pow(sin(x2*x1+c),2)
                    arg_new = self.merge_base(exp1.args[0], exp2.args[0])
                    exp1 = exp1.func(*[arg_new, exp1.args[1]])
                else:
                    exp1 = exp1 * exp2
            elif (exp1.func == exp2.func) and isinstance(exp1, sympy.Mul):
                # Extract operators in exp1 and exp2, compare them and check coincidences
                exp1_args, exp2_args = [op for op in exp1.args], [op for op in exp2.args]
                [exp_short, exp_long] = [exp1_args, exp2_args] if len(exp2_args) >= len(exp1_args) else [exp2_args, exp1_args]
                const1 = [arg.is_symbol and 'cm_' in str(arg) for arg in exp_short]
                const2 = [arg.is_symbol and 'cm_' in str(arg) for arg in exp_long]

                if all([arg.is_symbol for arg in exp_short]):  # If there are only symbols, merge them by multiplying them
                    exp_short, exp_long = [arg for arg in exp_short if not (arg.is_symbol and 'c' in str(arg))], [arg for arg in exp_long if not (arg.is_symbol and 'c' in str(arg))]
                    if const1 and not const2:
                        exp1 = avoid_operations_between_constants(sympy.sympify('c') * (sympy.prod(exp_short) + sympy.sympify('c')) * sympy.prod(exp_long))
                    elif const2 and not const1:
                        exp1 = avoid_operations_between_constants(sympy.sympify('c') * (sympy.prod(exp_long) + sympy.sympify('c')) * sympy.prod(exp_short))
                    elif const1 and const2:
                        exp1 = avoid_operations_between_constants(sympy.sympify('c') * (sympy.prod(exp_short) + sympy.sympify('c')) * (sympy.prod(exp_long) + sympy.sympify('c')))
                    else:
                        exp1 = avoid_operations_between_constants(sympy.sympify('c') * exp1 * exp2)
                else:
                    # If there's a constant in exp_short, make sure it's the last listed argument
                    if any(const1):
                        temp = exp_short.copy()
                        temp[const1.index(True)] = exp_short[-1]
                        temp[-1] = exp_short[const1.index(True)]
                        exp_short = temp
                    flag = False
                    for i, xp in enumerate(exp_short):
                        if (i == len(exp_short) - 1) and 'cm_' in str(xp) and len(exp_long) > 0:
                            # If there are elements from exp_long that have not been merged, merge them with the constant term
                            if not inside_sum_with_const or \
                                    (len(exp_long) == 1 and exp_long[0].is_symbol and 'c' in str(exp_long[0])):
                                exp_short[i] = avoid_operations_between_constants(exp_short[i] * sympy.prod(exp_long))
                            else:
                                flag = True
                                exp1 = avoid_operations_between_constants(
                                    exp_short[i] * (sympy.prod(exp_short[0:i]) + sympy.sympify('c')) *
                                    sympy.prod([ex + sympy.sympify('c') for ex in exp_long if not (ex.is_symbol and 'c' in str(ex))]))
                        else:
                            # For each argument in exp_short, find compatible args in exp_long and choose one for merging
                            compatible_args = [op for op in exp_long if op.func == xp.func]
                            for op in compatible_args:
                                if isinstance(op, sympy.Pow):
                                    if abs(xp.args[1]) != abs(op.args[1]):
                                        compatible_args.remove(op)
                            if len(compatible_args) == 0:
                                continue
                            selected_arg = random.choice(compatible_args)
                            if random.random() < 0.5:
                                continue
                            exp_long.remove(selected_arg)
                            exp_short[i] = self.merge_base(exp_short[i], selected_arg)
                    if not flag:
                        exp1 = sympy.prod(exp_short)
            elif isinstance(exp1, sympy.Add) or isinstance(exp2, sympy.Add):  # It's implicit that only one is a sum
                exp1 = exp1 * exp2
        try:
            return add_constant_identifier(exp1)[0]
        except TypeError:
            return None

    def choose_combination(self, population=5000, response=None, verbose=False, all_var=False):
        """ Generate multiple combinations and choose the one that adapts better to the given response
        :param population: Population size used for the evolutionary algorithm
        :param response: Response data used for error minimization
        :param verbose: If True, the evolution process is print; otherwise, it's hidden
        :param all_var: If True, it indicates that all the system variables are about to be merged
        """
        combinations = []
        patience = 500
        limit_reached = False
        from tqdm import trange
        for _ in trange(population):
            comb = self.merge()
            if comb is None:
                continue
            count = 0
            comb = add_constant_identifier(avoid_operations_between_constants(comb))[0]
            while comb in combinations:
                # Generate combinations that haven't been generated so far
                comb = self.merge()
                comb = add_constant_identifier(avoid_operations_between_constants(comb))[0]
                count += 1
                if count == patience:
                    # If after "patience" generations, nothing new has been generated, assume all combinations have been explored
                    limit_reached = True
                    break
            if limit_reached:
                break
            # Make sure that this combination includes all variables that are being studied
            if self.merging_variables == len([sy for sy in comb.free_symbols if 'x' in str(sy)]):
                combinations.append(comb)

        if len(combinations) == 0:
            return None

        evolver = GP(X=response[0], Y=response[1], init_population=combinations, all_var=all_var,
                     max_generations=250, p_crossover=0.1, p_mutate=0.9, verbose=verbose, univ_sks=self.univ_sks)
        try:
            merged, corr_val, est_expr = evolver.evolve()
        except:
            return None

        print(est_expr)

        # Check if there's a coefficient in the skeleton that is too small to be considered
        # print(est_expr)
        # if isinstance(est_expr, sympy.Add):
        #     new_st_expr = 0
        #     for arg in est_expr.args:
        #         # Evaluate current arg
        #         fs1 = sympy.lambdify(sympy.flatten(evolver.symbols_list), arg)
        #         ys1 = fs1(*list(response[0].T))
        #         # Evaluate current arg
        #         other_f = 0
        #         for other_args in est_expr.args:
        #             if other_args != arg:
        #                 other_f += other_args
        #         fs2 = sympy.lambdify(sympy.flatten(evolver.symbols_list), other_f)
        #         ys2 = fs2(*list(response[0].T))
        #         # Compare the valuation of current arg against that of the others
        #         ratio = np.divide(ys1, ys2)
        #         print(np.mean(np.abs(ratio)))
        #         if np.mean(np.abs(ratio)) >= 0.01:
        #             new_st_expr += arg
        #     est_expr = new_st_expr
        #     merged = expr2skeleton(est_expr)

        if isinstance(corr_val, float):
            return merged, corr_val, est_expr
        else:
            return None


if __name__ == '__main__':
    random.seed(1)
    # skl1 = sympy.sympify('c*x1^2 + c*x1 + c*sin(c*x1 + c) + c')
    # skl2 = sympy.sympify('c*x2^2 + c*sin(c*x2 + c) + c')
    # y_orig = sympy.sympify('10*x1^2 - x1*x2^2 + 10*sin(x1 + 2*x2 + 2) + 5*x2^2')
    # x1_sampled = np.linspace(-5, 5, 500)
    # x2_sampled = np.linspace(-5, 5, 500)
    # np.random.shuffle(x1_sampled)
    # np.random.shuffle(x2_sampled)
    # samples = np.column_stack((x1_sampled, x2_sampled))
    # fs_lambda = sympy.lambdify(sympy.flatten(y_orig.free_symbols), y_orig)
    # t_response = fs_lambda(*samples.T)

    # skl1 = sympy.sympify('c*x1 + c*tan(c*x1 + c)*sin(c*x1^2 + c) + c*tan(c*x1 + c)*sin(c*x1 + c) + c')
    # skl2 = sympy.sympify('c*x2^2 + c*tan(c*x2 + c)*sin(c*x2 + c) + c*tan(c*x2^2 + c)*sin(c*x2 + c) + c')

    # skl1 = sympy.sympify('c*tan(c*x1)*sin(c*x1^2)*sin(c*log(x1) + c*sqrt(x1))')
    # skl2 = sympy.sympify('c*tan(c*x2)*sin(c*x2)')

    skl1 = sympy.sympify('c*sin(c*x0) + c')
    skl2 = sympy.sympify('c*sin(c*exp(c*x1)) + c')

    merger = MergeExpressions(skl1, skl2, 2)
    [print(merger.merge()) for _ in range(100)]
    # merger.choose_combination(response=[samples, t_response])
