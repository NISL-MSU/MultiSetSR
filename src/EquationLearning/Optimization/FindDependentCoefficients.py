import time
import warnings
import numpy as np
from tqdm import trange
from scipy.stats import pearsonr
from src.utils import calc_distance_curves
from src.EquationLearning.models.utilities_expressions import *
from src.EquationLearning.Data.sympy_utils import modify_trig_expr
from src.EquationLearning.Optimization.CoefficientFitting import FitGA
from src.EquationLearning.models.utilities_expressions import add_constant_identifier


warnings.filterwarnings("ignore")


class CheckDependency:

    def __init__(self, skeleton, t, list_vars, gen_func, v_limits, c_limits, types=None):
        """
        Given a univariate symbolic expression, find which coefficients are dependent on other functions
        :param skeleton: Symbolic skeleton generated for the t-th variable
        :param t: Index of the variable whose skeleton is currently being analyzed
        :param list_vars: List of Sympy symbols of the original expression
        :param gen_func: Generative function. It can be an NN model or a sympy function
        :param v_limits: Limits of the values that all variables can take
        :param c_limits: Limits of the values that all expression coefficients can take
        :param types: Types of each variable (i.e., continuous or discrete)
        """
        self.skeleton = skeleton  # The skeleton is a function of variable x_t (the t-th variable in list_vars)
        self.t = t
        self.list_vars = list_vars
        self.gen_func = gen_func
        self.v_limits = v_limits
        if len(v_limits) == 2 and isinstance(v_limits, tuple):
            self.v_limits = [v_limits] * len(list_vars)
        self.c_limits = c_limits
        self.ns = 3
        self.nxs = 1000
        self.types = types
        if types is None:
            self.types = ['continuous'] * len(list_vars)

    def sample_values(self, v):
        """Sample values for the variables that are different from x_t.
        :param v: Index of the variable that will be allowed to vary. The other variables (except x_t will be fixed)
        """
        # Sample nxs values for variable x_t
        if self.types[self.t] == 'continuous':
            xt_values = np.random.uniform(self.v_limits[self.t][0], self.v_limits[self.t][1], size=self.nxs)
        else:
            xt_values = np.linspace(self.v_limits[self.t][0], self.v_limits[self.t][1], self.nxs)

        # Sample values for variables that are different from x_t
        values_temp = np.zeros((len(self.list_vars), self.ns), dtype=object)
        ct, max_dist, values, y = 0, 0, None, None
        while ct < 10:  # The sampling process is repeated multiple times, we'll keep the values with the highest variation
            y_temp = np.zeros((self.nxs, self.ns))
            for i in range(len(self.list_vars)):
                if i == v:  # Variable that is allowed to vary
                    if self.types[self.t] == 'continuous':
                        values_temp[i, :] = np.random.uniform(self.v_limits[i][0], self.v_limits[i][1], size=self.ns)
                    else:
                        range_values = np.linspace(self.v_limits[i][0], self.v_limits[i][1], 100)
                        values_temp[i, :] = np.random.choice(range_values, size=self.ns)
                elif i == self.t:  # Multiple values are sampled
                    for si in range(self.ns):
                        values_temp[i, si] = list(xt_values)
                else:  # Fixed variables
                    if self.types[self.t] == 'continuous':
                        values_temp[i, :] = np.random.uniform(self.v_limits[i][0], self.v_limits[i][1], size=1)
                    else:
                        range_values = np.linspace(self.v_limits[i][0], self.v_limits[i][1], 100)
                        values_temp[i, :] = np.random.choice(range_values, size=1)
            gen_fun = None
            if isinstance(self.gen_func, sympy.Expr):
                gen_fun = sp.lambdify(sympy.flatten(self.list_vars), self.gen_func)
            for ix, xx in enumerate(xt_values):
                vals = values_temp.copy()
                vals[self.t, :] = xt_values[ix]
                if isinstance(self.gen_func, sympy.Expr):
                    y_temp[ix, :] = gen_fun(*list(vals.astype(np.float)))
                else:
                    y_temp[ix, :] = np.array(self.gen_func.evaluateFold(vals.T.astype(np.float16), batch_size=vals.shape[1]))[:, 0]
            # Avoid generating vectors y_temp that represent a constant value
            if np.any([np.std(y_temp[:, i]) <= 0.1 for i in range(self.ns)]):
                continue
            dist = calc_distance_curves(list(y_temp.T))
            # Keep the values that led to the highest distance between functions
            if dist > max_dist:
                y = y_temp.copy()
                values = values_temp.copy()
                max_dist = dist
            ct += 1
        return xt_values, values, y

    def run(self):
        # Analyze the dependency of coefficients skeleton(x_t) on each variable x_v != x_t
        depend_vars = [i for i in range(len(self.list_vars)) if i != self.t]
        skeleton, _, _ = add_constant_identifier(self.skeleton)
        original_coeff = get_args(skeleton).copy()
        final_coeff = original_coeff.copy()
        for v in depend_vars[0:]:
            print("Checking which coefficients depend on variable " + '\033[1m' + str(self.list_vars[v]) + '\033[0m')
            # Sample values
            xt_values, values, y = self.sample_values(v)

            # Check if the results are correlated, if they are, stop
            if not check_correlation(y, epsi=0.99):
                continue
            else:
                skeleton = remove_coeffs(skeleton)

            # Fit coefficients
            est_exprs = []
            print("\tFitting expression...")
            time.sleep(0.5)
            for r in trange(self.ns):
                # Optimize correlation, not error
                problem = FitGA(skeleton, xt_values, y[:, r], [np.min(xt_values), np.max(xt_values)],
                                self.c_limits, max_it=100, loss_MSE=True)
                est_expr, _ = problem.run()
                est_exprs.append(modify_trig_expr(est_expr))
            print('\t', est_exprs)

            if not (isinstance(skeleton, sp.Add) or isinstance(skeleton, sp.Mul)):
                est_exprs = [xp + 1 for xp in est_exprs]
                skeleton = skeleton + sp.sympify('ca_0')
            dependent_coeffs = find_dependent_coeffs(xt_values, skeleton, est_exprs)

            # Analyze the tree and find if there are subtrees that differ (i.e., that produce different results)
            for i, co in enumerate(original_coeff):
                new_var = sp.sympify(str(self.list_vars[v]).replace('x', 'f'))
                if co in dependent_coeffs:
                    if isinstance(final_coeff[i], sympy.Symbol):
                        if final_coeff[i] == co:
                            final_coeff[i] = new_var
                        else:
                            final_coeff[i] = [final_coeff[i], new_var]
                    elif isinstance(final_coeff[i], list):
                        (final_coeff[i]).append(new_var)
        for cf in range(len(final_coeff)):
            if isinstance(final_coeff[cf], list):
                final_coeff[cf] = sp.Symbol(str(final_coeff[cf]))

        # Apply modifications to final expression
        new_skeleton = set_args(self.skeleton, final_coeff)
        print('-----------------------------------------------------------')
        print('The resulting skeleton with dependencies is as follows: ' + str(new_skeleton) + '\n')
        return new_skeleton, final_coeff


def find_dependent_coeffs(x_values, skeleton, exprs, coeffs=None):
    """
    Analyze the expression tree and find if there are subtrees that differ (i.e., that produce different results).
    If True, find which coefficients are causing these differences
    """
    if coeffs is None:
        coeffs = []
    for arg_id in range(len(exprs[0].args)):
        # Evaluate each sub-expression
        ys = []
        coef = None
        for i, expr in enumerate(exprs):
            if 'x' not in str(expr.args[arg_id].free_symbols):
                ys.append(np.repeat(float(expr.args[arg_id]), len(x_values), axis=0))
                if i == 0:
                    # Match this number with a coefficient of the skeleton at the same tree level
                    if isinstance(expr, sympy.Mul):  # If this coeff. is inside a mult, it's a mult coeff.
                        for cf in skeleton.args:
                            if 'cm' in str(cf):
                                coef = cf
                                break
                    else:  # If this coefficient is inside a sum, it's an additive coeff.
                        for cf in skeleton.args:
                            if 'ca' in str(cf):
                                coef = cf
                                break
            else:
                fs_lambda = sp.lambdify(sympy.flatten(expr.args[arg_id].free_symbols), expr.args[arg_id])
                ys.append(fs_lambda(x_values))
        # Check if there's at least one column that differs from the rest
        go_deeper = False  # check_difference(np.array(ys).T)
        if check_correlation(np.array(ys).T):
            go_deeper = True
        elif check_difference(np.array(ys).T):
            go_deeper = True
        if go_deeper:
            if coef is not None:  # If this subtree is a coefficient leaf of the skeleton
                coeffs.append(coef)
            else:  # If there's a difference, jump to a deeper level of the tree to find the coeff, that is causing it
                next_exprs = [expr.args[arg_id] for expr in exprs]
                # Find what's the subtree of 'skeleton' that matches the skeleton of the analyzed expressions
                tgt_sk = expr2skeleton(next_exprs[0])
                skeleton_noid = remove_constant_identifier(skeleton)
                next_skeleton = None
                for noid_arg, id_arg in zip(skeleton_noid.args, skeleton.args):
                    if str(noid_arg) == str(tgt_sk):
                        next_skeleton = id_arg
                        break
                coeffs = find_dependent_coeffs(x_values, next_skeleton, next_exprs, coeffs=coeffs)
    return coeffs


def check_difference(matrix, epsi=0.08):
    num_columns = matrix.shape[1]
    for i in range(num_columns):
        for j in range(i+1, num_columns):
            if np.min(np.abs((matrix[:, i] - matrix[:, j]) / matrix[:, i])) >= epsi:
                return True
    return False


def check_correlation(matrix, epsi=0.98):
    num_columns = matrix.shape[1]
    for i in range(num_columns):
        for j in range(i+1, num_columns):
            corr, _ = pearsonr(matrix[:, i], matrix[:, j])
            if np.abs(corr) < epsi:
                return True
    return False


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure()

    def g_func(symb):
        # return sp.sin(symb[0] + (symb[1]) * symb[2]) + sp.exp(0.2 * symb[3])
        return (sp.Abs(symb[2])) / symb[0] * sp.exp(-1/10 * (symb[1] / symb[0]) ** 2)


    def g_skeleton(symb):
        c = sp.sympify(str('c'))
        # return sp.sin(c + c * (symb[1])) + c
        return c / symb[0] * sp.exp(c / (symb[0] ** 2))


    # Parameters
    symbols = sp.symbols("{}:{}".format('x', 3))
    limits = [-10, 10]
    clim = [-10, 10]
    xt = 0  # Variable that is currently being analyzed

    # Execute program
    print('Initial skeleton: ' + str(g_skeleton(symbols)))
    print('-----------------------------------------------------------')
    check_dependency = CheckDependency(skeleton=g_skeleton(symbols), t=xt, list_vars=symbols, gen_func=g_func(symbols),
                                       v_limits=limits, c_limits=clim)
    new_sk, new_c = check_dependency.run()

    # exprsv = [sympy.sympify('7.6251*exp(-2.5/x0**2)/x0'),
    #           sympy.sympify('9.25025*exp(-0.356682/x0**2)/x0'),
    #           sympy.sympify('9.250221*exp(-2.270248/x0**2)/x0')]

    # skel, _, _ = add_constant_identifier(g_skeleton(symbols))
    # cs = find_dependent_coeffs(np.random.uniform(limits[0], limits[1], size=500), skel, exprsv)
