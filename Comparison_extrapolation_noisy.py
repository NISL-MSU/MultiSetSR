import sympy as sp
import numpy as np
from EquationLearning.utils import get_project_root
from EquationLearning.Data.GenerateDatasets import DataLoader

np.random.seed(7)


def evaluate(exp, Xs, Ys, xvars):
    exp = sp.sympify(exp)
    fs_lambda = sp.lambdify(sp.flatten(xvars), exp)
    if Xs.ndim > 1:
        ys = fs_lambda(*list(Xs.T))
    else:
        ys = fs_lambda(Xs)
    return np.mean((Ys - ys) ** 2)


def round_expr(expr):
    return expr.xreplace({n: round(n, 3) for n in expr.atoms(sp.Number)})


# Parameters
names = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13']
noise_levels = [0, 0.01, 0.03, 0.05]
method = "SETGAP"

# Store results
error_results = {name: {} for name in names}
eq_results = {name: {} for name in names}

# Process each problem
for name in names:
    namex = name.replace("E10", "CS1").replace("E11", "CS2").replace("E12", "CS3").replace("E13", "CS4")

    for noise in noise_levels:

        # Load interpolation and extrapolation datasets
        dataLoader_interp = DataLoader(name=namex, extrapolation=False, noise=noise).dataset
        dataLoader_extra = DataLoader(name=namex, extrapolation=True, noise=noise).dataset

        X_interp, Y_interp, var_names, _ = dataLoader_interp.X, dataLoader_interp.Y, dataLoader_interp.names, dataLoader_interp.expr
        X_extra, Y_extra, _, _ = dataLoader_extra.X, dataLoader_extra.Y, dataLoader_extra.names, dataLoader_extra.expr
        noise_suff = f'/Noise_{noise}/' if noise > 0 else '/Without-noise/'

        if noise == 0:
            method = 'MST'
            path = str(
                get_project_root().parent) + "/output/LearnedEquations/" + noise_suff + namex + '/' + method + ".txt"
        else:
            method = "SETGAP"
            path = str(get_project_root().parent) + "/output/LearnedEquations/" + noise_suff + namex + '/' + method

        try:
            with open(path, "r") as myfile:
                expr_pred = myfile.read().splitlines()[0]
                expr_sympy = round_expr(sp.simplify(sp.sympify(expr_pred)))

            interp_error = evaluate(expr_pred, X_interp, Y_interp, var_names)
            extra_error = evaluate(expr_pred, X_extra, Y_extra, var_names)
            error_results[name][noise] = (interp_error, extra_error)
            eq_results[name][noise] = expr_sympy
        except:
            error_results[name][noise] = ("---", "---")
            eq_results[name][noise] = "---"

# Generate LaTeX table for MSE values
latex_table_mse = "\\begin{table}[]\n\\resizebox{\\textwidth}{!}{%\n"
latex_table_mse += "\\begin{tabular}{|c|" + "c|" * (2 * len(noise_levels)) + "}\\hline\n"
latex_table_mse += "\\textbf{Problem} & " + " & ".join(["\\begin{tabular}[c]{@{}c@{}} Interpolation\\\\" + f"($\\sigma={noise}$)" + "\\end{tabular}" for noise in noise_levels]) + " & " + \
               " & ".join(["\\begin{tabular}[c]{@{}c@{}} Extrapolation\\\\" + f"($\\sigma={noise}$)" + "\\end{tabular}" for noise in noise_levels]) + " \\\\ \\hline\n"

for name in names:
    row = f"{name} & " + " & ".join(f"{error_results[name][noise][0]:.3e}" if isinstance(error_results[name][noise][0], float) else f"{error_results[name][noise][0]}" for noise in noise_levels)
    row += " & " + " & ".join(f"{error_results[name][noise][1]:.3e}" if isinstance(error_results[name][noise][1], float) else f"{error_results[name][noise][1]}" for noise in noise_levels) + " \\\\ \\hline\n"
    latex_table_mse += row

latex_table_mse += "\\end{tabular}%\n}\n\\end{table}"

# Generate LaTeX table for learned expressions
latex_table_expr = "\\begin{table}[]\n\\resizebox{\\textwidth}{!}{%\n"
latex_table_expr += "\\begin{tabular}{|c|" + "c|" * len(noise_levels) + "}\\hline\n"
latex_table_expr += "\\textbf{Problem} & " + " & ".join([f"$\sigma={noise}$" for noise in noise_levels[1:]]) + " \\\\ \\hline\n"

for name in names:
    row = f"{name} & " + " & ".join(f"${eq_results[name][noise]}$" if eq_results[name][noise] != "---" else "---" for noise in noise_levels[1:]) + " \\\\ \\hline\n"
    latex_table_expr += row

latex_table_expr += "\end{tabular}%\n}\n\end{table}"

# Print both LaTeX tables
print(latex_table_mse)
print("\n")
print(latex_table_expr)
