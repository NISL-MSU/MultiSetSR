import omegaconf
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import tukey_hsd
from EquationLearning.utils import get_project_root, tukeyLetters
from EquationLearning.Data.dclasses import SimpleEquation
from EquationLearning.Data.GenerateDatasets import DataLoader
from EquationLearning.Optimization.CoefficientFitting import FitGA
from EquationLearning.models.utilities_expressions import add_constant_identifier, get_skeletons
from EquationLearning.Transformers.GenerateTransformerData import Dataset, evaluate_and_wrap

np.random.seed(7)

####################################
# Parameters
####################################
name = 'E1'
clim = [-20, 20]
cfg = omegaconf.OmegaConf.load("./EquationLearning/Transformers/config.yaml")
training_dataset = Dataset(cfg.train_path, cfg.dataset_train, mode="train")
word2id = training_dataset.word2id
scratch = True  # If False, just load saved results and plot them

# Methods
methods = ['PYSR', 'TaylorGP', 'NESYMRES', 'E2E', 'MST']

####################################
# Load underlying equation
####################################
dataLoader = DataLoader(name=name, extrapolation=False).dataset
X, Y, var_names, expr = dataLoader.X, dataLoader.Y, dataLoader.names, dataLoader.expr
print("Underlying function: " + str(expr))

####################################
# Analyze one variable at a time
####################################
original_skeletons = get_skeletons(expr, var_names)
for iv, var in enumerate(var_names):
    if scratch:
        # The evaluation domain is TWICE the one used during training
        limits = [dataLoader.limits[iv][0]*2, dataLoader.limits[iv][1]*2]
        # Get skeleton for each variable present in the expression
        skeleton = original_skeletons[iv]
        print('Analyzing variable ' + var + '. Skeleton: ')
        print('\t' + str(skeleton))

        # Sample coefficient values for the given skeleton
        coeff_dict = dict()
        for constant in skeleton.free_symbols:
            if 'c' in str(constant):
                coeff_dict[str(constant)] = constant

        eq = SimpleEquation(expr=skeleton, coeff_dict=coeff_dict, variables=[var])
        result = evaluate_and_wrap(eq, cfg.dataset_train, word2id, return_exprs=True, n_sets=30,
                                   xmin=limits[0], xmax=limits[1])
        Xs, Ys, _, _, exprs = result

        # Analyze each of the 10 sampled expressions
        errors = np.zeros((Xs.shape[1], len(methods)))
        for it in range(Xs.shape[1]):
            Xi, Yi = Xs[:, it], Ys[:, it]
            print('\tIteration ' + str(it))
            print("\t|\tSampled expression: " + str(exprs[it]).replace('x_1', var))
            est_skeletons, est_exprs = [], []
            for im, method in enumerate(methods):
                # Load the expressions generated by each method
                path = str(get_project_root()) + "/output/LearnedEquations/" + name + "/" + method + ".txt"
                with open(path, "r") as myfile:
                    sk = myfile.read().splitlines()
                if method != 'MST':
                    # Get the skeleton for the current variable
                    est_skeletonG = get_skeletons(expr=sp.sympify(sk[0]), var_names=var_names)
                    est_skeleton = est_skeletonG[iv]
                else:
                    # In the case of the Multi-Set Transformer, it directly saves the skeletons
                    est_skeleton = get_skeletons(expr=sp.sympify(sk[iv]), var_names=var_names)[iv]
                    est_skeleton = add_constant_identifier(est_skeleton)[0]

                print("\t|\t\tUsing skeleton obtained by the " + method + " method: " + str(est_skeleton))
                if est_skeleton in est_skeletons:
                    pos = est_skeletons.index(est_skeleton)
                    error = errors[it, pos]
                    errors[it, im] = error
                    est_expr = est_exprs[pos]
                else:
                    # Fit coefficients of the estimated skeletons
                    problem = FitGA(est_skeleton, Xi, Yi, limits, clim)
                    est_expr, error = problem.run()
                    errors[it, im] = error
                est_skeletons.append(est_skeleton)
                est_exprs.append(est_expr)
                print("\t|\t\t\tFitted expression: " + str(est_expr) + ". Error = " + str(np.round(error, 8)))

        # Save results
        np.save(str(get_project_root()) + '/output/metrics/comparison_var-' + var + '_problem-' + name + '.npy',
                errors)

    else:  # Load results
        errors = np.load(str(get_project_root()) + '/output/metrics/comparison_var-' + var + '_problem-' + name +
                         '.npy', allow_pickle=True)

    # Create box plots for each method
    plt.figure()
    plt.boxplot(errors, labels=methods)
    plt.xlabel('Methods')
    plt.ylabel('Errors')
    plt.tight_layout()
    plt.savefig(str(get_project_root()) + '/output/metrics/comparison_var-' + var + '_problem-' + name + '.jpg',
                dpi=600)

    # Perform Tukey Test
    columns = [errors[:, i] for i in range(errors.shape[1])]
    res = tukey_hsd(*columns)
    print("Tukey's HSD Pairwise Group Comparisons (Grouping)")
    print("___________________")
    print(methods)
    lett = tukeyLetters(res.pvalue)
    print(lett)

    means = np.mean(errors, axis=0)
    stds = np.std(errors, axis=0)

    # Format the results in LaTeX format
    latex_output = ""
    for i in range(len(means)):
        if means[i] < 0.1:
            mean_str = "{:.0e}".format(means[i]).replace('e', '\!\\times\! 10^{') + '}'
        else:
            means_rounded = np.round(means, decimals=1)
            mean_str = "{}".format(means_rounded[i])

        if stds[i] < 0.1:
            std_str = "{:.0e}".format(stds[i]).replace('e', '\!\\times\! 10^{') + '}'
        else:
            stds_rounded = np.round(stds, decimals=1)
            std_str = "{}".format(stds_rounded[i])

        latex_output += "${} \pm {}$".format(mean_str, std_str)

        if i < len(means) - 1:
            latex_output += " & "

    print(latex_output)
