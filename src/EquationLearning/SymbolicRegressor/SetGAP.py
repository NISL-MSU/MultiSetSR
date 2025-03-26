from EquationLearning.SymbolicRegressor.MSSP import *
from EquationLearning.Merge.MergeExpressions import MergeExpressions
from EquationLearning.Optimization.GP import simplify


class SetGAP:

    def __init__(self, dataset: InputData, bb_model, n_candidates=None):
        """SetGAP: Symbolic Regression using Transformers, genetic Algorithms, and genetic Programming
        :param dataset: An InputData object
        :param bb_model: Black-box prediction model (e.g., a feedforward neural-network) that we want to distill into a multivariate expression
        :param n_candidates: Number of candidate skeletons that are generated
        """
        # Define problem
        import torch
        self.X, self.Y, self.var_names, self.types = dataset.X, dataset.Y, dataset.names, dataset.types
        self.target_function = dataset.expr
        if self.target_function != '':
            self.f_lambdified = sp.lambdify(sp.utilities.iterables.flatten(sp.sympify(dataset.names)), dataset.expr)
        self.limits, self.n_features = dataset.limits, dataset.n_features
        self.symbols = sp.symbols("{}:{}".format('x', self.n_features))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bb_model = bb_model
        self.n_candidates = n_candidates
        self.root = get_project_root()
        self.MSSP = MSSP(dataset=dataset, bb_model=bb_model, n_candidates=n_candidates, mult_outputs=True)
        self.n_samples = int(self.MSSP.n_samples / 2)
        self.skeletons = []

    def sample(self, x_inds: list):
        """Sample variable values, allowing variables in x_inds to vary while fixing the rest"""
        R2s, XXs, YYs, valuess = [], [], [], []
        x_inds = sorted(x_inds)
        for it in range(25):
            # Sample random values for all the variables
            values = np.zeros((len(self.symbols)))
            for isy in range(len(self.symbols)):
                if self.types[isy] == 'continuous':
                    rang = (self.limits[isy][1] - self.limits[isy][0])
                    values[isy] = np.random.uniform(self.limits[isy][0] + rang * .05,
                                                    self.limits[isy][1] - rang * .05)
                else:
                    range_values = np.linspace(self.limits[isy][0], self.limits[isy][1], 100)
                    values[isy] = np.random.choice(range_values)
            values = np.repeat(values[:, None], self.n_samples, axis=1)
            for iv in x_inds:
                # Sample values of the variable that is being analyzed
                sample = np.random.uniform(self.limits[iv][0], self.limits[iv][1], self.n_samples)
                values[iv, :] = sample
            # Estimate the response of the generated set
            Y = np.array(self.bb_model.evaluateFold(values.T, batch_size=values.shape[1]))[:, 0]
            X = values[x_inds, :]
            r2 = test_linearity(X.T, Y)
            R2s.append(r2)
            XXs.append(X.copy())
            YYs.append(Y.copy())
            valuess.append(values.copy())
        sorted_indices = np.argsort(np.array(R2s))
        ind = sorted_indices[8]
        return XXs[ind].T, YYs[ind], valuess[ind]

    def run(self):
        # Execute univariate skeleton prediction using MSSP and sort variables
        skeletons, corr_vals, fitted_exprs = self.MSSP.get_skeletons()

        self.skeletons = [x for _, x in sorted(zip(corr_vals, skeletons), reverse=True)]
        sorted_symbols = [v for _, v in sorted(zip(corr_vals, self.symbols), reverse=True)]

        # Start merging skeletons progressively
        merged_skeletons, merged_programs, new_corr_vals = self.skeletons[0], fitted_exprs, corr_vals.copy()
        changing_variables = [sorted_symbols[0]]
        for i in range(1, len(sorted_symbols)):
            print('\n******************************')
            print('Merging skeletons of variables ', str(changing_variables), ', and ', str(sorted_symbols[i]))
            print('******************************')
            # Generate samples fixing values of the other variables
            changing_variables.append(sorted_symbols[i])
            changing_variables_inds = [np.where(np.array(self.symbols) == xv)[0][0] for xv in
                                       np.array(changing_variables)]
            all_var = False
            if set(changing_variables) == set(self.symbols):
                samples, t_response = self.X, self.Y
                all_var = True
            else:
                samples, t_response, _ = self.sample(changing_variables_inds)
            # Merge each of the skeletons in merged_skeletons with each candidate
            new_merged, new_programs, new_corr_vals, count = [], [], [], 0

            for merged_skeleton in merged_skeletons:
                for s in range(len(self.skeletons[i])):
                    merger = MergeExpressions(merged_skeleton + sp.sympify('c'), self.skeletons[i][s],
                                              len(changing_variables))
                    result_comb = merger.choose_combination(response=[samples, t_response], verbose=False,
                                                            all_var=all_var)
                    count += 1
                    if result_comb is not None:
                        merged, corr_val, program = result_comb
                        new_merged.append(merged)
                        new_programs.append(program)
                        new_corr_vals.append(corr_val)
                        print("Generated skeleton ", str(count), '/',
                              str(len(merged_skeletons) * len(self.skeletons[i])),
                              ":\t ", str(merged), "\t Fitness = " + str(np.round(corr_val, 6)))
                    else:
                        print("These skeletons didn't yield a good combination")
            # Take the best self.n_candidates
            merged_skeletons = [x for cv, x in sorted(zip(new_corr_vals, new_merged), reverse=True) if cv <= 1]
            merged_programs = [x for cv, x in sorted(zip(new_corr_vals, new_programs), reverse=True) if cv <= 1]
            if not all_var:
                merged_skeletons = merged_skeletons[:self.n_candidates]
                merged_programs = merged_programs[:self.n_candidates]

        # Fit final coefficients
        print("\nFitting final coefficients")
        est_exprs = []
        for i, program in enumerate(merged_programs):
            if new_corr_vals[i] > 0.999:
                new_var = sp.sympify('x')
                fs_lambda = sp.lambdify(sp.flatten(self.symbols), program)
                int_x = fs_lambda(*list(self.X.T))  # Substitute here to avoid evaluating it repeatedly during the evolution
                max_it, pop_size = 500, 300
            else:
                new_var, int_x = program, self.X
                max_it, pop_size = 600, 300
            candidate = sp.sympify('cm_1') * new_var + sp.sympify('ca_1')
            problem = FitGA(candidate, int_x, self.Y, [np.min(self.X), np.max(self.X)], [-20, 20], max_it=max_it,
                            loss_MSE=True, pop_size=pop_size)
            est_expr, MSE, _ = problem.run()
            est_expr = est_expr.subs({str(new_var): str(program)})
            est_expr = sp.sympify(str((simplify(est_expr.simplify(), all_var=True)[0]).simplify()))
            est_exprs.append(est_expr)
            print('\n******************************')
            print(f'Final estimated expression {i+1}/{len(merged_programs)}: ', str(est_expr), '. MSE = ', MSE)
            print('******************************')

        return est_exprs, merged_skeletons


if __name__ == '__main__':
    ###########################################
    # Import data
    ###########################################
    datasetName = 'E4'
    data_loader = DataLoader(name=datasetName)
    data = data_loader.dataset

    ###########################################
    # Define NN and load weights
    ###########################################
    import torch
    from EquationLearning.models.NNModel import NNModel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder = os.path.join(get_project_root(), "EquationLearning//saved_models//saved_NNs//" + datasetName)
    filepath = folder + "//weights-NN-" + datasetName
    nn_model = None
    if os.path.exists(filepath.replace("weights", "NNModel") + '.pth'):
        # If this file exists, it means we saved the whole model
        network = torch.load(filepath.replace("weights", "NNModel") + '.pth', map_location=device)
        nn_model = NNModel(device=device, n_features=data.n_features, loaded_NN=network)
    elif os.path.exists(filepath):
        # If this file exists, initiate a model and load the weigths
        nn_model = NNModel(device=device, n_features=data.n_features, NNtype=data_loader.modelType)
        nn_model.loadModel(filepath)
    else:
        # If neither files exist, we haven't trained a NN for this problem yet
        if data.n_features > 1:
            sys.exit("We haven't trained a NN for this problem yet. Use the TrainNNModel.py file first.")

    ###########################################
    # Get skeletons
    ###########################################
    regressor = SetGAP(dataset=data, bb_model=nn_model, n_candidates=4)
    results = regressor.run()
