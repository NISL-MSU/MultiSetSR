import sys
import omegaconf
import sympy as sp
from EquationLearning.utils import *
from EquationLearning.Data.math_rules import sk_equivalence
from EquationLearning.Optimization.CoefficientFitting import FitGA
from EquationLearning.Data.GenerateDatasets import DataLoader, InputData
from EquationLearning.Transformers.GenerateTransformerData import Dataset
from EquationLearning.models.utilities_expressions import expr2skeleton, avoid_operations_between_constants, \
                                                              count_nodes, remove_coeffs


class MSSP:

    def __init__(self, dataset: InputData, bb_model, n_candidates=None, mult_outputs=False):
        """Distills univariate symbolic skeleton expressions given an experimental dataset
        :param dataset: An InputData object
        :param bb_model: Black-box prediction model that was trained to capture the association between inputs and
                         outputs of the system (e.g., a feedforward neural-network)
        :param n_candidates: Number of candidate skeletons that are generated
        :param mult_outputs: If True, return multiple output candidates
        """
        # Define problem
        import torch  # NOTE: Imports of "torch" are done within the functions to avoid problems with multiprocessing
        from EquationLearning.Transformers.model import Model
        self.X, self.Y, self.var_names, self.types = dataset.X, dataset.Y, dataset.names, dataset.types
        self.target_function = dataset.expr
        if self.target_function != '':
            self.f_lambdified = sp.lambdify(sp.utilities.iterables.flatten(sp.sympify(dataset.names)), dataset.expr)
        self.limits, self.n_features = dataset.limits, dataset.n_features
        self.symbols = sp.symbols("{}:{}".format('x', self.n_features))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bb_model = bb_model
        self.root = get_project_root()
        self.mult_outputs = mult_outputs

        # Read config yaml
        try:
            self.cfg = omegaconf.OmegaConf.load(os.path.join(self.root, "EquationLearning/Transformers/config.yaml"))
        except FileNotFoundError:
            self.cfg = omegaconf.OmegaConf.load("../Transformers/config.yaml")

        # Initialize MST
        self.data_train_path = self.cfg.train_path
        self.training_dataset = Dataset(self.data_train_path, self.cfg.dataset_train, mode="train")
        self.word2id = self.training_dataset.word2id
        self.id2word = self.training_dataset.id2word
        self.model = Model(cfg=self.cfg.architecture, cfg_inference=self.cfg.inference, word2id=self.word2id)

        # Load models
        self._load_models()

        # Initialize class variables
        self.univariate_skeletons = []
        self.merged_expressions = []
        self.n_sets = self.cfg.architecture.number_of_sets
        self.n_samples = self.cfg.architecture.block_size

        if n_candidates is None:
            self.n_candidates = self.cfg.inference.beam_size
        else:
            self.n_candidates = n_candidates

    def _load_models(self):
        import torch
        # Load weights of MST
        MST_path = os.path.join(self.root, "EquationLearning//saved_models//saved_MSTs/Model480-batch_20-Q1")
        self.model.load_state_dict(torch.load(MST_path, map_location=self.device))
        if torch.cuda.is_available():
            self.model.cuda()

    def sample_sets(self, iv=0):
        Rs = np.zeros(self.n_sets)
        # Generate multiple sets of data where only the current variable is allowed to vary
        if len(self.symbols) == 1:
            if len(self.X) > self.n_samples:
                ra = np.arange(0, len(self.X))
                np.random.shuffle(ra)
                ra = ra[0:self.n_samples]
                self.X, self.Y = self.X[ra, :], self.Y[ra]
            Xs = np.repeat(self.X, self.n_sets, axis=1)
            Ys = np.repeat(self.Y[:, None], self.n_sets, axis=1)
            Ys_real = Ys
        else:
            Xs = np.zeros((self.n_samples, self.n_sets))
            Ys = np.zeros((self.n_samples, self.n_sets))
            Ys_real = np.zeros((self.n_samples, self.n_sets))

            for ns in range(self.n_sets):
                # Repeat the sampling process a few times and keep the one the looks more different from a line
                R2s, XXs, YYs, valuess = [], [], [], []
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
                    # Sample values of the variable that is being analyzed
                    sample = np.random.uniform(self.limits[iv][0], self.limits[iv][1], self.n_samples)
                    values[iv, :] = sample
                    # Estimate the response of the generated set
                    Y = np.array(self.bb_model.evaluateFold(values.T, batch_size=values.shape[1]))[:, 0]
                    X = sample
                    r2 = test_linearity(X[:, None], Y)
                    if np.std(Y) < 0.3:
                        R2s.append(0)
                    else:
                        R2s.append(r2)
                    XXs.append(X.copy())
                    YYs.append(Y.copy())
                    valuess.append(values.copy())
                sorted_indices = np.argsort(np.array(R2s))
                ind = sorted_indices[7]
                best_X, best_Y, best_values = XXs[ind], YYs[ind], valuess[ind]
                Ys[:, ns] = best_Y
                Xs[:, ns] = best_X
                if self.target_function != '':
                    Ys_real[:, ns] = np.array(self.f_lambdified(*list(best_values)))
                Rs[ns] = R2s[ind]
        # Normalize data
        sorted_indices = np.argsort(np.array(Rs))
        ind = sorted_indices[2]
        Xi, Yi = Xs[:, ind].copy(), Ys[:, ind].copy()
        means, std = np.mean(Ys, axis=0), np.std(Ys, axis=0)
        Ys = (Ys - means) / std
        means, std = np.mean(Ys_real, axis=0), np.std(Ys_real, axis=0)
        Ys_real = (Ys_real - means) / std

        return Xs, Ys, Ys_real, Xi, Yi

    def pre_process_data(self, iv):
        import torch
        Xs, Ys, Ys_real, Xi, Yi = self.sample_sets(iv=iv)
        # Format the data as inputs to the Multi-set transformer
        scaling_factor = 20 / (np.max(Xs) - np.min(Xs))
        Xs = (Xs - np.min(Xs)) * scaling_factor - 10
        XY_block = torch.zeros((1, self.n_samples, 2, self.n_sets)).to(self.device)
        Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
        Xs = Xs.to(self.device)
        Ys = Ys.to(self.device)
        XY_block[0, :, 0, :] = Xs
        XY_block[0, :, 1, :] = Ys
        return XY_block, Xi, Yi

    def get_skeletons(self):
        from EquationLearning.Trainer.TrainMultiSetTransformer import seq2equation
        """Retrieve the estimated symbolic equation"""
        corr_vals, univ_sorted_exprs = [], []
        # Analyze each variable and obtain univariate expressions
        for iv, va in enumerate(self.symbols):
            np.random.seed(7)
            print("********************************")
            print("Analyzing variable " + str(va))
            print("********************************")
            pred_skeletons = []
            Xi, Yi = None, None
            reps = self.n_candidates
            for ii in range(reps):
                XY_block, Xi, Yi = self.pre_process_data(iv=iv)

                # Perform Multi-Set Skeleton Prediction
                cand = 2
                preds = self.model.inference(XY_block, cand)
                for ip, pred in enumerate(preds):
                    try:
                        tokenized = list(pred[1].cpu().numpy())[1:]
                        skeleton = seq2equation(tokenized, self.id2word, printFlag=False)
                        skeleton = sp.sympify(skeleton.replace('x_1', str(va)))
                        skeleton = sk_equivalence(avoid_operations_between_constants(sp.expand(sp.sympify('c') * skeleton)))
                        if skeleton in pred_skeletons:
                            ct = 0
                            while skeleton in pred_skeletons:
                                XY_block, Xi, Yi = self.pre_process_data(iv=iv)
                                preds = self.model.inference(XY_block, cand)
                                for predd in preds:
                                    tokenized = list(predd[1].cpu().numpy())[1:]
                                    skeleton = seq2equation(tokenized, self.id2word, printFlag=False)
                                    skeleton = sp.sympify(skeleton.replace('x_1', str(va)))
                                    skeleton = sk_equivalence(avoid_operations_between_constants(sp.expand(sp.sympify('c') * skeleton)))
                                    if skeleton not in pred_skeletons:
                                        pred_skeletons.append(skeleton)
                                        print('Predicted skeleton ' + str(len(pred_skeletons)) + ' for variable ' + str(va) + ': ' + str(skeleton))
                                        break
                                ct += 1
                                if ct == 3:
                                    break
                            if ct == 3:
                                continue
                        else:
                            pred_skeletons.append(skeleton)
                            print('Predicted skeleton ' + str(len(pred_skeletons)) + ' for variable ' + str(va) + ': ' + str(skeleton))
                    except:  # TypeError:
                        print("Invalid response created by the model")

            print("\nChoosing the best skeleton... (skeletons ordered based on number of nodes)")
            best_corr, best_sk = 0, ''
            pred_skeletons = sorted(pred_skeletons, key=lambda expr: count_nodes(expr))
            nodes_num = [count_nodes(expr) for expr in pred_skeletons]
            tested_skeletons, tested_skeletons_orig, corr_ind_vals, fitted_exprs = [], [], [], []
            for ip, skeleton in enumerate(pred_skeletons):
                # Fit coefficients of the estimated skeletons
                if skeleton not in tested_skeletons_orig:
                    try:
                        problem = FitGA(remove_coeffs(skeleton), Xi, Yi, [np.min(Xi), np.max(Xi)], [-20, 20], max_it=None,
                                        loss_MSE=False)
                        est_expr, corr, _ = problem.run()
                    except:
                        continue

                    # If the expression is a sum of arguments check that none of them is too small
                    skeleton_orig = skeleton
                    try:
                        if isinstance(est_expr, sp.Add):
                            new_st_expr = 0
                            for arg in est_expr.args:
                                # Evaluate current arg
                                fs1 = sp.lambdify(va, arg)
                                ys1 = fs1(Xi)
                                # Evaluate current arg
                                other_f = 0
                                for other_args in est_expr.args:
                                    if other_args != arg:
                                        other_f += other_args
                                fs2 = sp.lambdify(va, other_f)
                                ys2 = fs2(Xi)
                                # Compare the valuation of current arg against that of the others
                                ratio = np.divide(ys1, ys2)
                                if np.mean(np.abs(ratio)) >= 0.01:
                                    new_st_expr += arg
                            est_expr = new_st_expr
                        skeleton = expr2skeleton(est_expr)
                    except AttributeError:
                        skeleton = skeleton
                    try:
                        skeleton = sk_equivalence(skeleton)
                    except IndexError:
                        continue

                    corr = (abs(corr) * 1000 - nodes_num[ip]) / 1000
                    if 0.7 < abs(corr) <= 1 and abs(corr) not in corr_ind_vals:  # Avoid including skeletons that obviously don't fit
                        tested_skeletons_orig.append(skeleton_orig)
                        tested_skeletons.append(skeleton)
                        corr_ind_vals.append(corr)
                        fitted_exprs.append(est_expr)
                    print("\tSkeleton: " + str(skeleton) + ". Correlation: " + str(abs(corr)) + ". Expr: " + str(est_expr) + ". Perf: " + str((abs(corr) * 1000 - nodes_num[ip]) / 1000))
                    if not self.mult_outputs:
                        if abs(best_corr - corr) > 0.002:
                            best_corr = corr
                            best_sk = expr2skeleton(2 * est_expr + 1)
                            if 1 >= abs(corr) > 0.999:  # If correlation is very high, assume this is the best
                                break
            if not self.mult_outputs:
                print('-----------------------------------------------------------')
                print("Selected skeleton: " + str(best_sk) + "\n")
                self.univariate_skeletons.append(best_sk)
                corr_vals.append(abs(best_corr))
            else:
                # Sort skeletons according to their correlation values
                sorted_skeletons = [x for _, x in sorted(zip(corr_ind_vals, tested_skeletons), reverse=True)]
                sorted_expressions = [x for _, x in sorted(zip(corr_ind_vals, fitted_exprs), reverse=True)]
                corr_ind_vals = sorted(corr_ind_vals, reverse=True)
                corr_ind_vals2, sorted_skeletons2, sorted_expressions2 = [], [], []
                n_sk = len(sorted_skeletons)
                if n_sk > self.n_candidates:
                    n_sk = self.n_candidates
                i, fS = 0, True
                while len(sorted_skeletons2) < n_sk and i < len(sorted_skeletons):
                    # if i == 0 and fS:  # Always includes the first skeleton that was generated
                    #     sorted_skeletons2.append(tested_skeletons[0])
                    #     sorted_expressions2.append(fitted_exprs[0])
                    #     corr_ind_vals2.append(corr_ind_vals[0])
                    #     fS = False
                    #     continue
                    if not any([str(sorted_skeletons[i]) == str(others) for others in sorted_skeletons2]):
                        sorted_skeletons2.append(sorted_skeletons[i])
                        sorted_expressions2.append(sorted_expressions[i])
                        corr_ind_vals2.append(corr_ind_vals[i])
                    i += 1
                corr_ind_vals, sorted_skeletons, sorted_expressions = corr_ind_vals2, sorted_skeletons2, sorted_expressions2
                self.univariate_skeletons.append(sorted_skeletons)
                univ_sorted_exprs.append(sorted_expressions)
                corr_vals.append(max(corr_ind_vals))

        return self.univariate_skeletons, corr_vals, univ_sorted_exprs


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    ###########################################
    # Import data
    ###########################################
    datasetName = 'E2'
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
    regressor = MSSP(dataset=data, bb_model=nn_model, n_candidates=5)
    print(regressor.get_skeletons())
