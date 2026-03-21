import sys
import omegaconf
import sympy as sp
from EquationLearning.utils import *
from EquationLearning.Data.math_rules import sk_equivalence
from EquationLearning.Optimization.CoefficientFitting import FitGA
from EquationLearning.Data.GenerateDatasets import DataLoader, InputData
from EquationLearning.Transformers.GenerateTransformerData import Dataset
from EquationLearning.models.utilities_expressions import expr2skeleton, avoid_operations_between_constants, \
                                                              count_nodes, remove_coeffs, standardize_expression


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
        self.limits, self.n_features, self.unique_vals = dataset.limits, dataset.n_features, dataset.values
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
        if len(self.symbols) == 0:
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
                            # range_values = np.linspace(self.limits[isy][0], self.limits[isy][1], 100)
                            values[isy] = np.random.choice(self.unique_vals[isy])
                    values = np.repeat(values[:, None], self.n_samples, axis=1)
                    # Sample values of the variable that is being analyzed
                    if self.types[iv] == 'continuous':
                        sample = np.random.uniform(self.limits[iv][0], self.limits[iv][1], self.n_samples)
                    else:
                        sample = np.random.choice(np.array(self.unique_vals[iv]), self.n_samples)
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
                ind = sorted_indices[2]  # 7
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
        perf_vals, univ_sorted_exprs = [], []
        # Analyze each variable and obtain univariate expressions
        for iv, va in enumerate(self.symbols):
            if iv >= 7:
                flag = False
                pred_skeletons = []
                # if iv == 0:
                #     XY_block, Xi, Yi = self.pre_process_data(iv=iv)
                #     pred_skeletons.append(sp.sympify('x0'))
                #     flag = True
                # elif iv == 1:
                #     XY_block, Xi, Yi = self.pre_process_data(iv=iv)
                #     pred_skeletons.append(sp.sympify('c*sin(c*x1 + c)'))
                #     flag = True
                # elif iv == 2:
                #     XY_block, Xi, Yi = self.pre_process_data(iv=iv)
                #     pred_skeletons.append(sp.sympify('c*sin(c*x2 + c)'))
                #     flag = True
                # elif iv == 3:
                #     XY_block, Xi, Yi = self.pre_process_data(iv=iv)
                #     pred_skeletons.append(sp.sympify('c/(c + x3)'))
                #     flag = True
                # else:
                np.random.seed(7)
                print("********************************")
                print("Analyzing variable " + str(va))
                print("********************************")
                pred_skeletons = []
                Xi, Yi = None, None
                reps = self.n_candidates
                for ii in range(reps):
                    XY_block, Xii, Yii = self.pre_process_data(iv=iv)
                    if ii == 0:
                        Xi, Yi = Xii, Yii

                    # Perform Multi-Set Skeleton Prediction
                    cand = 4
                    preds = self.model.inference(XY_block, cand)
                    for ip, pred in enumerate(preds):
                        try:
                            tokenized = list(pred[1].cpu().numpy())[1:]
                            skeleton = seq2equation(tokenized, self.id2word, printFlag=False)
                            skeleton = sp.sympify(skeleton.replace('x_1', str(va)))
                            skeleton = sk_equivalence(avoid_operations_between_constants(standardize_expression(remove_coeffs(skeleton))), alts=True)
                            for skeleton_i in skeleton:
                                skeleton_i = avoid_operations_between_constants(skeleton_i)
                                if skeleton_i in pred_skeletons:
                                    ct = 0
                                    while skeleton_i in pred_skeletons:
                                        breakloop = False
                                        XY_block, Xii, Yii = self.pre_process_data(iv=iv)
                                        preds = self.model.inference(XY_block, cand)
                                        for predd in preds:
                                            tokenized = list(predd[1].cpu().numpy())[1:]
                                            skeleton = seq2equation(tokenized, self.id2word, printFlag=False)
                                            skeleton = sp.sympify(skeleton.replace('x_1', str(va)))
                                            skeleton = sk_equivalence(avoid_operations_between_constants(standardize_expression(remove_coeffs(skeleton))), alts=True)
                                            for skeleton_i2 in skeleton:
                                                skeleton_i2 = avoid_operations_between_constants(skeleton_i2)
                                                if skeleton_i2 not in pred_skeletons:
                                                    breakloop = True
                                                    pred_skeletons.append(skeleton_i2)
                                                    print('Predicted skeleton ' + str(len(pred_skeletons)) + ' for variable ' + str(va) + ': ' + str(skeleton_i2))
                                        if breakloop:
                                            break
                                        ct += 1
                                        if ct == 5:
                                            break
                                    if ct == 5:
                                        continue
                                else:
                                    pred_skeletons.append(skeleton_i)
                                    print('Predicted skeleton ' + str(len(pred_skeletons)) + ' for variable ' + str(va) + ': ' + str(skeleton_i))
                        except:  # TypeError:
                            print("Invalid response created by the model")

                continue
                print("\nChoosing the best skeleton... (skeletons ordered based on number of nodes)")
                best_perf, best_sk = 0, ''
                pred_skeletons = sorted(pred_skeletons, key=lambda expr: count_nodes(expr))
                nodes_num = [0 for expr in pred_skeletons]
                tested_skeletons, tested_skeletons_orig, perf_ind_vals, fitted_exprs = [], [], [], []
                print(pred_skeletons)
                for ip, skeleton in enumerate(pred_skeletons):
                    print(skeleton)
                    # Fit coefficients of the estimated skeletons
                    if skeleton not in tested_skeletons_orig:
                        try:
                            nre = 200
                            if flag:
                                nre = 20
                            problem = FitGA(remove_coeffs(skeleton), Xi, Yi, [np.min(Xi), np.max(Xi)], [-80, 80], max_it=nre,  # TODO None
                                            loss_MSE=True, pop_size=400)
                            est_expr, perf, _ = problem.run()
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
                                    if np.mean(np.abs(ratio)) >= 0.015:
                                        new_st_expr += arg
                                est_expr = new_st_expr
                            if len(str(expr2skeleton(est_expr))) >= len(str(remove_coeffs(skeleton))):
                                skeleton = avoid_operations_between_constants(standardize_expression(remove_coeffs(skeleton)))
                            else:
                                skeleton = avoid_operations_between_constants(expr2skeleton(est_expr))
                        except AttributeError:
                            skeleton = avoid_operations_between_constants(skeleton)
                        try:
                            skeleton = avoid_operations_between_constants(sk_equivalence(skeleton))
                        except IndexError:
                            continue
                        nodes_num[ip] = count_nodes(skeleton)
                        skeleton = avoid_operations_between_constants(sp.expand(skeleton))
                        print("\tSkeleton: " + str(skeleton) + ". Correlation: " + str(abs(perf)) + ". Expr: " + str(est_expr) + ". Perf: " + str((abs(perf) * 100 + nodes_num[ip]) / 100))
                        perf = (abs(perf) * 100 + nodes_num[ip]) / 100
                        if abs(perf) <= 10:  # Avoid including skeletons that obviously don't fit
                            tested_skeletons_orig.append(skeleton_orig)
                            tested_skeletons.append(skeleton)
                            perf_ind_vals.append(-perf)
                            fitted_exprs.append(est_expr)
                        if not self.mult_outputs:
                            if abs(best_perf - perf) > 0.002:
                                best_perf = perf
                                best_sk = expr2skeleton(2 * est_expr + 1)
                                if 1 >= abs(perf) > 0.999:  # If perfelation is very high, assume this is the best
                                    break
                if not self.mult_outputs:
                    print('-----------------------------------------------------------')
                    print("Selected skeleton: " + str(best_sk) + "\n")
                    self.univariate_skeletons.append(best_sk)
                    perf_vals.append(abs(best_perf))
                else:
                    # Sort skeletons according to their perfelation values
                    sorted_indices = sorted(range(len(corr_ind_vals)), key=lambda i: perf_ind_vals[i], reverse=True)
                    sorted_skeletons = [tested_skeletons[i] for i in sorted_indices]
                    sorted_expressions = [fitted_exprs[i] for i in sorted_indices]
                    perf_ind_vals = sorted(perf_ind_vals, reverse=True)
                    perf_ind_vals2, sorted_skeletons2, sorted_expressions2 = [], [], []
                    n_sk = len(sorted_skeletons)
                    n_candidates = self.n_candidates
                    # n_candidates = 1
                    if n_sk > n_candidates:
                        n_sk = n_candidates
                    i, fS = 0, True
                    while len(sorted_skeletons2) < n_sk and i < len(sorted_skeletons):
                        # if i == 0 and fS:  # Always includes the first skeleton that was generated
                        #     sorted_skeletons2.append(tested_skeletons[0])
                        #     sorted_expressions2.append(fitted_exprs[0])
                        #     perf_ind_vals2.append(perf_ind_vals[0])
                        #     fS = False
                        #     continue
                        if not any([str(sorted_skeletons[i]) == str(others) for others in sorted_skeletons2]):
                            sorted_skeletons2.append(sorted_skeletons[i])
                            sorted_expressions2.append(sorted_expressions[i])
                            perf_ind_vals2.append(perf_ind_vals[i])
                        i += 1
                    perf_ind_vals, sorted_skeletons, sorted_expressions = perf_ind_vals2, sorted_skeletons2, sorted_expressions2
                    self.univariate_skeletons.append(sorted_skeletons)
                    univ_sorted_exprs.append(sorted_expressions)
                    perf_vals.append(max(perf_ind_vals))
                    print(sorted_skeletons)

        return self.univariate_skeletons, perf_vals, univ_sorted_exprs


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
