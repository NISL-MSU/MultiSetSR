import torch
import omegaconf
import sympy as sp
from src.utils import *
from src.EquationLearning.models.NNModel import NNModel
from src.EquationLearning.Transformers.model import Model
from src.EquationLearning.Data.GenerateDatasets import DataLoader
from src.EquationLearning.Transformers.GenerateTransformerData import Dataset
from src.EquationLearning.Trainer.TrainMultiSetTransformer import seq2equation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class SymbolicRegressor:

    def __init__(self, dataset: str = 'E1'):
        """Distills symbolic skeleton expressions given an experimental dataset"""
        # Define problem
        self.dataset = dataset
        dataLoader = DataLoader(name=self.dataset)
        self.X, self.Y, self.var_names = dataLoader.X, dataLoader.Y, dataLoader.names
        self.target_function = dataLoader.expr
        self.f_lambdified = sp.lambdify(sp.utilities.iterables.flatten(sp.sympify(dataLoader.names)), dataLoader.expr)
        self.limits = [dataLoader.limits[0][0], dataLoader.limits[0][1]]
        self.modelType = dataLoader.modelType
        self.n_features = self.X.shape[1]
        self.symbols = sp.symbols("{}:{}".format('x', self.n_features))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Read config yaml
        try:
            self.cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config.yaml")
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
        self.n_sets = 10
        self.n_samples = 5000

    def _load_models(self):
        # Define NN and load weights
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn_model = NNModel(device=device, n_features=self.n_features, NNtype=self.modelType)
        root = get_project_root()
        folder = os.path.join(root, "src//EquationLearning//models//saved_NNs//" + self.dataset)
        self.filepath = folder + "//weights-NN-" + self.dataset
        self.nn_model.loadModel(self.filepath)
        # Load weights of MST
        MST_path = os.path.join(root, "src//EquationLearning//models//saved_models/Model-dataset0")
        self.model.load_state_dict(torch.load(MST_path))
        self.model.cuda()

    def get_skeleton(self):
        """Retrieve the estimated symbolic equation"""
        # Analyze each variable and obtain univariate expressions
        for iv, va in enumerate(self.symbols):
            print("********************************")
            print("Analyzing variable " + str(va))
            print("********************************")
            if iv >= 0:
                # Generate multiple sets of data where only the current variable is allowed to vary
                Xs = np.zeros((self.n_samples, self.n_sets))
                Ys = np.zeros((self.n_samples, self.n_sets))
                Ys_real = np.zeros((self.n_samples, self.n_sets))
                for ns in range(self.n_sets):
                    # Repeat the sampling process a few times and keep the one the looks more different from a line
                    R2s, XXs, YYs, valuess = [], [], [], []
                    for it in range(5):
                        # Sample random values for all the variables
                        values = np.expand_dims(np.array([np.random.uniform(self.limits[0], self.limits[1])
                                                              for _ in range(len(self.symbols))]), axis=1)
                        values = np.repeat(values, self.n_samples, axis=1)
                        # Sample values of the variable that is being analyzed
                        sample = np.random.uniform(self.limits[0], self.limits[1], self.n_samples)
                        values[iv, :] = sample
                        # Estimate the response of the generated set
                        Y = np.array(self.nn_model.evaluateFold(values.T, batch_size=len(values)))[:, 0]
                        X = sample
                        # Fit linear regression model and calculate R2
                        model = LinearRegression()
                        model.fit(X[:, None], Y)
                        Y_pred = model.predict(X[:, None])
                        r2 = r2_score(Y, Y_pred)
                        R2s.append(r2)
                        XXs.append(X.copy())
                        YYs.append(Y.copy())
                        valuess.append(values.copy())
                        # print(r2)
                        # if abs(r2) < abs(R2min):
                        #     R2min = r2
                        #     best_X, best_Y, best_values = X.copy(), Y.copy(), values.copy()
                        # if abs(R2min) < 0.9:  # If it's obvious it's not a line, break the loop
                        #     break
                    sorted_indices = np.argsort(np.array(R2s))
                    ind = sorted_indices[1]
                    best_X, best_Y, best_values = XXs[ind], YYs[ind], valuess[ind]
                    Ys[:, ns] = best_Y
                    Xs[:, ns] = best_X
                    Ys_real[:, ns] = np.array(self.f_lambdified(*list(best_values)))
                    # Normalize data
                    means, std = np.mean(Ys, axis=0), np.std(Ys_real, axis=0)
                    Ys = (Ys - means) / std
                    means, std = np.mean(Ys_real, axis=0), np.std(Ys_real, axis=0)
                    Ys_real = (Ys_real - means) / std

                # Format the data as inputs to the Multi-set transformer
                scaling_factor = 20 / (np.max(Xs) - np.min(Xs))
                Xs = (Xs - np.min(Xs)) * scaling_factor - 10
                XY_block = torch.zeros((1, self.n_samples, 2, self.n_sets)).to(self.device)
                Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
                Xs = Xs.to(self.device)
                Ys = Ys.to(self.device)
                XY_block[0, :, 0, :] = Xs
                XY_block[0, :, 1, :] = Ys

                # Perform Multi-Set Skeleton Prediction
                # tokenized = self.model.validation_step(XY_block)[1][0]
                # skeleton = sp.sympify(seq2equation(tokenized, self.id2word))
                skeleton = None
                preds = self.model.inference(XY_block)
                for ip, pred in enumerate(preds):
                    try:
                        tokenized = list(pred[1].cpu().numpy())[1:]
                        skeleton = seq2equation(tokenized, self.id2word)
                        skeleton = sp.sympify(skeleton.replace('x_1', str(va)))
                        print('Predicted skeleton' + str(ip) + ' for variable ' + str(va) + ': ' + str(skeleton))
                    except TypeError:
                        print("Invalid response created by the model")

                self.univariate_skeletons.append(skeleton)

        return self.univariate_skeletons


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()

    regressor = SymbolicRegressor(dataset='E3')
    regressor.get_skeleton()
