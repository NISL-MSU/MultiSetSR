import torch
import omegaconf
import sympy as sp
from src.utils import *
from src.EquationLearning.models.NNModel import NNModel
from src.EquationLearning.Transformers.model import Model
from src.EquationLearning.Data.GenerateDatasets import DataLoader
from src.EquationLearning.Transformers.GenerateTransformerData import Dataset


class SymbolicRegressor:

    def __init__(self, dataset: str = 'E1'):
        """Distills symbolic skeleton expressions given an experimental dataset"""
        # Define problem
        self.dataset = dataset
        dataLoader = DataLoader(name=self.dataset)
        self.X, self.Y, self.var_names = dataLoader.X, dataLoader.Y, dataLoader.names
        self.limits = [-10, 10]
        self.modelType = dataLoader.modelType
        self.n_features = self.X.shape[1]
        self.symbols = sp.symbols("{}:{}".format('x', self.n_features))
        self._load_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Read config yaml
        try:
            self.cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config.yaml")
        except FileNotFoundError:
            self.cfg = omegaconf.OmegaConf.load("../Transformers/config.yaml")

        # Load MST
        self.data_train_path = self.cfg.train_path
        self.training_dataset = Dataset(self.data_train_path, self.cfg.dataset_train, mode="train")
        self.word2id = self.training_dataset.word2id
        self.model = Model(cfg=self.cfg.architecture, cfg_inference=self.cfg.inference, word2id=self.word2id)

        # Initialize class variables
        self.univariate_skeletons = []
        self.merged_expressions = []
        self.n_sets = 10
        self.n_samples = 1000

    def _load_model(self):
        # Define NN and load weights
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn_model = NNModel(device=device, n_features=self.n_features, NNtype=self.modelType)
        root = get_project_root()
        folder = os.path.join(root, "src//EquationLearning//models//saved_NNs//" + self.dataset)
        self.filepath = folder + "//weights-NN-" + self.dataset
        self.nn_model.loadModel(self.filepath)

    def get_skeleton(self):
        """Retrieve the estimated symbolic equation"""
        # Analyze each variable and obtain univariate expressions
        for iv, va in enumerate(self.symbols):
            print("********************************")
            print("Analyzing variable " + str(va))
            print("********************************")

            # Generate multiple sets of data where only the current variable is allowed to vary
            Xs = np.zeros((self.n_samples, len(self.symbols), self.n_sets))
            Ys = np.zeros((self.n_samples, self.n_sets))
            for ns in range(self.n_sets):
                # Sample random values for all the variables
                values = np.expand_dims(np.array([np.random.uniform(self.limits[0], self.limits[1])
                                                      for v in range(len(self.symbols))]), axis=1)
                values = np.repeat(values, self.n_samples, axis=1)
                # Sample values of the variable that is being analyzed
                sample = np.random.uniform(self.limits[0], self.limits[1], self.n_samples)
                values[:, iv] = sample
                # Estimate the response of the generated set
                Ys[:, :, ns] = np.array(self.nn_model.evaluateFold(values.T, batch_size=len(values)))[:, 0]
                Xs[:, :, ns] = values

            # Format the data as inputs to the Multi-set transformer
            XY_block = torch.zeros((1, self.n_samples, 2, self.n_sets)).to(self.device)
            Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
            Xs = Xs.to(self.device)
            Ys = Ys.to(self.device)
            XY_block[0, :, 0, :] = Xs
            XY_block[0, :, 1, :] = Ys

            self.univariate_skeletons.append(self.model.inference(XY_block))

        return self.univariate_skeletons


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()

    regressor = SymbolicRegressor(dataset='E1')
    regressor.get_skeleton()
