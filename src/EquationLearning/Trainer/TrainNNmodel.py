import torch
from src.utils import *
from sklearn.model_selection import KFold
from src.EquationLearning.models.NNModel import NNModel
from src.EquationLearning.Data.GenerateDatasets import DataLoader


class Trainer:
    """Train NN model using cross-validation"""

    def __init__(self, dataset: str = 'S0', method: str = 'NN', n_layers: int = 3):
        """
        Initialize Trainer class
        :param dataset: Name of the dataset to be used. E.g., 'S0', 'S1', ..., 'S7', 'A1', ..., 'A5'.
        :param method: Name of the chosen equation learning method. E.g., 'EQL', 'iEQL', 'iEQL-GA'.
        :param n_layers: Number of hidden layers used by the neural network
        """
        # Class variables
        self.dataset = dataset
        self.method = method
        self.n_layers = n_layers

        # Read dataset
        dataLoader = DataLoader(name=dataset)
        self.X, self.Y, self.var_names = dataLoader.X, dataLoader.Y, dataLoader.names
        self.modelType = dataLoader.modelType
        self.n_features = self.X.shape[1]

        # Load model
        print("Loading model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()

    def reset_model(self):
        return NNModel(device=self.device, n_features=self.n_features, n_layers=self.n_layers, NNtype=self.modelType)

    def init_kfold(self):
        # Initialize kfold object
        kfold = KFold(n_splits=10, shuffle=True, random_state=7)
        iterator = kfold.split(self.X)
        return iterator

    def train(self, batch_size=32, epochs=500, printProcess=True, scratch: bool = False):
        """Train using cross validation
        :param batch_size: Mini batch size. It is recommended a small number, like 16
        :param epochs: Number of training epochs
        :param printProcess: If True, print the training process (loss and validation metrics after each epoch)
        :param scratch: If True, re-train all networks from scratch
        """

        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "src//EquationLearning//models//saved_NNs//" + self.dataset)
        if not os.path.exists(os.path.join(root, "src//EquationLearning//models//saved_NNs//" + self.dataset)):
            os.mkdir(os.path.join(root, "src//EquationLearning//models//saved_NNs//" + self.dataset))
        if not os.path.exists(folder):
            os.mkdir(folder)

        print("\n*****************************************")
        print("*****************************************")
        print("Start MLP training")
        print("*****************************************")
        print("*****************************************")
        iterator = self.init_kfold()
        # Use only first partition
        for first, second in iterator:
            train = np.array(first)
            test = np.array(second)
            # Normalize using the training set
            Xtrain, means, stds = self.X[train], None, None  # normalize(self.X[train])
            Ytrain, maxs, mins = self.Y[train], None, None  # minMaxScale(self.Y[train])
            Xval = self.X[test]  # applynormalize(self.X[test], means, stds)
            Yval = self.Y[test]  # applyMinMaxScale(self.Y[test], maxs, mins)

            # Train the model using the current training-validation split
            filepath = folder + "//weights-" + self.method + "-" + self.dataset
            if scratch or not os.path.exists(filepath):
                _, val_mse = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                  batch_size=batch_size, epochs=epochs, filepath=filepath,
                                                  printProcess=printProcess,
                                                  yscale=[maxs, mins])
            else:  # Or just load pre-trained NN
                self.model.loadModel(path=filepath)
                # Evaluate model on the validation set
                val_mse = mse(Yval, np.array(self.model.evaluateFold(Xval))[:, 0])
            # Reset all weights
            self.model = self.reset_model()
            print("Val MSE: " + str(val_mse))
            break


if __name__ == '__main__':
    names = ['CS1']
    for name in names:
        predictor = Trainer(dataset=name)
        predictor.train(scratch=True, batch_size=128, epochs=10000, printProcess=True)
