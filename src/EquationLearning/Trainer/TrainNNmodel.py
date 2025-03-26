import torch
from EquationLearning.utils import *
from sklearn.model_selection import KFold
from EquationLearning.models.NNModel import NNModel
from EquationLearning.Data.GenerateDatasets import DataLoader, InputData


class Trainer:
    """Train NN model using cross-validation"""

    def __init__(self, dataset: InputData, modelType: str = 'NN', name: str = '', noise: float = 0):
        """
        Initialize Trainer class
        :param dataset: An InputData object.
        """
        # Class variables
        self.dataset = dataset
        if name == '':
            name = 'temp'
        self.name = name
        # Read dataset
        self.X, self.Y, self.var_names = self.dataset.X, self.dataset.Y, self.dataset.names
        self.modelType = modelType
        self.n_features = self.X.shape[1]
        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()
        self.noise = noise
        self.noise_name = ''
        if self.noise > 0:
            self.noise_name = '_noise-' + str(noise)

    def reset_model(self):
        return NNModel(device=self.device, n_features=self.n_features, NNtype=self.modelType)

    def init_kfold(self):
        # Initialize kfold object
        kfold = KFold(n_splits=10, shuffle=True, random_state=7)
        iterator = kfold.split(self.X)
        return iterator

    def train(self, batch_size=32, epochs=500, printProcess=True, scratch: bool = True):
        """Train using 90% of the data for training and 10% for validation
        :param batch_size: Mini batch size. It is recommended a small number, like 16
        :param epochs: Number of training epochs
        :param printProcess: If True, print the training process (loss and validation metrics after each epoch)
        :param scratch: If True, re-train all networks from scratch
        """

        # If the folder does not exist, create it
        root = get_project_root()
        folder = ''
        if self.name != '':
            folder = os.path.join(root, "EquationLearning//saved_models//saved_NNs//" + self.name)
            if not os.path.exists(os.path.join(root, "EquationLearning//saved_models//saved_NNs//" + self.name)):
                os.mkdir(os.path.join(root, "EquationLearning//saved_models//saved_NNs//" + self.name))
            if not os.path.exists(folder):
                os.mkdir(folder)

        print("*****************************************")
        print("Start MLP training")
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
            filepath = None
            if self.name != '':
                filepath = folder + "//weights-NN-" + self.name + self.noise_name
            if scratch or not os.path.exists(filepath):
                _, val_mse = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                  batch_size=batch_size, epochs=epochs, filepath=filepath,
                                                  printProcess=printProcess,
                                                  yscale=[maxs, mins])
                self.model.loadModel(path=filepath)
            else:  # Or just load pre-trained NN
                self.model.loadModel(path=filepath)
                # Evaluate model on the validation set
                val_mse = mse(Yval, np.array(self.model.evaluateFold(Xval))[:, 0])
            # Reset all weights
            # self.model = self.reset_model()
            print("Val MSE: " + str(val_mse))
            break


if __name__ == '__main__':
    names = ['Y1']  # E6  # CS1
    noise_level = 0  # 0.05
    for nme in names:
        data_loader = DataLoader(name=nme, noise=noise_level)
        predictor = Trainer(dataset=data_loader.dataset, modelType=data_loader.modelType, name=data_loader.name, noise=noise_level)
        predictor.train(scratch=True, batch_size=128, epochs=3000, printProcess=True)
