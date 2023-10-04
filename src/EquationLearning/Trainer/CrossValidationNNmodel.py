import sys
import torch
from src.utils import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from src.EquationLearning.models.NNModel import NNModel
from src.EquationLearning.Data.GenerateDatasets import DataLoader


def get_train_test(first, second, dataset_size, crossval='10x1'):
    """Get indices of training and test samples depending on the selected cross-validation type"""
    if crossval == '10x1':
        # Gets the list of training and test samples using kfold.split
        train = np.array(first)
        test = np.array(second)
    else:
        # Split the dataset in 2 parts with the current seed
        train, test = train_test_split(range(dataset_size), test_size=0.50, random_state=second)
        train = np.array(train)
        test = np.array(test)
    return train, test


class Trainer:
    """Train NN model using cross-validation"""

    def __init__(self, dataset: str = 'S0', method: str = 'iEQL-GA', n_layers: int = 3):
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

    def init_kfold(self, crossval: str = "10x1"):
        # Initialize kfold object
        kfold = KFold(n_splits=10, shuffle=True, random_state=13)
        if crossval == "10x1":
            iterator = kfold.split(self.X)
            print("Using 10x1 cross-validation for this dataset")
        elif crossval == "5x2":
            # Choose seeds for each iteration is using 5x2 cross-validation
            seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
            iterator = enumerate(seeds)
            print("Using 5x2 cross-validation for this dataset")
        else:
            sys.exit("Only '10x1' and '5x2' cross-validation are permited.")
        return iterator

    def train(self, crossval='10x1', batch_size=32, epochs=500, printProcess=True, scratch: bool = False):
        """Train using cross validation
        :param crossval: Type of cross-validation. Options: '10x1' or '5x2'
        :param batch_size: Mini batch size. It is recommended a small number, like 16
        :param epochs: Number of training epochs
        :param printProcess: If True, print the training process (loss and validation metrics after each epoch)
        :param scratch: If True, re-train all networks from scratch
        """
        # Create lists to store metrics
        cvmse, cvpicp, cvmpiw, cvdiffs = [], [], [], []

        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "output//CVResults//" + self.dataset + "//" + self.method)
        if not os.path.exists(os.path.join(root, "output//CVResults//" + self.dataset)):
            os.mkdir(os.path.join(root, "output//CVResults//" + self.dataset))
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(folder + "//pruning_results"):
            os.mkdir(folder + "//pruning_results")

        print("\n*****************************************")
        print("*****************************************")
        print("Start MLP training")
        print("*****************************************")
        print("*****************************************")
        ntrain = 1
        iterator = self.init_kfold(crossval=crossval)
        # Iterate through each partition
        for first, second in iterator:
            train, test = get_train_test(first=first, second=second, dataset_size=len(self.X), crossval=crossval)
            print("******************************")
            print("Training fold: " + str(ntrain))
            print("******************************")
            # Normalize using the training set
            Xtrain, means, stds = self.X[train], None, None  # normalize(self.X[train])
            Ytrain, maxs, mins = self.Y[train], None, None  # minMaxScale(self.Y[train])
            Xval = self.X[test]  # applynormalize(self.X[test], means, stds)
            Yval = self.Y[test]  # applyMinMaxScale(self.Y[test], maxs, mins)

            # Train the model using the current training-validation split
            filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-" + str(ntrain)
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
            # Add metrics to the list
            cvmse.append(val_mse)
            ntrain += 1

        # Save metrics of all folds
        np.save(folder + '//validation_MSE-' + self.method + "-" + self.dataset, cvmse)
        # Save metrics in a txt file
        file_name = folder + "//regression_report-" + self.method + "-" + self.dataset + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall MSE %.6f (+/- %.6f)" % (float(np.mean(cvmse)), float(np.std(cvmse))))
            x_file.write('\n')

        return cvmse, cvmpiw, cvpicp


if __name__ == '__main__':
    name = 'gauss'
    predictor = Trainer(dataset=name)
    predictor.train(scratch=True, batch_size=128, epochs=5000, printProcess=True)
