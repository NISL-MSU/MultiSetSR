import pickle
import random
from src import utils
from tqdm import trange
from torch import optim
from src.EquationLearning.models.network import *

np.random.seed(7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#######################################################################################################################
# Static functions and Loss functions
#######################################################################################################################


#######################################################################################################################
# Class Definitions
#######################################################################################################################

class NNObject:
    """Helper class used to store the main information of a NN model."""

    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class NNModel:
    """Define a feedforward symbolic neural network"""

    def __init__(self, device, n_features: int, NNtype: str = 'NN', operations: dict = None, n_layers: int = 4):
        """
        Initialize NN object
        :param device: Where is stored the model. "cuda:0" or "cpu".
        :param NNtype: Name of the NN architecture that will be used.
        :param operations: Dictionary consisting of two sets: the unary and binary operations.
                           E.g. operations['unary'] = {'id', 'sin', 'cos'}; operations['binary'] = {'+', '-', '*'}.
        :param n_features: Input shape of the network.
        :param n_layers: Number of hidden layers used by the neural network
        """
        self.device = device
        self.n_features = n_features
        self.operations = operations
        self.output_size = 1
        self.n_layers = n_layers

        criterion = nn.MSELoss()
        if NNtype == "NN":
            network = MLP(input_features=self.n_features,
                          output_size=self.output_size,
                          n_layers=self.n_layers)
        elif NNtype == "NN2":
            network = MLP2(input_features=self.n_features,
                           output_size=self.output_size,
                           n_layers=self.n_layers)
        else:
            network = MLP3(input_features=self.n_features,
                           output_size=self.output_size,
                           n_layers=self.n_layers)
        network.to(self.device)
        # Training parameters
        optimizer = optim.Adadelta(network.parameters(), lr=0.05)

        self.model = NNObject(network, criterion, optimizer)

    def trainFold(self, Xtrain, Ytrain, Xval, Yval, batch_size, epochs, filepath, printProcess, yscale):
        # Set seeds
        np.random.seed(7)
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare list of indexes for shuffling
        indexes = np.arange(len(Xtrain))
        np.random.shuffle(indexes)
        Xtrain = Xtrain[indexes]
        Ytrain = Ytrain[indexes]
        T = np.ceil(1.0 * len(Xtrain) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch

        val_mse = np.infty
        MSE, MSEtr = [], []

        for epoch in trange(epochs):  # Epoch loop
            # Prepare list of indexes for shuffling
            indexes = np.arange(len(Xtrain))
            np.random.shuffle(indexes)

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]

                # Get actual batches
                Xtrainb = torch.from_numpy(Xtrain[inds]).float().to(self.device)
                Ytrainb = torch.from_numpy(Ytrain[inds]).float().to(self.device)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.network(Xtrainb)
                outputs = outputs.squeeze(1)
                loss = self.model.criterion(outputs, Ytrainb)
                loss.backward()
                self.model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if printProcess and epoch % 10 == 0:
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, step + 1, loss.item()))

            # Validation step
            with torch.no_grad():
                self.model.network.eval()
                ypredtr = self.evaluateFold(Xtrain, batch_size=1024)
                ypred = self.evaluateFold(Xval, batch_size=1024)
                # Reverse normalization
                if yscale[0] is not None and yscale[1] is not None:
                    Ytrain_original = utils.reverseMinMaxScale(Ytrain, yscale[0], yscale[1])
                    Yval_original = utils.reverseMinMaxScale(Yval, yscale[0], yscale[1])
                    ypredtr = utils.reverseMinMaxScale(ypredtr, yscale[0], yscale[1])
                    ypred = utils.reverseMinMaxScale(ypred, yscale[0], yscale[1])
                else:
                    Ytrain_original = Ytrain
                    Yval_original = Yval
                    ypredtr = ypredtr
                    ypred = ypred

                # Calculate MSE
                msetr = utils.mse(Ytrain_original, ypredtr[:, 0])
                mse = utils.mse(Yval_original, ypred[:, 0])
                MSEtr.append(msetr)
                MSE.append(mse)  # np.concatenate((np.reshape(Yval_original, (len(Yval_original), 1)), ypred), axis=1)

            # Save model if MSE decreases
            if mse < val_mse:
                val_mse = mse
                if filepath is not None:
                    torch.save(self.model.network.state_dict(), filepath)

            # Print every 10 epochs
            if printProcess and epoch % 10 == 0:
                print('VALIDATION: Training_MSE: %.6f. MSE val: %.6f. Best_MSE: %.6f' % (msetr, mse, val_mse))

        # Save model
        if filepath is not None:
            with open(filepath + '_validationMSE', 'wb') as fil:
                pickle.dump(val_mse, fil)
            # Save history
            np.save(filepath + '_historyMSEtr', MSEtr)
            np.save(filepath + '_historyMSE', MSE)

        return MSE, val_mse

    def evaluateFold(self, valxn, maxs=None, mins=None, batch_size=96):
        """Retrieve point predictions."""
        if maxs is not None and mins is not None:
            valxn = utils.reverseMinMaxScale(valxn, maxs, mins)

        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(valxn) / batch_size).astype(np.int32)
            indtest = np.arange(len(valxn))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                ypred = ypred + (ypred_batch.cpu().numpy()).tolist()
        return np.array(ypred)

    def loadModel(self, path):
        self.model.network.load_state_dict(torch.load(path))
