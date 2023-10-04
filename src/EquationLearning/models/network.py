import torch
import numpy as np
from abc import ABC
import torch.nn as nn
from torch import stack, cat
from src.EquationLearning.models.functions import get_function


class MLP(nn.Module, ABC):
    """Defines conventional NN architecture"""

    def __init__(self, input_features: int = 10, output_size: int = 1, n_layers: int = 3):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        :param n_layers: Number of hidden layers.
        """
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=100), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.1)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        # x = self.drop1(x)
        x = self.hidden_layer2(x)
        # x = self.drop2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        return self.out(x)


class MLP2(nn.Module, ABC):
    """Defines conventional NN architecture"""

    def __init__(self, input_features: int = 10, output_size: int = 1, n_layers: int = 3):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        :param n_layers: Number of hidden layers.
        """
        super(MLP2, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=300), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.05)
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(in_features=300, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        # x = self.drop1(x)
        # x = self.hidden_layer2(x)
        # x = self.hidden_layer3(x)
        # x = self.drop2(x)
        # x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return self.out(x)


class MLP3(nn.Module, ABC):
    """Defines conventional NN architecture"""

    def __init__(self, input_features: int = 10, output_size: int = 1, n_layers: int = 3):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        :param n_layers: Number of hidden layers.
        """
        super(MLP3, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=200), nn.ReLU())
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=500), nn.ReLU())
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return self.out(x)


class HiddenLayer(nn.Module):
    """Defines the structure of a hidden layer"""

    def __init__(self, operations: dict, in_features: int):
        """
        Initialize a hidden layer
        :param operations: Dictionary consisting of two sets: the unary and binary operations.
        """
        super().__init__()
        # Get operations
        self.unary_operations = operations['unary']
        self.binary_operations = operations['binary']
        self.in_features = in_features
        # Set torch functions
        self.unary_functions, self.binary_functions = [], []
        self._define_functions()
        # Define feedforward layer
        self.n_hidden_units = len(self.unary_functions) + 2 * len(self.binary_functions)
        self.linear_layer = nn.Linear(in_features=self.in_features, out_features=self.n_hidden_units)
        self.output_size = len(self.unary_functions) + len(self.binary_functions)

    def _define_functions(self):
        """Generate the set of functions to be used based on the user-defined operations"""
        # Unary functions
        for f in self.unary_operations:
            self.unary_functions.append(get_function(f)[0])
        # Binary functions
        for f in self.binary_operations:
            self.binary_functions.append(get_function(f)[0])

    def forward(self, x):
        """Define forward operation"""
        # First apply a feedforward layer
        z = self.linear_layer(x)
        # Apply the non-linear unary functions
        unary_transformations = stack([f(z[..., i]) for i, f in enumerate(self.unary_functions)], -1)
        # Apply the non-linear binary functions
        if len(self.binary_functions) > 0:
            binary_transformations, n = [], 0
            for i in range(len(self.unary_functions), self.n_hidden_units, 2):  # Grab consecutive pairs
                binary_transformations.append(self.binary_functions[n](z[..., i], z[..., i + 1]))
                n += 1
            binary_transformations = stack(binary_transformations, -1)
            # Concatenate unary and binary transformations
            y = cat((unary_transformations, binary_transformations), -1)
        else:
            y = unary_transformations
        return y


class NN(nn.Module, ABC):
    """Defines NN architecture that uses unary and binary operations"""

    def __init__(self, operations: dict,
                 input_features: int = 10,
                 output_size: int = 1,
                 n_layers: int = 2):
        """
        Initialize NN
        :param operations: Dictionary consisting of two sets: the unary and binary operations.
                           E.g. operations['unary'] = {'log', 'sin', 'cos'}; operations['binary'] = {'+', '-', '*'}.
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        :param n_layers: Number of hidden layers.
        """
        super(NN, self).__init__()
        self.n_layers = n_layers
        self.operations = operations

        # Define first layer
        self.layers = nn.ModuleList()

        # Define remaining layers
        for n in range(self.n_layers):
            self.layers.append(HiddenLayer(operations=self.operations, in_features=input_features))
            input_features += self.layers[-1].output_size  # Increases because we'll use skip connections from previous layers

        # Define last layer
        self.out = nn.Linear(input_features, output_size)

    def forward(self, x):
        prev_r = []
        for layer in self.layers:
            x_new = layer(x)
            x = cat((x_new, x), 1)  # Concatenate with results from previous layer
            prev_r.append(x)
        return self.out(x)

    def flatten_parameters(self):
        """Flatten parameters
        :return flattened_parameters: List of flattened parameters that are different than 0
        :return pruning_mask: Mask that identifies zero-weight parameters"""
        flattened_parameters = []
        # Flatten parameters from the HiddenLayer layers
        for moduleList in self.layers:
            for nm, module in moduleList.named_modules():
                if isinstance(module, torch.nn.Linear):
                    weights = module.weight.to('cpu').detach().numpy()
                    bias = module.bias.to('cpu').detach().numpy()
                    flattened_parameters += list(weights.flatten())
                    flattened_parameters += list(bias.flatten())
        # Flatten parameters from the output layer
        weights = self.out.weight.to('cpu').detach().numpy()
        bias = self.out.bias.to('cpu').detach().numpy()
        flattened_parameters += list(weights.flatten())
        flattened_parameters += list(bias.flatten())
        pruning_mask = [p == 0 for p in flattened_parameters]
        flattened_parameters = np.array(flattened_parameters)
        return flattened_parameters[flattened_parameters != 0], pruning_mask

    def set_parameters(self, parameters: list, pruning_mask: list):
        """Takes a modified set of parameters and set them as the new network parameters
        :param parameters: Flattened parameters
        :param pruning_mask: Mask that indicates where the flattened parameters will be inserted"""
        parameters_complete = [0] * len(pruning_mask)
        c = 0
        for n in range(len(parameters_complete)):
            if not pruning_mask[n]:
                parameters_complete[n] = parameters[c]
                c += 1
        parameters = parameters_complete
        with torch.no_grad():
            for moduleList in self.layers:
                for nm, module in moduleList.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Weights
                        shape = module.weight.to('cpu').detach().numpy().shape
                        flattened_shape = shape[0] * shape[1]
                        # Pop parameters from list
                        unflattened_parameters = np.reshape(parameters[:flattened_shape], shape)
                        module.weight.copy_(torch.from_numpy(unflattened_parameters))
                        parameters = parameters[flattened_shape:]
                        # Bias
                        shape = module.bias.to('cpu').detach().numpy().shape
                        flattened_shape = shape[0]
                        # Pop parameters from list
                        module.bias.copy_(torch.from_numpy(np.array(parameters[:flattened_shape])))
                        parameters = parameters[flattened_shape:]
            # Weights output layer
            shape = self.out.weight.to('cpu').detach().numpy().shape
            flattened_shape = shape[0] * shape[1]
            # Pop parameters from list
            unflattened_parameters = np.reshape(parameters[:flattened_shape], shape)
            self.out.weight.copy_(torch.from_numpy(unflattened_parameters))
            parameters = parameters[flattened_shape:]
            # Bias output layer
            shape = self.out.bias.to('cpu').detach().numpy().shape
            flattened_shape = shape[0]
            # Pop parameters from list
            self.out.bias.copy_(torch.from_numpy(np.array(parameters[:flattened_shape])))


if __name__ == '__main__':
    ops = {'unary': ('cos', 'sin'), 'binary': ('+', '-')}
    # Define NN
    netw = NN(operations=ops, input_features=3, n_layers=3)
    # Create simulated input and pass it through the network
    x_in = torch.zeros(200, 3)
    y_out = netw(x_in)
