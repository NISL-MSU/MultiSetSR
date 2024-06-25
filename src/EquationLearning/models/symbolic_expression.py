import sympy
import torch
from ...EquationLearning.models.network import NN, HiddenLayer
from ...EquationLearning.models.functions import get_sym_function


def get_weights_bias_layer(module):
    """Get weight and bias from the linear layer of a given module"""
    # Check if the input is a list of modules or a linear layer
    weights, bias = None, None
    if isinstance(module, HiddenLayer):
        for nm, mod in module.named_modules():
            if isinstance(mod, torch.nn.Linear):
                weights = mod.weight.to('cpu').detach().numpy()
                bias = mod.bias.to('cpu').detach().numpy()
    else:
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.to('cpu').detach().numpy()
            bias = module.bias.to('cpu').detach().numpy()
    return weights, bias


def get_unary_binary_operations(network: NN):
    """Take the set of unary and binary operations and transform them into sympy functions"""
    unary_operations = network.operations['unary']
    binary_operations = network.operations['binary']
    unary_functions, binary_functions = [], []
    # Unary functions
    for f in unary_operations:
        unary_functions.append(get_sym_function(f)[0])
    # Binary functions
    for f in binary_operations:
        binary_functions.append(get_sym_function(f)[0])
    return unary_functions, binary_functions


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sympy.Number)})


def get_expression(n_inputs: int, network: NN):
    """Get symbolic expression of a given NN
    @param n_inputs: Number of input variables
    @param network: Trained NN"""
    # Initialize variable names from x_1, ..., x_n
    in_symbols = sympy.symbols("{}:{}".format('x', n_inputs))

    # Get sympy functions
    unary_functions, binary_functions = get_unary_binary_operations(network)

    # Get expression of first layer
    exp = 0
    for moduleList in network.layers:
        weight, bias = get_weights_bias_layer(moduleList)
        z = []
        # Obtain activations after the linear layer
        for i in range(weight.shape[0]):
            exp = 0
            for j in range(weight.shape[1]):
                exp += in_symbols[j] * weight[i, j]
            exp += bias[i]
            z.append(exp)
        z = [sympy.expand(zi) for zi in z]
        # Pass the activation z through the operations layer
        # Apply the non-linear unary functions
        unary_transformations = [f(z[i]) for i, f in enumerate(unary_functions)]
        # Apply the non-linear binary functions
        if len(binary_functions) > 0:
            binary_transformations, n = [], 0
            for i in range(len(unary_functions), weight.shape[0], 2):  # Grab consecutive pairs
                binary_transformations.append(binary_functions[n](z[i], z[i + 1]))
                n += 1
            # Concatenate unary and binary transformations
            y = unary_transformations + binary_transformations
        else:
            y = unary_transformations
        # y = [sympy.simplify(yi) for yi in y]
        in_symbols = y + list(in_symbols)

    # Get expression of last layer
    weight, bias = get_weights_bias_layer(network.out)
    for i in range(weight.shape[0]):
        exp = 0
        for j in range(weight.shape[1]):
            exp += in_symbols[j] * weight[i, j]
        exp += bias
    exp = exp[0]
    return round_expr(exp, 3)
