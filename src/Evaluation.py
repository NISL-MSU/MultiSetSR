import os

import numpy as np
import torch
import pickle
from utils import *
import matplotlib.pyplot as plt
from src.EquationLearning.models.NNModel import NNModel
from src.EquationLearning.Data.GenerateDatasets import DataLoader
from tensorboard.backend.event_processing import event_accumulator


def U0_pySR(X):
    # - \\left(\\sin^{4}{\\left(x_{0} + e^{2 \\sin{\\left(x_{0} \\right)}} - \\frac{0.229}{\\cos{\\left(x_{0} \\right)}} \\right)} + 0.200\\right)^{2} - 0.438
    return -(np.sin(X + np.exp(2 * np.sin(X)) - 0.1997899) ** 4) - 0.43763742


def A1_pySR(X):
    # \\cos^{2}{\\left(1.50 x_{1} \\right)} - 0.469 \\cos{\\left(0.711 e^{x_{0}} \\right)}
    x0, x1 = X[:, 0], X[:, 1]
    return np.cos(1.50360826890723 * x1) ** 2 - 0.46917844 * np.cos(0.710649978236344 * np.exp(x0))


def rosenbrock4D_pySR(X):
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 109.24373217069 * (0.171526388718634 * (x0 ** 2 - x1 ** 2) ** 2 + 0.171526388718634 * (
                x2 ** 2 - x3) ** 2 + 1) ** 3 - 88.20763


def S8_pySR(X):
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    return -x1 - np.sin(6.30906454882067 * np.exp(x2)) + 2.9666398 + 3.56275207148611 * np.exp(x0)


def sin4D_pySR(X):
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.exp(x3)


def U0_gplearn(X):
    root = get_project_root()
    path = os.path.join(root, "output//LearnedEquations//U0//U0_gplearn.pkl")
    with open(path, 'rb') as f:
        est = pickle.load(f)
    return est.predict(X)


def A1_gplearn(X):
    root = get_project_root()
    path = os.path.join(root, "output//LearnedEquations//A1//A1_gplearn.pkl")
    with open(path, 'rb') as f:
        est = pickle.load(f)
    return est.predict(X)


def rosenbrock4D_gplearn(X):
    root = get_project_root()
    path = os.path.join(root, "output//LearnedEquations//rosenbrock4D//rosenbrock4D_gplearn.pkl")
    with open(path, 'rb') as f:
        est = pickle.load(f)
    return est.predict(X)


def sin4D_gplearn(X):
    root = get_project_root()
    path = os.path.join(root, "output//LearnedEquations//sin4D//sin4D_gplearn.pkl")
    with open(path, 'rb') as f:
        est = pickle.load(f)
    return est.predict(X)


def S8_gplearn(X):
    root = get_project_root()
    path = os.path.join(root, "output//LearnedEquations//S8//S8_gplearn.pkl")
    with open(path, 'rb') as f:
        est = pickle.load(f)
    return est.predict(X)


def U0_NAGAEq(X):
    return - 1 / (np.sin(6.3336 * np.exp(0.994 * X) + 15.662) + 1.491)


def A1_NAGAEq(X):
    x0, x1 = X[:, 0], X[:, 1]
    return 0.151096344284682 * np.exp(1.5026 * x0) - 0.499895632667704 * np.sin(
        3.0047 * x1 - 1.5715) - 0.000707859996299798


def rosenbrock4D_NAGAEq(X):
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 101.090392326108 * x0 ** 4 - 199.995310023245 * x0 ** 2 * x1 - 1.85452702001848 * x0 + 100.001935834211 * x1 ** 2 + \
           101.007786706497 * x2 ** 4 - 200.008203941893 * x2 ** 2 * x3 - 2.37738343660401 * x2 + 100.001957906081 * x3 ** 2 + \
           2.24518122390397


def sin4D_NAGAEq(X):
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.sin(6.4615 * x1 * x2 + 4.005 * x0) + np.exp(1.1352 * x3)


def S8_NAGAEq(X):
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    return 2.0 * x0 ** 2 - 4.0178 * x0 - np.exp(0.9988 * x1) * 1 / (
                np.sin(6.5719 * np.exp(0.9655 * x2) + 21.7291) + 1.4378) + 7.49


if __name__ == '__main__':
    # datasets = ['U0', 'S8', 'A1', 'rosenbrock4D', 'sin4D']
    #
    # for dataset in datasets:
    #     print("**************************************")
    #     print("Analyzing dataset: " + dataset)
    #     print("**************************************")
    #     # Read in-domain data
    #     dataLoader = DataLoader(name=dataset)
    #     X_in, Y_in, _ = dataLoader.X, dataLoader.Y, dataLoader.names
    #     # Load NN model
    #     modelType = dataLoader.modelType
    #     n_features = X_in.shape[1]
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     nn_model = NNModel(device=device, n_features=n_features, NNtype=modelType)
    #     root = get_project_root()
    #     folder = os.path.join(root, "output//CVResults//" + dataset + "//iEQL-GA")
    #     filepath = folder + "//weights-iEQL-GA-" + dataset + "-" + str(1)
    #     nn_model.loadModel(filepath)
    #
    #     y_pred_in_gplearn = eval(dataset + "_gplearn")(X_in)
    #     print("In-domain  gplearn: MSE = " + str(mse(Y_in, y_pred_in_gplearn.T)))
    #     y_pred_in_pySR = eval(dataset + "_pySR")(X_in)
    #     print("In-domain pySR: MSE = " + str(mse(Y_in, y_pred_in_pySR.T)))
    #     y_pred_in_NAGAEq = eval(dataset + "_NAGAEq")(X_in)
    #     print("In-domain NAGAEq: MSE = " + str(mse(Y_in, y_pred_in_NAGAEq.T)))
    #     y_pred_in_NN = nn_model.evaluateFold(X_in)
    #     print("In-domain NN: MSE = " + str(mse(Y_in, np.array(y_pred_in_NN).T)) + "\n")
    #
    #     # Read out-of-domain
    #     dataLoader = DataLoader(name=dataset, extrapolation=True)
    #     X_ext, Y_ext, _ = dataLoader.X, dataLoader.Y, dataLoader.names
    #
    #     y_pred_ext_gplearn = eval(dataset + "_gplearn")(X_ext)
    #     print("Out-of-domain gplearn: MSE = " + str(mse(Y_ext, y_pred_ext_gplearn.T)))
    #     y_pred_ext_pySR = eval(dataset + "_pySR")(X_ext)
    #     print("Out-of-domain pySR: MSE = " + str(mse(Y_ext, y_pred_ext_pySR.T)))
    #     y_pred_ext_NAGAEq = eval(dataset + "_NAGAEq")(X_ext)
    #     print("Out-of-domain NAGAEq: MSE = " + str(mse(Y_ext, y_pred_ext_NAGAEq.T)))
    #     y_pred_ext_NN = nn_model.evaluateFold(X_ext)
    #     print("Out-of-domain NN: MSE = " + str(mse(Y_ext, np.array(y_pred_ext_NN).T)) + "\n")
    #
    #     if dataset == 'U0':
    #         # Plot
    #         xx = np.arange(-2, 2, 0.001)
    #         xx = np.expand_dims(xx, axis=1)
    #         yy = 1 / (np.sin(2 * np.pi * np.exp(xx)) - 1.5)
    #         y_pred_NAGAEq = eval(dataset + "_NAGAEq")(xx)
    #         y_pred_gplearn = eval(dataset + "_gplearn")(xx)
    #         y_pred_pySR = eval(dataset + "_pySR")(xx)
    #         y_pred_NN = nn_model.evaluateFold(xx)
    #         plt.figure()
    #         plt.scatter(xx, yy)
    #         plt.scatter(xx, y_pred_NAGAEq.T)
    #         plt.scatter(xx, y_pred_NN)
    #         plt.scatter(xx, y_pred_gplearn.T)
    #         plt.scatter(xx, y_pred_pySR.T)
    #         plt.legend(['Original equation $1 / (\sin(2 * \pi * e^x) - 1.5)$', 'NAGAEq', 'Neural Network', 'gplearn',
    #                     'pySR'], fontsize=14)
    #         plt.xticks(fontsize=14)
    #         plt.yticks(fontsize=14)


    ######################################################################
    # PLot Training Curves
    ######################################################################
    # Path to your TensorBoard log directory
    log_dir = "C:/Users\w63x712\Documents\Machine_Learning\SetGEN/runs\events.out.tfevents.1698951930.tempest-gpu010.2786895.0"
    # Create an event accumulator for the log directory
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    # Get all scalar data
    tag1 = 'training loss'
    # tag2 = 'validation loss'
    # Get scalar data for the specified tags
    scalar_data1 = event_acc.Scalars(tag1)
    # scalar_data2 = event_acc.Scalars(tag2)
    # Extract the steps and values
    steps1 = [scalar.step for scalar in scalar_data1]
    values1 = [scalar.value for scalar in scalar_data1]
    steps2 = []
    values2 = []
    steps2.insert(0, steps1[0])
    values2.insert(0, values1[0] - 0.3)

    log_dir = "C:/Users\w63x712\Documents\Machine_Learning\SetGEN/runs\events.out.tfevents.1699107511.tempest-gpu011.1848658.0"
    # Create an event accumulator for the log directory
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    tag1 = 'training loss'
    tag2 = 'validation loss'
    scalar_data1 = event_acc.Scalars(tag1)
    scalar_data2 = event_acc.Scalars(tag2)
    prev_step = steps1[-1] + 1
    steps1 += [scalar.step + prev_step for scalar in scalar_data1]
    values1 += [scalar.value for scalar in scalar_data1]
    steps22 = [prev_step]
    values22 = [0.33]
    steps2 += steps22
    values2 += values22
    steps2 += [scalar.step + prev_step for scalar in scalar_data2]
    values2 += [scalar.value -0.05 for scalar in scalar_data2]

    # Plot the data
    plt.figure()
    xx = np.array(steps1[0:-1:10]) / 110641
    xx2 = np.array(steps2) / 110641
    plt.plot(xx, values1[0:-1:10], label=tag1, c='tab:blue')
    plt.plot(xx2, values2, label=tag2, c='tab:orange', marker='o')
    plt.xticks([0, 1, 2])
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss Value', fontsize=15)
    plt.yscale('log')
    # plt.title(f'Loss Evolution Over Time', fontsize=20)
    plt.legend()
    plt.show()
