from EquationLearning.SymbolicRegressor.SetGAP import SetGAP
from EquationLearning.SymbolicRegressor.MSSP import *


if __name__ == '__main__':
    ###########################################
    # Import data
    ###########################################
    import torch
    from EquationLearning.models.NNModel import NNModel

    datasetNames = ['E6']
    noise = 0
    noise_name = ''
    if noise > 0:
        noise_name = '_noise-' + str(noise)

    for datasetName in datasetNames:
        if datasetName == 'E10':
            datasetName = 'CS1'
        elif datasetName == 'E11':
            datasetName = 'CS2'
        elif datasetName == 'E12':
            datasetName = 'CS3'
        elif datasetName == 'E13':
            datasetName = 'CS4'
        data_loader = DataLoader(name=datasetName, noise=noise)
        data = data_loader.dataset

        ###########################################
        # Define NN and load weights
        ###########################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        folder = os.path.join(get_project_root(), "EquationLearning//saved_models//saved_NNs//" + datasetName)
        filepath = folder + "//weights-NN-" + datasetName + noise_name
        nn_model = None
        if os.path.exists(filepath.replace("weights", "NNModel") + '.pth'):
            # If this file exists, it means we saved the whole model
            network = torch.load(filepath.replace("weights", "NNModel") + '.pth', map_location=device)
            nn_model = NNModel(device=device, n_features=data.n_features, loaded_NN=network)
        elif os.path.exists(filepath):
            # If this file exists, initiate a model and load the weights
            nn_model = NNModel(device=device, n_features=data.n_features, NNtype=data_loader.modelType)
            nn_model.loadModel(filepath)
        else:
            # If neither files exist, we haven't trained a NN for this problem yet
            if data.n_features > 1:
                sys.exit("We haven't trained a NN for this problem yet. Use the TrainNNModel.py file first.")

        ###########################################
        # Get Estimated Multivariate Expressions
        ###########################################
        regressor = SetGAP(dataset=data, bb_model=nn_model, n_candidates=2)
        results = regressor.run()
