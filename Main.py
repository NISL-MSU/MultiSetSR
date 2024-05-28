# from src.EquationLearning.SymbolicRegressor.MSSP import SymbolicRegressor
from src.EquationLearning.Trainer.TrainMultiSetTransformer import TransformerTrainer
from src.EquationLearning.Trainer.TrainMultiSetTransformerWPriors import TransformerTrainerwPrior


if __name__ == '__main__':
    # Dataset names: E1 - E9, CS1-CS4
    # regressor = SymbolicRegressor(dataset='E1')
    # regressor.get_skeleton()
    trainer = TransformerTrainer()
    trainer.fit()
    # trainer = TransformerTrainerwPrior()
    # trainer.fit()

