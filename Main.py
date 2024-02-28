from src.EquationLearning.SymbolicRegressor.MSSP import SymbolicRegressor


if __name__ == '__main__':
    # Dataset names: E1 - E9, CS1-CS4
    regressor = SymbolicRegressor(dataset='E1')
    regressor.get_skeleton()
