import sympy as sp
import numpy as np
import pandas as pd
from EquationLearning.utils import get_project_root


class FeynmanReader:

    def __init__(self, name):
        """Import a Feynman equation without units
        :arg name: Equation name. E.g., 'I.6.2'"""
        # Set directory path
        self.problem = name
        self.path = str(get_project_root().parent) + '/Feynman_with_units/'
        self.path_eq = self.path + name
        self.X, self.Y, self.names = self.read_data()
        self.expr = self.get_expression()

    def read_data(self):
        """Read data from CSV file"""
        data = pd.read_csv(self.path_eq, sep=' ')
        if data.iloc[:, -1].isnull().values.any():
            data = data.iloc[:, :-1]
        # Independent variables and system response
        Y = np.array(data.iloc[:, -1])
        X = np.array(data.iloc[:, :-1])
        # Set variable names
        names = ['x' + str(v) for v in range(X.shape[1])]
        return X, Y, names

    def get_expression(self) -> sp.Expr:
        """Read the equation CSV file and returns it
        :return expr: A sympy expression with variables x0, x1, ..."""
        eq_path = str(self.path) + '/FeynmanEquations.csv'
        eqs = pd.read_csv(eq_path, sep=',')
        # Find data regarding the specific Feynman problem
        eq_data = eqs[eqs['Filename'] == self.problem]
        expr = sp.sympify(eq_data['Formula'].iat[0])
        # Replace original variable names with x0, x1,...
        for i in range(self.X.shape[1]):
            vname = eq_data['v' + str(i + 1) + '_name'].iat[0]  # Get variable name of the i-th variable
            expr = expr.subs(sp.sympify(vname), sp.sympify('x' + str(i)))
        return sp.sympify(expr)


if __name__ == '__main__':
    reader = FeynmanReader(name='I.6.2')
