import random
import sympy as sp
from src.utils import *
from sympy.utilities.iterables import flatten
from sklearn.linear_model import LinearRegression
from src.EquationLearning.models.functions import *
from src.EquationLearning.Optimization.CFE import get_args, set_args
from src.EquationLearning.Data.GenerateDatasets import DataLoader


def random_operator():
    # Return a random Sympy operator from a list of common operators
    operators = [get_sym_function("+")[0], get_sym_function("-")[0], get_sym_function("*")[0]]
    return random.choice(operators)


def select_random_node(expr):
    # Helper function to select a random node in an expression excluding the root node
    while True:
        random_node = random.choice(expr.args)  # Choose a random node from the expression
        if random_node != expr:  # Exclude the root node
            return random_node


class GeneticProgramming:
    def __init__(self, X, Y, population_size, max_generations, p_crossover, p_mutate, initial_nodes, symbols_list):
        self.population_size = population_size
        self.max_generations = max_generations
        self.p_crossover = p_crossover
        self.p_mutate = p_mutate
        self.population = []
        self.generation = 0
        self.best_program = None
        self.best_fitness = float('inf')
        self.initial_nodes = initial_nodes
        self.symbols_list = symbols_list
        self. X, self.Y = X, Y

    def fitness_func(self, program):
        # Example fitness function: sum of absolute differences between program's expression and target
        expr = sp.sympify(program)  # Convert program's expression to Sympy expression
        # expr = self.symbols_list[0] - self.symbols_list[1] * self.symbols_list[2]

        # If it's a sum of terms or a single term
        if isinstance(expr, sp.Add):
            # Evaluate the values of each term
            term_eval = np.zeros((len(self.Y), len(expr.args)))
            for i, arg in enumerate(expr.args):
                term = sp.lambdify(flatten(self.initial_nodes), arg)
                term_eval[:, i] = term(*list(self.X.T))
        else:
            # Evaluate the values of the expression
            term = sp.lambdify(flatten(self.initial_nodes), expr)
            term_eval = term(*list(self.X.T))
            term_eval = np.reshape(term_eval, (len(term_eval), 1))

        # Perform linear regression
        reg = LinearRegression().fit(term_eval, self.Y)
        lr_coeff = reg.coef_
        # Apply estimated coefficients to expression
        if isinstance(expr, sp.Add):
            fitted_expr = expr.func(*[sp.sympify(arg * lr_coeff[i]) for i, arg in enumerate(expr.args)]) + reg.intercept_
        else:
            fitted_expr = expr * lr_coeff[0] + reg.intercept_

        expr = sp.lambdify(flatten(self.initial_nodes), fitted_expr)
        ypred = expr(*list(self.X.T))
        fitness_value = mse(ypred, self.Y)  # Evaluate fitness as absolute difference from target
        return fitness_value, fitted_expr

    def initialize_population(self):
        # Initialize the population with random programs
        for i in range(self.population_size):
            program = self.generate_random_expression(n_nodes=len(self.symbols_list))
            self.population.append(program)

    def generate_random_expression(self, n_nodes):
        # Generate a random Sympy expression with n_nodes as the maximum number of nodes
        random_args = []
        for _ in range(n_nodes):
            terminal = random.choice(self.initial_nodes)
            while terminal in random_args:  # Avoid repetition of nodes
                terminal = random.choice(self.initial_nodes)
            random_args.append(terminal)

        random_expr = random_args[0]
        for n in range(1, n_nodes):
            rand_operator = random_operator()
            random_expr = rand_operator(*[random_expr, random_args[n]])

        return str(random_expr)

    def evaluate_population(self):
        # Evaluate the fitness of each program in the population
        all_fitness = []
        for ip, program in enumerate(self.population):
            fitness, fitted_expr = self.fitness_func(program)
            self.population[ip] = fitted_expr
            all_fitness.append(fitness)
            if fitness < self.best_fitness:
                self.best_program = sp.simplify(self.population[ip])
                self.best_fitness = fitness
        return all_fitness

    def evolve(self):
        self.initialize_population()
        # Evolve the population through genetic operators (crossover and mutation)
        while self.generation < self.max_generations and self.best_fitness > 0.0001:
            all_fitness = self.evaluate_population()
            offspring = []
            while len(offspring) < self.population_size:
                # Select two parents through tournament selection
                parent1 = self.tournament_selection(all_fitness)
                parent2 = self.tournament_selection(all_fitness)

                # Perform crossover
                # child1, child2 = self.crossover(parent1, parent2)

                # Perform mutation
                child1 = self.mutate(parent1)
                child2 = self.mutate(parent2)

                offspring.append(child1)
                offspring.append(child2)
            print("Generation " + str(self.generation) + "\t Best fitness = " + str(self.best_fitness) +
                  "\t Best program so far = " + str(self.best_program))

            self.population = offspring
            self.generation += 1
        return self.best_program, self.best_fitness

    def tournament_selection(self, all_fitness):
        # Select a parent using tournament selection
        tournament_size = 20
        tournament_indices = random.sample(range(self.population_size), tournament_size)
        tournament_fitnesses = [all_fitness[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        return self.population[winner_index]

    def mutate(self, program):
        # Perform mutation by randomly modifying a subtree in the expression
        if random.random() < self.p_mutate:
            expr = sp.sympify(program)  # Convert program's expression to Sympy expression
            mutated_expr = self.mutate_expression(expr)  # Perform mutation
            mutated_program = str(mutated_expr)  # Convert mutated expression back to string
        else:
            mutated_program = program
        return mutated_program

    def random_terminal(self, prob_term=None):
        if prob_term is None:
            prob_term = random.random()
        # Return a random terminal node (variable or constant)
        if prob_term < 0.5:
            # 50% chance of selecting a variable
            return random.choice(self.initial_nodes)
        else:
            # 50% chance of selecting a constant
            return sp.sympify(random.randint(-25, 25))

    def mutate_expression(self, expr):
        # Recursive mutation of an expression by randomly changing operators or terminals
        if len(expr.args) == 0:
            # Base case: if expression is a leaf, mutate by randomly changing terminal
            # mutated_expr = self.random_terminal()
            mutated_expr = self.generate_random_expression(n_nodes=len(self.symbols_list))
        else:
            # Recursive case: mutate by randomly changing operator or subtree
            prob = random.random()
            if prob < 0.5:
                # xx = get_args(expr, return_symbols=True)
                if random.random() < 0.5:  # Mutate operator by randomly selecting a new operator
                    mutated_operator = random.choice([sp.Add, sp.Mul])
                    mutated_expr = mutated_operator(*[arg for arg in expr.args])
                else:  # Or choose a node and if it's a symbol, replace by another symbol; if it's a number, add noise
                    expr_args = get_args(expr, return_symbols=True)
                    arg_id = random.choice(np.arange(len(expr_args)))
                    if isinstance(expr_args[arg_id], sp.Symbol):
                        pass
                    if random.random() < 0.5:
                        expr_args[arg_id] = expr_args[arg_id] + np.random.randn()
                    else:
                        expr_args[arg_id] = expr_args[arg_id] * np.random.randn()
                    mutated_expr = set_args(xp=expr, target_args=expr_args, return_symbols=True)
            else:
                # Mutate subtree by recursively mutating each child
                # mutated_expr = expr.func(*[sp.sympify(self.mutate_expression(arg)) for arg in expr.args])
                mutated_expr = expr
        return str(mutated_expr)

    def crossover(self, program1, program2):
        # Perform crossover by exchanging genetic material (sub-trees) between two parent expressions
        if random.random() < self.p_crossover:
            expr1 = sp.sympify(program1)  # Convert program1's expression to Sympy expression
            expr2 = sp.sympify(program2)  # Convert program2's expression to Sympy expression

            if len(expr1.args) == 0 or len(expr2.args) == 0:
                # Base case: if either expression is a leaf, swap the entire subtree
                expr1_crossover = expr2
                expr2_crossover = expr1
            else:
                # Select a random node in each expression to exchange
                random_node_expr1 = select_random_node(expr1)
                random_node_expr2 = select_random_node(expr2)
                # Perform crossover by swapping the selected nodes
                expr1_crossover = expr1.replace(random_node_expr1, random_node_expr2)
                expr2_crossover = expr2.replace(random_node_expr2, random_node_expr1)

            # Convert crossovered expressions back to string representation
            crossovered_program1 = str(expr1_crossover)
            crossovered_program2 = str(expr2_crossover)
        else:
            crossovered_program1 = program1
            crossovered_program2 = program2

        return crossovered_program1, crossovered_program2


if __name__ == '__main__':

    # Define problem
    dataset = 'U1'
    dataLoader = DataLoader(name=dataset)
    x, y, var_names = dataLoader.X, dataLoader.Y, dataLoader.names
    # Create symbols and analyze each variable
    symbols = sp.symbols("{}:{}".format('x', 2))
    node1 = 1 / (sp.sin(6.2831 * symbols[0] + 3.1411) - 1.5051)
    node2 = symbols[1] ** 2
    node1_fun = sp.lambdify(flatten(symbols), node1)
    node1_values = node1_fun(*list(x.T))
    node2_fun = sp.lambdify(flatten(symbols), node2)
    node2_values = node2_fun(*list(x.T))
    node_values = np.array([node1_values, node2_values])
    nodes_list = sp.symbols("{}:{}".format('n', 2))

    # # Define problem
    # dataset = 'S8'
    # dataLoader = DataLoader(name=dataset)
    # x, y, var_names = dataLoader.X, dataLoader.Y, dataLoader.names
    # # Create symbols and analyze each variable
    # symbols = sp.symbols("{}:{}".format('x', 3))
    # node1 = (symbols[0] - 0.9965) ** 2
    # node2 = sp.exp(1.0097 * symbols[1])
    # node3 = 1/(sp.sin(6.3304 * sp.exp(0.9946 * symbols[2]) - 25.1749) - 1.4886)
    # node1_fun = sp.lambdify(flatten(symbols), node1)
    # node1_values = node1_fun(*list(x.T))
    # node2_fun = sp.lambdify(flatten(symbols), node2)
    # node2_values = node2_fun(*list(x.T))
    # node3_fun = sp.lambdify(flatten(symbols), node3)
    # node3_values = node3_fun(*list(x.T))
    # node_values = np.array([node1_values, node2_values, node3_values])
    # nodes_list = sp.symbols("{}:{}".format('n', 3))

    # GP
    GP = GeneticProgramming(population_size=300,
                            max_generations=50,
                            p_crossover=0.2,
                            p_mutate=0.5,
                            X=node_values, Y=y,
                            initial_nodes=nodes_list,
                            symbols_list=nodes_list
                            )
    equation, performance = GP.evolve()
    print("The approximated equation is: " + equation)
