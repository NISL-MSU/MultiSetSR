import time
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count
from tqdm import trange
from scipy.stats import pearsonr
from EquationLearning.utils import *
from src.EquationLearning.models.utilities_expressions import *

import warnings

warnings.filterwarnings("ignore")


#############################################################################
# HELPER FUNCTIONS
#############################################################################
# NOTE: All these functions are located outside the class to improve parallelization
def simplify(expr, all_var=False):
    if not all_var:
        th = 0.0008
    else:
        th = 0.0005
    est_expr = expr.xreplace({n: n if abs(n) >= th else 0 for n in expr.atoms(sp.Number)})

    if not all_var:
        est_expr = norm_func(sympy.expand(est_expr))
        args = np.array(get_args(est_expr))
        args[np.abs(args) <= th] = 0
        est_expr = set_args(est_expr, list(args))
    else:
        est_expr = est_expr.xreplace({n: n if n != 1.0001 else 1 for n in expr.atoms(sp.Number)})
        est_expr = est_expr.xreplace({n: n if n != -1.0001 else -1 for n in expr.atoms(sp.Number)})
    merged = avoid_operations_between_constants(expr2skeleton(est_expr))

    return est_expr, merged


def norm_func(expr):
    """Normalize the outter multiplicative terms of the expression"""
    if isinstance(expr, sp.Add):
        try:
            expr = expr.xreplace({n: n.evalf() for n in expr.atoms(sp.Number)})
            num_args = []
            for arg in expr.args:
                if arg.is_number:
                    num_args.append(float(arg))
                elif isinstance(arg, sp.Mul):
                    for arg2 in arg.args:
                        if arg2.is_number:
                            num_args.append(float(arg2))
            max_arg = np.max(np.array(np.abs(num_args)))
            return expr/max_arg
        except:
            return expr
    else:
        return expr


def fitness_func(expr, X, Y, symbols_list):
    try:
        fs_lambda = sp.lambdify(sp.flatten(symbols_list), expr)
        ys = fs_lambda(*list(X))
        inconsistencies = np.isinf(ys) | np.isnan(ys) | (np.abs(ys) > 10 ** 14) | np.iscomplex(ys)
        if np.any(inconsistencies):
            return 100000
        if isinstance(ys, float):
            ys = np.repeat(ys, len(Y), axis=0)
        return -abs(pearsonr(Y, ys)[0])
    except:
        return 100000


def evolve_skeleton(all_fitness, population, fixed_population, generation, p_mutate, p_crossover, symbols):
    offspring = []
    # Elitism: Preserve the best and worst individuals
    elite_size = int(0.05 * len(population))  # 5% of the population
    elites = select_elites(all_fitness, elite_size, population)
    offspring.extend([norm_func(el) for el in elites])
    elites = select_elites(-all_fitness, elite_size, population)
    offspring.extend([norm_func(el) for el in elites])

    try:
        while len(offspring) < len(population):
            parent1, index1 = tournament_selection(all_fitness, population)
            parent2, _ = tournament_selection(all_fitness, population)

            # Perform crossover
            parent1, parent2 = crossover(parent1, parent2, p_crossover)

            # Perform mutation
            child1 = mutate(parent1, generation, p_mutate, fixed_population)
            child2 = mutate(parent2, generation, p_mutate, fixed_population)

            if len(symbols) == len([sy for sy in child1.free_symbols if 'x' in str(sy)]):
                offspring.append(norm_func(child1))
            if len(symbols) == len([sy for sy in child2.free_symbols if 'x' in str(sy)]):
                offspring.append(norm_func(child2))
    except ValueError:
        return None
    except TypeError:
        return None

    return offspring


def select_elites(all_fitness, elite_size, population):
    sorted_indices = sorted(range(len(all_fitness)), key=lambda idx: all_fitness[idx])[:elite_size]
    return [population[idx] for idx in sorted_indices]


def tournament_selection(all_fitness, population):
    sample_from = range(len(population))

    # Select a parent using tournament selection
    tournament_size = 3
    if len(sample_from) < tournament_size:
        tournament_size = len(sample_from) / 2
    tournament_indices = random.sample(sample_from, tournament_size)
    tournament_fitnesses = [all_fitness[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
    return population[winner_index], winner_index


def mutate(expr, generation, p_mutate, fixed_population):
    # Perform mutation by randomly modifying a subtree in the expression
    if random.random() < p_mutate:
        mutated_expr = mutate_expression(expr, generation, fixed_population)  # Perform mutation
    else:
        mutated_expr = expr
    return mutated_expr


def mutate_expression(expr, generation, fixed_population):
    # If fixed_population != 0, subtract it from the original expression to mutate just part of it
    init_expr = expr
    # expr = expr - fixed_population
    expr_args = get_args(expr, return_symbols=True)
    orig_args = expr_args.copy()
    arg_ids = np.random.choice(np.arange(len(expr_args)), size=np.random.randint(1, len(expr_args)))
    mutated_expr = expr
    arg_ids_symb = []
    for arg_id in arg_ids:
        ops = get_op(expr, arg_id)
        if isinstance(expr_args[arg_id], sp.Symbol):
            if not isinstance(ops, bool):
                if not any([('sqr' in op) or ('log' in op) for op in ops]):
                    arg_ids_symb.append(arg_id)
            else:
                arg_ids_symb.append(arg_id)  # Change the symbol later to prevent changing the order of the args
        else:
            in_trig = False     # Flag to limit offset values within trig functions between -2pi and 2pi
            in_mul = False      # Flag to avoid assigning ~0 values to multiplicative constants
            #  Choose a numerical node and add noise
            if expr_args[arg_id] == -1:
                mut_rate = 0
                expr_args[arg_id] = 1
            else:
                if len(ops) >= 2:
                    if (('sin' in ops[-2]) or ('cos' in ops[-2]) or ('tan' in ops[-2])) and 'Add' in ops[-1]\
                            or any([('sin' in op) or ('cos' in op) or ('tan' in op) for op in ops]):
                        in_trig = True
                    if 'Mul' in ops[-1]:
                        in_mul = True
                mut_rate = np.random.randn()
                if random.random() < 0.1:
                    if not in_trig:
                        mut_rate = 10 * mut_rate
                    if generation <= 50 and random.random() < 0.1:
                        if random.random() < 0.5 and not in_mul:
                            mut_rate = 0
                            expr_args[arg_id] = 0.00001
                        else:
                            mut_rate = 0
                            expr_args[arg_id] = 1.0001
                elif generation > 50 and random.random() < 0.1:  # self.generation > 100 and random.random() < 0.1:
                    if random.random() < 0.5 and not in_mul:
                        mut_rate = 0
                        expr_args[arg_id] = 0.00001
                    else:
                        mut_rate = 0
                        expr_args[arg_id] = 1.0001

            expr_args[arg_id] = expr_args[arg_id] + mut_rate
            if 'I' in str(expr_args[arg_id]):
                continue  # Avoid creating imaginary arguments
            if in_trig:
                expr_args[arg_id] = np.clip(expr_args[arg_id], - 2*np.pi, 2*np.pi)
            if expr_args[arg_id] == 1.0:
                expr_args[arg_id] = 1.0001
            [expr_args.remove(arg) for arg in expr_args.copy() if 'x' in str(arg)]
            mutated_expr = set_args(xp=expr, target_args=expr_args, return_symbols=False)
            expr_args = orig_args.copy()
    for arg_id in arg_ids_symb:
        # Change symbol's sign
        mutated_expr = change_sign_k_constant(mutated_expr, k=arg_id)[0]

    # mutated_expr += fixed_population  # After mutation, add fixed_population back

    while mutated_expr == init_expr:
        mutated_expr = mutate_expression(expr, generation, fixed_population)
    return mutated_expr


def crossover(expr1, expr2, p_crossover):
    # Perform crossover by swapping coefficient values. One point cross-over
    if random.random() < p_crossover:
        expr_args1 = get_args(expr1, return_symbols=False)
        expr_args2 = get_args(expr2, return_symbols=False)
        if len(expr_args1) == len(expr_args2) and len(expr_args1) > 2:
            for i in range(len(expr_args1)):
                if (expr_args1[i] == -1 and expr_args2[i] != -1) or (expr_args2[i] == -1 and expr_args1[i] != -1):
                    return expr1, expr2
            crossover_point = random.randint(1, len(expr_args1) - 1)
            new_expr_args1 = expr_args1[:crossover_point] + expr_args2[crossover_point:]
            new_expr_args2 = expr_args2[:crossover_point] + expr_args1[crossover_point:]
            expr1 = set_args(xp=expr1, target_args=new_expr_args1, return_symbols=False)
            expr2 = set_args(xp=expr2, target_args=new_expr_args2, return_symbols=False)
    return expr1, expr2


#############################################################################
# CLASS DEFINITION
#############################################################################
class GP:
    def __init__(self, X, Y, init_population, univ_sks, max_generations, p_crossover, p_mutate, bounds=None, verbose=False, all_var=False):
        self.all_var = all_var
        self.univ_sks = [add_constant_identifier(remove_coeffs(sk))[0] if ('c' in str(remove_coeffs(sk))) else sympy.sympify('c') * remove_coeffs(sk) for sk in univ_sks]
        self.max_generations = max_generations
        self.p_crossover = p_crossover
        self.p_mutate = p_mutate
        self.skeletons = [remove_coeffs(sk) for sk in init_population]
        self.population = []
        self.fixed_subpopulation = []
        self.generation = 0
        self.best_program = [None] * len(self.skeletons)
        self.best_fitness = [float('inf')] * len(self.skeletons)
        self.X, self.Y = X, Y
        self.verbose = verbose
        if bounds is None:
            self.bounds = [-30, 30]
        self.skeleton_inds = None
        self.temporal_subtract = []
        self.counters = []
        self.init_population()
        self.symbols_list = self.population[0][0].free_symbols
        self.symbols_list = [sp.sympify(st) for st in sorted([str(sym) for sym in self.symbols_list], key=lambda s: int(s[1:]))]
        self.stagnation_limit = 30  # Number of generations to wait before considering stagnation
        self.fixing_period = 40

    def assign_random_values(self, expr, explore=False):
        if explore:
            bounds_l, bounds_h = self.bounds[0], self.bounds[1]
        else:
            bounds_l, bounds_h = int(self.bounds[0] / 2), int(self.bounds[1] / 2)
        args = expr.free_symbols
        for arg in args:
            if 'c' in str(arg):
                op = get_op_constant(expr, str(arg))
                if op:
                    val = random.randint(bounds_l, bounds_h)
                elif isinstance(op, list):
                    val = random.randint(int(-5), int(5))
                else:
                    if ('sin' in op) or ('cos' in op) or ('tan' in op):
                        val = random.randint(int(-2*np.pi), int(2*np.pi))
                    elif 'exp' in op:
                        val = random.randint(int(-3), int(3))
                    else:
                        val = random.randint(bounds_l, bounds_h)
                if val == 0:
                    val = 0.1
                if val == 1:
                    val = 1.1
                if val == -1:
                    val = -1.1
                expr = expr.subs({arg: val})
        return expr

    def init_population(self):
        repetitions = 150
        self.population = []
        # self.skeletons = [element for element in self.population for _ in range(repetitions)]
        for sk in self.skeletons:
            self.population.append([self.assign_random_values(sk) for _ in range(repetitions)])
            self.fixed_subpopulation.append(0)
            self.temporal_subtract.append(0)
            self.counters.append(0)
        self.skeleton_inds = list(np.arange(len(self.skeletons)))

    def evolve(self):
        no_improvement_count = [0] * len(self.skeletons)  # Counter for generations with no significant improvement
        previous_best_fitness = [None] * len(self.skeletons)

        import sys
        import importlib

        if 'torch' in sys.modules:
            del sys.modules['torch']
            importlib.invalidate_caches()  # Deactivate pytorch to avoid problems with multiprocessing

        start = time.time()
        best_sk = 0
        for self.generation in trange(self.max_generations):

            # Add fixed sub-expressions back after a fixing period
            for sk in range(len(self.skeletons)):
                if self.fixed_subpopulation[sk] != 0:
                    self.counters[sk] += 1
                    if self.counters[sk] == self.fixing_period or self.generation == self.max_generations - 50:
                        if self.verbose:
                            print("'Un'-fixing previously fixed subexpression in skeleton ", self.skeleton_inds[sk])
                        for ie, exp in enumerate(self.population[sk]):
                            self.population[sk][ie] += self.fixed_subpopulation[sk]
                        self.temporal_subtract[sk] = 0
                        no_improvement_count[sk] = 0
                        self.fixed_subpopulation[sk] = 0

            # Evaluate fitness
            start1 = time.time()
            all_programs = [program for sublist in self.population for program in sublist]
            all_subtract = [subtract for subtract, pop in zip(self.temporal_subtract, self.population) for _ in pop]
            all_fitness = []
            batch_size = cpu_count() * 2
            # Process in batches
            for i in range(0, len(all_programs), batch_size):
                batch_programs = all_programs[i:i + batch_size]
                batch_subtract = all_subtract[i:i + batch_size]
                batch = [(program, self.X.T, self.Y - sub, self.symbols_list) for program, sub in zip(batch_programs, batch_subtract)]
                njobs = len(batch)
                batch_fitness = Parallel(n_jobs=njobs)(delayed(fitness_func)(*arg) for arg in batch)
                all_fitness.extend(batch_fitness)
            all_fitness = np.array(all_fitness)
            # Split the fitness results back into separate lists for each population
            split_indices = np.cumsum([len(pop) for pop in self.population[:-1]])
            fitness_per_skeleton = np.split(all_fitness, split_indices)
            end1 = time.time()
            if self.verbose:
                print(f'Time for evaluation: {end1 - start1:.6f} s')

            # Update the best program and fitness for each population
            for sk, fitness in enumerate(fitness_per_skeleton):
                min_fitness_idx = np.argmin(fitness)
                self.best_program[sk] = self.population[sk][min_fitness_idx]
                self.best_fitness[sk] = fitness[min_fitness_idx]

            # If necessary (and there's potential), extend the number of iterations
            if self.generation == self.max_generations - 1:
                if not any([bf > 0.998 for bf in self.best_fitness]) and any([bf > 0.9 for bf in self.best_fitness]):
                    self.max_generations = np.clip(self.max_generations + 50, 0, 500)

            # Early stopping if only one individual is performing good
            check_final = np.array(self.best_fitness) <= -0.99985
            if any(check_final) and np.sum(np.array(self.best_fitness) <= -0.995) == 1 and \
                    ((self.generation > 250 and self.all_var) or (self.generation > 50 and not self.all_var)):
                r_ind = np.where(check_final)[0][0]
                if self.counters[r_ind] == 0:
                    res_program, res_skeleton = simplify(self.best_program[r_ind], all_var=self.all_var)
                    return res_skeleton, np.abs(self.best_fitness[r_ind]), res_program
            elif any(check_final) and self.generation > 150:
                break

            # Select (to remove) combinations that haven't been improving sufficiently well
            rem_inds = []
            # if self.generation == 50 and any(np.abs(np.array(self.best_fitness)) >= 0.8):
            #     rem_inds = np.where(np.abs(np.array(self.best_fitness)) < 0.8)[0]
            # elif self.generation == 50 and any(np.abs(np.array(self.best_fitness)) < 0.9):
            #     rem_inds = np.where(np.abs(np.array(self.best_fitness)) < 0.9)[0]
            if self.generation == 150 and any(np.abs(np.array(self.best_fitness)) >= 0.95):
                rem_inds = np.where(np.abs(np.array(self.best_fitness)) < 0.95)[0]

            # Select (to remove) combinations that violate the original skeletons
            rem_inds = list(rem_inds)
            for isk, skk in enumerate(self.skeletons):
                univ_sks = get_skeletons(skk, [str(ss) for ss in self.symbols_list])
                if any([sk not in self.univ_sks for sk in [add_constant_identifier(remove_coeffs(sk))[0] if ('c' in str(remove_coeffs(sk))) else sympy.sympify('c') * remove_coeffs(sk) for sk in univ_sks]]):
                    rem_inds.append(isk)

            # REMOVE selected combinations
            rem_inds = np.unique(np.array(rem_inds))
            rem_inds = [ind for ind in rem_inds if self.counters[ind] == 0]  # If it's not currently under a fixing process
            if len(rem_inds) > 0:
                self.population = [item for idx, item in enumerate(self.population) if idx not in rem_inds]
                self.fixed_subpopulation = [item for idx, item in enumerate(self.fixed_subpopulation) if idx not in rem_inds]
                self.temporal_subtract = [item for idx, item in enumerate(self.temporal_subtract) if idx not in rem_inds]
                self.skeletons = [item for idx, item in enumerate(self.skeletons) if idx not in rem_inds]
                self.counters = [item for idx, item in enumerate(self.counters) if idx not in rem_inds]
                self.skeleton_inds = [item for idx, item in enumerate(self.skeleton_inds) if idx not in rem_inds]
                self.best_fitness = [item for idx, item in enumerate(self.best_fitness) if idx not in rem_inds]
                self.best_program = [item for idx, item in enumerate(self.best_program) if idx not in rem_inds]
                fitness_per_skeleton = [item for idx, item in enumerate(fitness_per_skeleton) if idx not in rem_inds]
                no_improvement_count = [item for idx, item in enumerate(no_improvement_count) if idx not in rem_inds]
                previous_best_fitness = [item for idx, item in enumerate(previous_best_fitness) if idx not in rem_inds]
            if len(self.skeletons) == 0:
                return None
            elif len(self.skeletons) == 1 and (not self.all_var):
                res_program, res_skeleton = simplify(self.best_program[0], all_var=self.all_var)
                return res_skeleton, np.abs(self.best_fitness[0]), res_program

            # Evolve populations
            results = []
            varpersk = [[sy for sy in sk[0].free_symbols if 'x' in str(sy)] for sk in self.population]
            # Process in batches
            for i in range(0, len(fitness_per_skeleton), batch_size):
                # batch = args[i:i + batch_size]
                batch_fitness = fitness_per_skeleton[i:i + batch_size]
                batch_population = self.population[i:i + batch_size]
                batch_fixed = self.fixed_subpopulation[i:i + batch_size]
                batch_vars = varpersk[i:i + batch_size]
                batch = [(f, p, fxd, self.generation, self.p_mutate, self.p_crossover, symbs)
                         for f, p, fxd, symbs in zip(batch_fitness, batch_population, batch_fixed, batch_vars)]
                batch_results = Parallel(n_jobs=len(batch))(delayed(evolve_skeleton)(*arg) for arg in batch)
                results.extend([bb for bb in batch_results if bb is not None])

            # Collect results from parallel execution
            for sk, offspring in enumerate(results):
                # fitness_per_skeleton[sk] = all_fitness
                self.population[sk] = offspring
                if self.verbose:
                    print("Generation " + str(self.generation) + "\t Skeleton Ind = " + str(self.skeleton_inds[sk]) +
                          "\t Best fitness = " + str(self.best_fitness[sk]) +
                          "\t Best program so far = " + str(self.best_program[sk]))
            end2 = time.time()
            if self.verbose:
                print(f'Time taken per generation: {end2 - start1:.6f} s')
            best_sk = np.argmin(np.array([bf if (bf != np.nan and self.fixed_subpopulation[ibf] == 0) else 0 for ibf, bf in enumerate(self.best_fitness)]))

            # Stagnation detection
            for sk in range(len(self.skeletons)):
                if previous_best_fitness[sk] is None or (abs(self.best_fitness[sk]) - abs(previous_best_fitness[sk])) > 0.0001:
                    previous_best_fitness[sk] = self.best_fitness[sk]
                    no_improvement_count[sk] = 0
                else:
                    no_improvement_count[sk] += 1
                if no_improvement_count[sk] >= self.stagnation_limit and self.fixed_subpopulation[sk] == 0 and \
                        abs(self.best_fitness[sk]) < 0.99:
                    if self.verbose:
                        print(f"Stagnation detected at skeleton {self.skeleton_inds[sk]}. Injecting diversity.")
                    self.inject_diversity(sk)
                    no_improvement_count[sk] = 0  # Reset the counter after injecting diversity

            if self.verbose:
                print('******************************')
                print("Generation " + str(self.generation) + "\t Best Ind = " +
                      str(self.skeleton_inds[best_sk]) + "\t Fitness = " + str(np.round(self.best_fitness[best_sk], 6)) +
                      "\t Best comb. = " + str(self.skeletons[best_sk]))
                print("\t Best program so far = " + str(self.best_program[best_sk]))
                print('******************************')

        end = time.time()
        elapsed = end - start
        if self.verbose:
            print(f'Time taken: {elapsed/60:.6f} minutes')

        # If more than one solution has corr val ~1, count nodes and select as best the solution the shortest one
        best_inds = np.where((abs(np.array(self.best_fitness)) > 0.9995) & (abs(np.array(self.best_fitness)) <= 1) &
                             (self.fixed_subpopulation == 0))[0]
        if len(best_inds) > 1:
            best_skeletons = [self.skeletons[bind] for bind in best_inds]  # [simplify(self.best_program[skl])[1] for skl in best_inds]
            best_skeletons_correct = []
            for isk, skk in enumerate(best_skeletons):
                univ_sks = get_skeletons(skk, [str(ss) for ss in self.symbols_list])
                if not any([sk not in self.univ_sks for sk in [
                    add_constant_identifier(remove_coeffs(sk))[0] if ('c' in str(remove_coeffs(sk))) else sympy.sympify(
                            'c') * remove_coeffs(sk) for sk in univ_sks]]):
                    best_skeletons_correct.append(skk)
            print(best_skeletons_correct)
            min_cands = np.argsort(np.array([count_nodes(expr) for expr in best_skeletons_correct]))
            min_cands = best_inds[min_cands]
        else:
            min_cands = np.argsort(np.array([bf if (bf != np.nan and self.fixed_subpopulation[ibf] == 0) else 0 for ibf, bf in enumerate(self.best_fitness)]))

        # Return skeletons that contain all variables
        ii = 0
        res_program, res_skeleton = simplify(self.best_program[min_cands[ii]], all_var=self.all_var)
        while len(self.symbols_list) != len([sy for sy in res_program.free_symbols if 'x' in str(sy)]):
            ii += 1
            res_program, res_skeleton = simplify(self.best_program[min_cands[ii]], all_var=self.all_var)
            if ii == len(min_cands):
                return None

        if 'x' not in str(res_skeleton) or str(res_skeleton) == 'c':
            return None
        return res_skeleton, np.abs(self.best_fitness[min_cands[ii]]), res_program

    def inject_diversity_simple(self, sk):
        # Replace a portion of the population with new random individuals
        # self.p_crossover = 0.5
        num_new_individuals = int(0.2 * len(self.population[sk]))  # Inject new individuals
        for _ in range(num_new_individuals):
            new_individual = self.assign_random_values(self.skeletons[sk], explore=True)
            self.population[sk][random.randint(0, len(self.population[sk]) - 1)] = new_individual

    def inject_diversity(self, sk):
        # Select best performing individual from skeleton sk and find besst independent term
        program = self.best_program[sk]
        best_subprogram, best_fitness, best_sub_sk = None, 0, None
        # self.p_crossover = 0.5
        if isinstance(program, sp.Add) and self.fixed_subpopulation[sk] == 0 and self.counters[sk] == 0:
            # self.stagnation_limit = 30
            sub_skeletons = self.skeletons[sk].args
            for arg in program.args:
                fitness = fitness_func(arg, self.X.T, self.Y, self.symbols_list)
                if fitness < best_fitness:
                    best_subprogram, best_fitness = arg, fitness
                    # Find to which subs-keleton this expression corresponds to and obtain constant values
                    for sub_sk in sub_skeletons:
                        try:
                            curr_sk = expr2skeleton(best_subprogram)
                            if str(remove_constant_identifier(sub_sk)) == str(curr_sk):
                                best_sub_sk = curr_sk
                                break
                        except TypeError:
                            if str(sub_sk) == str(best_subprogram):
                                best_sub_sk = best_subprogram
                                break
            # Remove constant value from the fixed program
            if isinstance(best_subprogram, sp.Mul):
                const = 1
                for arg in best_subprogram.args:
                    if arg.is_number:
                        const = arg
                best_subprogram /= const
            # Fix the values of the best subprogram
            for ie, exp in enumerate(self.population[sk]):
                new_args = []
                for arg in exp.args:
                    try:
                        c_sk = expr2skeleton(arg)
                        # if str(c_sk) == str(best_sub_sk):
                        #     new_args.append(best_subprogram)
                        # else:
                        if str(c_sk) != str(best_sub_sk):
                            new_args.append(self.assign_random_values(add_constant_identifier(c_sk)[0], explore=True))
                    except TypeError:
                        new_args.append(arg)
                self.population[sk][ie] = self.population[sk][ie].func(*new_args)
            self.fixed_subpopulation[sk] = best_subprogram

            # Evaluate fixed subprogram
            fs_lambda = sp.lambdify(sp.flatten(self.symbols_list), best_subprogram)
            self.temporal_subtract[sk] = fs_lambda(*list(self.X.T))

        # elif isinstance(program, sp.Add) and self.fixed_subpopulation[sk] != 0:
        #     num_new_individuals = int(0.2 * len(self.population[sk]))  # Inject new individuals
        #     new_inds = np.random.choice(np.arange(len(self.population[sk])), num_new_individuals, replace=False)
        #     best_sub_sk = expr2skeleton(self.fixed_subpopulation[sk])
        #     for ie in new_inds:
        #         new_args = []
        #         if isinstance(self.population[sk][ie], sp.Add):
        #             for arg in self.population[sk][ie].args:
        #                 try:
        #                     c_sk = expr2skeleton(arg)
        #                     if str(c_sk) == str(best_sub_sk):
        #                         new_args.append(self.fixed_subpopulation[sk])
        #                     else:
        #                         new_args.append(self.assign_random_values(add_constant_identifier(c_sk)[0], explore=True))
        #                 except TypeError:
        #                     new_args.append(arg)
        #             self.population[sk][ie] = self.population[sk][ie].func(*new_args)

        else:
            self.inject_diversity_simple(sk)
