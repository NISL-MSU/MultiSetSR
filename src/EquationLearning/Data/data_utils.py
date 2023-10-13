# Code adapted from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales
# The sample_symbolic_constants function was modified

import random
from typing import Tuple
from src.EquationLearning.Data.dclasses import Equation


def group_symbolically_indetical_eqs(data, indexes_dict, disjoint_sets):
    for i, val in enumerate(data.eqs):
        if val.expr not in indexes_dict:
            indexes_dict[val.expr].append(i)
            disjoint_sets[i].append(i)
        else:
            first_key = indexes_dict[val.expr][0]
            disjoint_sets[first_key].append(i)
    return indexes_dict, disjoint_sets


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def return_dict_metadata_dummy_constant(metadata):
    dictd = {key: 0 for key in metadata.total_coefficients}
    for key in dictd.keys():
        if key[:2] == "cm":
            dictd[key] = 1
        elif key[:2] == "ca":
            dictd[key] = 0
        else:
            raise KeyError
    return dictd


def sample_symbolic_constants(eq: Equation, cfg=None) -> Tuple:
    """Given an equation, returns randomly sampled constants and dummy constants
    :param eq: an Equation.
    :param cfg: Used for specifying how many and in which range to sample constants. If None, consts equal to dummy_consts

    Returns:
      consts: 
      dummy_consts: 
    """
    dummy_consts = {const: 1 if const[:2] == "cm" else 0 for const in eq.coeff_dict.keys()}
    consts = dummy_consts.copy()
    if cfg:
        # Use at least 2 constants if there are 2 or more constants in the equation
        used_consts = random.randint(min(len(eq.coeff_dict), 2), min(len(eq.coeff_dict), cfg.num_constants))
        symbols_used = random.sample(set(eq.coeff_dict.keys()), used_consts)
        while 'ca_0' not in symbols_used:  # Make sure that the first additive constant is in the selection
            symbols_used = random.sample(set(eq.coeff_dict.keys()), used_consts)
        for si in set(eq.coeff_dict.keys()):
            # If the constant is used, set it to "ca" or "cm"
            if si in symbols_used:
                if si[0] == "c":
                    consts[si] = si
            else:
                if si[:2] == "ca":
                    consts[si] = 0
                elif si[:2] == "cm":
                    consts[si] = 1
    else:
        consts = dummy_consts
    return consts, dummy_consts


def bounded_operations():
    """Returns the bounded and double-bounded unary operations.
    Single-bounded operations are assigned their minimum or maximum admissible value.
    Double-bounded operations are assigned their minimum and maximum admissible values."""

    bounded = {'sqrt': ('min', 0.01),
               'log': ('min', 0.001),
               'exp': ('max', 6),
               'sinh': ('max', 6),
               'cosh': ('max', 6),
               'tanh': ('max', 6)}
    double_bounded = {'asin': [-0.999, 0.999],
                      'acos': [-0.999, 0.999]}
    op_with_singularities = {'tan': 50, 'div': 50}

    return bounded, double_bounded, op_with_singularities
