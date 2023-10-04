import sympy as sy
import numpy as np
from typing import Any
from torch import clamp, abs, nan_to_num, ones_like, clip_
from torch import sin, cos, mul, div, log, exp, sqrt


def get_function(f: str) -> (Any, int):
    """Retrieve the specified function and its corresponding complexity"""
    f_dict_torch = {
        "id": (func_id, 1),
        "sin": (sin, 1),
        "cos": (cos, 1),
        "+": (add, 1),
        "-": (sub, 1),
        "*": (mul, 1),
        "/n": (div, 1),
        "log": (log_reg, 2),
        "exp": (exp, 2),
        "sqrt": (sqrt_reg, 3),
        "log_reg": (log_reg, 2),
        "exp_reg": (exp_reg, 2),
        "sing_div": (sing_div, 2),
        "/": (div_reg, 2),
        "sqrt_reg": (sqrt_reg, 3),
        "square": (func_square, 1),
        "cube": (func_cube, 2),
    }
    return f_dict_torch[f]


def get_sym_function(f: str) -> (Any, int):
    """Retrieve the specified function and its corresponding complexity"""
    f_dict_sym = {
        "id": (func_id, func_id, 1),
        "sin": (sy.sin, np.sin, 1),
        "cos": (sy.cos, np.cos, 1),
        "+": (add, np.add, 1),
        "-": (sub, np.subtract, 1),
        "*": (sy.Symbol.__mul__, np.multiply, 2),
        "/": (sy.Symbol.__truediv__, np.divide, 2),
        "log": (log_reg_sy, np_log_reg, 1),
        "exp": (sy.exp, np_exp_reg, 1),
        "sqrt": (sy.sqrt, np_sqrt_reg, 1),
        "square": (func_square, np.square, 1),
    }
    return f_dict_sym[f]

# def log_reg(x):
#     return nan_to_num(log(x))


def np_sqrt_reg(x):
    y = np.sqrt(x)
    y[np.argwhere(np.isnan(y))] = -1000
    return y


def np_log_reg(x):
    y = np.log(x + 0.0001)
    y[np.argwhere(np.isnan(y))] = -1000
    return y


def np_exp_reg(x):
    y = np.exp(x)
    y[np.argwhere(np.isinf(y))] = 0
    return y


def func_cube(x):
    return x * x * x


def func_square(x):
    return x * x


def func_id(x):
    return x


def log_reg(x):
    return log(clamp(x + 1.0, min=0.001))


def log_reg_sy(x):
    return sy.log(x + 1)


def exp_reg(x):
    return exp(clamp(x, max=10.0)) - ones_like(x)


def sqrt_reg(x):
    return sqrt(abs(x) + 1e-8)


# def sqrt_reg_sy(x):
#     return sy.sqrt(sy.Abs(x))
#
#
# def exp_reg_sy(x):
#     return sy.exp(x) - 1


def sing_div(x):
    mask = abs(x) > 1e-2
    return (
        nan_to_num(div(1.0, abs(x) + 1e-6), posinf=1e5, neginf=-1e5)
        * mask
    )


def sing_div_sy(x):
    return 1 / x


def div_reg(x, y):
    return y.sign() * (div(x, clip_(y.abs(), min=1e-4)))


# def div_reg_sy(x, y):
#     return sy.Symbol.__truediv__(x, y)


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def square(x, y):
    return x ** 2 + y ** 2
