# Code adapted from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales
# The "process_equation" method had bugs, they were solved, and the method was improved
# The "_generate_expr" method was modified to avoid certain combinations of operations

import numpy as np
import sympy
import sympy as sp
from collections import Counter
from .sympy_utils import simplify
from collections import OrderedDict
from sympy.calculus.util import AccumBounds
from sympy.parsing.sympy_parser import parse_expr
from .sympy_utils import remove_root_constant_terms, add_constants, remove_numeric_constants
from src.EquationLearning.models.utilities_expressions import check_forbidden_combination

CLEAR_SYMPY_CACHE_FREQ = 10000


class NotCorrectIndependentVariables(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class ValueErrorExpression(Exception):
    pass


class ImAccomulationBounds(Exception):
    pass


class InvalidPrefixExpression(Exception):
    pass


class Generator(object):
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Pow: "pow",
        sp.exp: "exp",
        sp.log: "ln",
        sp.Abs: 'abs',

        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",

        # Trigonometric Inverses
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",

        # Hyperbolic Functions
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",

    }

    OPERATORS = {
        # Elementary functions
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "pow": 2,
        "inv": 1,
        "pow2": 1,
        "pow3": 1,
        "pow4": 1,
        "pow5": 1,
        "sqrt": 1,
        "exp": 1,
        "ln": 1,
        "abs": 1,

        # Trigonometric Functions
        "sin": 1,
        "cos": 1,
        "tan": 1,

        # Trigonometric Inverses
        "asin": 1,
        "acos": 1,
        "atan": 1,

        # Hyperbolic Functions
        "sinh": 1,
        "cosh": 1,
        "tanh": 1,
        # "coth": 1,
    }
    operators = sorted(list(OPERATORS.keys()))
    constants = ["pi", "E"]

    def __init__(self, params):
        self.max_ops = params.max_ops
        self.max_len = params.max_len
        # self.positive = params.positive

        # parse operators with their weights

        ops = params.operators.split(",")
        ops = sorted([x.split(":") for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 1]
        ).astype(np.float64)
        self.bin_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 2]
        ).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # symbols / elements
        self.variables = OrderedDict({})
        for var in params.variables:
            self.variables[str(var)] = sp.Symbol(str(var), real=True, nonzero=True)
        self.var_symbols = list(self.variables)
        self.pos_dict = {x: idx for idx, x in enumerate(self.var_symbols)}
        self.placeholders = {"cm": sp.Symbol("cm", real=True, nonzero=True),
                             "ca": sp.Symbol("ca", real=True, nonzero=True)}
        assert 1 <= len(self.variables)
        # We do not no a priori how many coefficients an expression has, so to be on the same side we equal to two
        # times the maximum number of expressions
        self.coefficients = [f"{x}_{i}" for x in self.placeholders.keys() for i in range(2 * params.max_len)]
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(
                self.variables.items()
        ):
            assert k not in self.local_dict
            self.local_dict[k] = v

        digits = [str(i) for i in range(-3, abs(6))]
        self.words = (
                list(self.variables.keys())
                + [
                    x
                    for x in self.operators
                    if x not in ("pow2", "pow3", "pow4", "pow5", "sub", "inv")
                ]
                + digits
        )

        self.id2word = {i: s for i, s in enumerate(self.words, 4)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.word2id["E"] = 32
        self.id2word[32] = "E"
        # ADD Start and Finish
        self.word2id["P"] = 0
        self.word2id["S"] = 1
        self.word2id["F"] = 2
        self.id2word[1] = "S"
        self.id2word[2] = "F"

        # ADD Constant Placeholder
        self.word2id["c"] = 3
        self.id2word[3] = "c"

        assert len(set(self.word2id.values())) == len(self.word2id.values())
        assert len(set(self.id2word.values())) == len(self.id2word.values())

        # assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = self.return_rewrite_functions(params)

    @classmethod
    def return_local_dict(cls, variables=None):
        local_dict = {}
        for k, v in list(
                variables.items()
        ):
            assert k not in local_dict
            local_dict[k] = v
        return local_dict

    @classmethod
    def return_rewrite_functions(cls, params):
        r = [
            x for x in params.rewrite_functions.split(",") if x != ""
        ]
        assert len(r) == len(set(r))
        assert all(
            x in ["expand", "factor", "expand_log", "logcombine", "powsimp", "simplify"]
            for x in r
        )
        return r

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = [[0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)])]
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        return D

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, curr_leaves, rng):
        if curr_leaves:
            max_idxs = max([self.pos_dict[x] for x in curr_leaves]) + 1
        else:
            max_idxs = 0
        return [list(self.variables.keys())[rng.randint(low=0, high=min(max_idxs + 1, len(self.variables.keys())))]]

    def _generate_expr(self, nb_total_ops, rng):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        una_ops_copy, una_ops_probs_copy = self.una_ops.copy(), self.una_ops_probs.copy()
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)

            # Next operator, arity and position
            # arity = np.random.choice([1, 2], p=[1/3, 2/3])
            # free_pos = [i for i, x in enumerate(stack) if x is None]
            # which_None = np.random.choice(np.arange(0, len(free_pos)))
            # pos = free_pos[which_None]

            if arity == 1:
                una_ops, una_ops_probs = np.array(una_ops_copy), np.array(una_ops_probs_copy)
                op = rng.choice(una_ops, p=una_ops_probs/np.sum(una_ops_probs))
                # Remove the current operation from the list of options
                f_ops_inds = np.array([io for io, iop in enumerate(una_ops_copy) if iop != op])
                una_ops_copy, una_ops_probs_copy = np.array(una_ops_copy)[f_ops_inds], np.array(una_ops_probs_copy)[f_ops_inds]
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += (self.OPERATORS[op] - 1 - skipped)  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1  # update number of total leaves
            l_leaves += skipped  # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = (
                    stack[:pos]
                    + [op]
                    + [None for _ in range(self.OPERATORS[op])]
                    + stack[pos + 1:]
            )

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        leaves = []
        curr_leaves = set()
        for _ in range(t_leaves):
            new_element = self.get_leaf(curr_leaves, rng)
            leaves.append(new_element)
            curr_leaves.add(*new_element)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0

        new_stack = []
        i = 0
        while i < len(stack):
            if (stack[i:i + 3] == ['mul', 'x_1', 'x_1']) or (stack[i:i + 3] == ['div', 'x_1', 'x_1']):
                new_stack.append('x_1')
                i += 3
            else:
                new_stack.append(stack[i])
                i += 1

        # Explore the expression tree and check if there's a ramification with two operators that belong to the same
        # forbidden group
        inf = self.prefix_to_infix(new_stack, coefficients=self.coefficients, variables=self.variables)
        if check_forbidden_combination(2 * sp.sympify(inf)):
            new_stack = self._generate_expr(nb_total_ops, rng)

        return new_stack

    @classmethod
    def write_infix(cls, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        elif token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "rac":
            return f"({args[0]})**(1/({args[1]}))"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token == "pow4":
            return f"({args[0]})**4"
        elif token == "pow5":
            return f"({args[0]})**5"
        elif token in [
            "sign",
            "sqrt",
            "exp",
            "ln",
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "csc",
            "asin",
            "acos",
            "atan",
            "acot",
            "asec",
            "acsc",
            "sinh",
            "cosh",
            "tanh",
            "coth",
            "sech",
            "csch",
            "asinh",
            "acosh",
            "atanh",
            "acoth",
            "asech",
            "acsch",
        ]:
            return f"{token}({args[0]})"
        elif token == "derivative":
            return f"Derivative({args[0]},{args[1]})"
        elif token == "f":
            return f"f({args[0]})"
        elif token == "g":
            return f"g({args[0]},{args[1]})"
        elif token == "h":
            return f"h({args[0]},{args[1]},{args[2]})"
        elif token.startswith("INT"):
            return f"{token[-1]}{args[0]}"
        else:
            return token

    @classmethod
    def add_identifier_constants(cls, expr_list):
        curr = Counter()
        curr["cm"] = 0
        curr["ca"] = 0
        for i in range(len(expr_list)):
            if expr_list[i] == "cm":
                expr_list[i] = "cm_{}".format(curr["cm"])
                curr["cm"] += 1
            if expr_list[i] == "ca":
                expr_list[i] = "ca_{}".format(curr["ca"])
                curr["ca"] += 1
        return expr_list

    def return_constants(self, expr_list):
        curr = Counter()
        curr["cm"] = [x for x in expr_list if x[:3] == "cm_"]
        curr["ca"] = [x for x in expr_list if x[:3] == "ca_"]
        return curr

    @classmethod
    def _prefix_to_infix(cls, expr, coefficients=None, variables=None):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            # raise InvalidPrefixExpression("Empty prefix list.")
            expr = sp.sympify('2')
        t = expr[0]
        if t in cls.operators:
            args = []
            l1 = expr[1:]
            for _ in range(cls.OPERATORS[t]):  # Arity
                i1, l1 = cls._prefix_to_infix(l1, coefficients=coefficients, variables=variables)
                args.append(i1)
            return cls.write_infix(t, args), l1
        elif t in coefficients:
            return "{" + t + "}", expr[1:]
        elif (
                t in variables
                or t in cls.constants
                or t == "I"
        ):
            return t, expr[1:]
        else:  # INT
            val = expr[0]
            return str(val), expr[1:]

    def _prefix_to_edges(self, expr):
        t = expr[0][1]
        edges = []
        li = expr[1:]
        if t in self.operators:
            for _ in range(self.OPERATORS[t]):
                new_edge = [expr[0][0], li[0][0]]
                edges.append(new_edge)
                inner_edges, li = self._prefix_to_edges(li)
                edges.extend(inner_edges)
        return edges, li

    @classmethod
    def prefix_to_infix(cls, expr, coefficients=None, variables=None):
        """
        Prefix to infix conversion.
        """
        p, r = cls._prefix_to_infix(expr, coefficients=coefficients, variables=variables)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"

    @classmethod
    def rewrite_sympy_expr(cls, expr, rewrite_functions=None):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in rewrite_functions:
            if f == "expand":
                expr_rw = sp.expand(expr_rw)
            elif f == "factor":
                expr_rw = sp.factor(expr_rw)
            elif f == "expand_log":
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == "logcombine":
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == "powsimp":
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == "simplify":
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    @classmethod
    def infix_to_sympy(cls, infix, variables, rewrite_functions, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        try:
            expr = parse_expr(infix, evaluate=True, local_dict=cls.return_local_dict(variables))
        except ValueError:
            raise ImAccomulationBounds
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        if not no_rewrite:
            expr = cls.rewrite_sympy_expr(expr, rewrite_functions)
        return expr

    @classmethod
    def _sympy_to_prefix(cls, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        # Check if the expression is of the form Pow( -, 0.5) and change it to sqrt
        if op == 'pow':
            arg1 = expr.args[1]
            if np.abs(arg1) < 0.5:
                arg1 = 0.5 * np.sign(arg1)

            if arg1 == 0.5:
                expr = sp.sympify('sqrt(' + str(expr.args[0]) + ')')
            elif arg1 == -0.5:
                expr = 1 / sp.sympify('sqrt(' + str(expr.args[0]) + ')')
            else:
                # Clip exponent between -4 and 4
                # expr = expr.func(*[expr.args[0], np.clip(int(np.round(float(expr.args[1]))), -4, 4)])
                with sp.evaluate(False):
                    expr = expr.func(*[expr.args[0], float(np.clip(int(np.round(float(expr.args[1]))), -4, 4))])

        n_args = len(expr.args)

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += Generator.sympy_to_prefix(expr.args[i])

        return parse_list

    @classmethod
    def sympy_to_prefix(cls, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]  # self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Float):
            return [str(sp.Integer(expr))]
        elif isinstance(expr, sp.Rational):
            return (
                    ["div"] + [str(expr.p)] + [str(expr.q)]
            )  # self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        elif expr == sp.I:
            return ["I"]
        # SymPy operator
        for op_type, op_name in cls.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                # If expression has a negative exponent, rewrite it as an inversion
                if isinstance(expr, sp.Pow):
                    if expr.args[1] < 0:
                        expr = 1 / expr
                        if isinstance(expr, sp.Pow) and expr.args[1] == 1:
                            expr = expr.args[0]
                        # Find new op_name after division
                        if isinstance(expr, sp.Symbol):
                            return ["div"] + ["1"] + [str(expr)]
                        else:
                            for op_type2, op_name2 in cls.SYMPY_OPERATORS.items():
                                if isinstance(expr, op_type2):
                                    op_name = op_name2
                        return ["div"] + ["1"] + cls._sympy_to_prefix(op_name, expr)
                return cls._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def evaluate_Rational(self, expr):
        """
        Convert any rational operation into a number so that it can be evaluated
        """
        if not expr.args:
            return expr

        args = expr.args
        new_args = []

        for arg in args:
            if ('Rational' in str(arg.func)) or ('Half' in str(arg.func)) or ('One' in str(arg.func)) or \
                    ('Zero' in str(arg.func)) or ('Integer' in str(arg.func)):
                new_args.append(arg * 1.0)
            else:
                new_args.append(self.evaluate_Rational(arg))

        new_xp = expr.func(*new_args)
        return new_xp

    def process_equation(self, infix):
        f = self.infix_to_sympy(infix, self.variables, self.rewrite_functions)  # .evalf()
        if '(E)' in str(f):
            f = sp.sympify(str(f).replace('(E)', '(1)'))
        # Convert any rational operation into a number so that it can be evaluated
        f = sp.sympify(str(self.evaluate_Rational(sp.sympify(f)).evalf()).replace("**1.0", ""))

        symbols = set([str(x) for x in f.free_symbols])
        if not symbols:
            raise NotCorrectIndependentVariables()
            # return None, f"No variables in the expression, skip"
        for s in symbols:
            if not len(set(self.var_symbols[:self.pos_dict[s]]) & symbols) == len(self.var_symbols[:self.pos_dict[s]]):
                raise NotCorrectIndependentVariables()
                # return None, f"Variable {s} in the expressions, but not the one before"

        f = remove_root_constant_terms(f, list(self.variables.values()), 'add')
        f = remove_root_constant_terms(f, list(self.variables.values()), 'mul')
        f2 = add_constants(f, self.placeholders)
        f2 = sp.sympify(remove_numeric_constants(f2))

        if f2.is_number:
            f2 = sp.sympify('1')  # If the expression can be reduced to a constant, return an empty string
        elif not (isinstance(f2, sp.Add) and str(f2.args[0]) == 'ca'):
            f2 = self.placeholders["ca"] + f2 * self.placeholders["cm"]
        else:
            if len(f2.args) > 2:
                new_args = []
                for arg in f2.args:
                    if arg.args:
                        # Add the multiplicative placeholder to those terms that don't have it already
                        if arg.args[0] != self.placeholders["cm"]:
                            new_args.append(arg * self.placeholders["cm"])
                            continue
                    new_args.append(arg)
                f2 = f2.func(*new_args)

        if 'zoo' in str(f2) or 'oo' in str(f2):
            return sp.sympify('1')
        return f2

    def generate_equation(self, rng):
        """
        Generate pairs of (function, primitive).
        Start by generating a random function f, and use SymPy to compute F.
        """
        nb_ops = rng.randint(2, self.max_ops)
        f_expr = self._generate_expr(nb_ops, rng)
        infix = self.prefix_to_infix(f_expr, coefficients=self.coefficients, variables=self.variables)
        f = self.process_equation(infix)

        while len(str(f)) <= 2:  # Generate again in case the equation can be simplified to a constant after processing
            f_expr = self._generate_expr(nb_ops, rng)
            infix = self.prefix_to_infix(f_expr, coefficients=self.coefficients, variables=self.variables)
            f = self.process_equation(infix)

        fstr = str(f)
        # Only use basic trigonometric operations
        if 'sec' in fstr:
            fstr = fstr.replace('sec', 'cos')
        elif 'csc' in fstr:
            fstr = fstr.replace('csc', 'sin')
        elif 'cot' in fstr:
            fstr = fstr.replace('cot', 'tan')
        f = sp.sympify(fstr)

        f_prefix = self.sympy_to_prefix(f)
        # skip too long sequences
        if len(f_expr) + 2 > self.max_len:
            raise ValueErrorExpression("Sequence longer than max length")
            # return None, "Sequence longer than max length"

        # skip when the number of operators is too far from expected
        real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in f_expr)
        if real_nb_ops < nb_ops / 2:
            raise ValueErrorExpression("Too many operators")
            # return None, "Too many operators"

        if f == "0" or type(f) == str:
            raise ValueErrorExpression("Not a function")
            # return None, "Not a function"

        sy = f.free_symbols
        variables = set(map(str, sy)) - set(self.placeholders.keys())
        return f_prefix, variables, f
