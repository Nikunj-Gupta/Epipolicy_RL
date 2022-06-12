from sympy import sympify, expand, Integer, fraction
from numba.typed import List

from .summant import Summant, SummantType, multiply
from .fraction import Fraction, get_fraction_from

PREFIX = 'x'

def is_number(expr):
    return "numbers" in str(expr.func)

def is_variable(expr):
    return "Symbol" in str(expr.func)

def is_add(expr):
    return "Add" in str(expr.func)

def is_mul(expr):
    return "Mul" in str(expr.func)

def is_pow(expr):
    return "Pow" in str(expr.func)

# expr.func = Add
# expr.args = [expr1, expr2, ...]
# expr = expr1 + expr2 + ... + ...

class EquationParser:

    def __init__(self, raw_expr):
        self.encode = {}
        self.decode = {}
        normalized_expr = self.normalize(raw_expr)
        self.expr = sympify(normalized_expr, evaluate=False)

    def normalize(self, raw_expr):
        raw_expr = raw_expr.replace(" ", "")
        normalized_expr = ""
        is_found = False
        var = ""
        for c in raw_expr:
            if not is_found and c.isalpha():
                is_found = True
                var = c
            elif is_found:
                if c.isalnum() or c == '_':
                    var += c
                else:
                    if var not in self.encode:
                        self.encode[var] = PREFIX + str(len(self.encode))
                        self.decode[self.encode[var]] = var
                    normalized_expr += self.encode[var]
                    is_found = False
            if not is_found:
                normalized_expr += c
        if is_found:
            if var not in self.encode:
                self.encode[var] = PREFIX + str(len(self.encode))
                self.decode[self.encode[var]] = var
            normalized_expr += self.encode[var]
        return normalized_expr

    def get_summant(self, expr):
        summant = Summant()
        if is_variable(expr):
            summant.muls.append(self.decode[str(expr)])
        elif is_number(expr):
            summant.coef *= float(str(expr))
        elif is_pow(expr):
            base, pow = expr.args
            pow = int(str(pow))
            for i in range(pow):
                summant.muls.append(self.decode[str(base)])
        elif is_mul(expr):
            for arg in expr.args:
                summant = multiply(summant, self.get_summant(arg))
        return summant

    def get_summants(self, expr):
        summants = List.empty_list(SummantType)
        if is_add(expr):
            for arg in expr.args:
                summants.append(self.get_summant(arg))
        else:
            summants.append(self.get_summant(expr))
        return summants

    def get_fraction(self, expr):
        numer, denom = fraction(expr)
        # Some problem when fraction does not work immediately like a*b/c = (a*b/c, 1)
        while '/' in str(numer):
            numer, denom = fraction(numer)
        return get_fraction_from(self.get_summant(numer), self.get_summants(denom))

    def get_fractions(self, expanded_expr):
        if is_add(expanded_expr):
            return [self.get_fraction(arg) for arg in expanded_expr.args]
        else:
            return [self.get_fraction(expanded_expr)]

    def get_all_fractions(self):
        if is_add(self.expr):
            fracs = []
            for arg in self.expr.args:
                fracs += self.get_fractions(expand(arg))
            return fracs
        else:
            return self.get_fractions(expand(self.expr))
