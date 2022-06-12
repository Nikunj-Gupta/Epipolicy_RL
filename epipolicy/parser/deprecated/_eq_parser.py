from sympy import sympify, expand, Integer, fraction
from .parser_utils import *
from .frac import Fraction, makeFraction
from numba.typed import List
from .summant import Summant, summantType, multiply

#
# expr.func = Add
# expr.args = [expr1, expr2, ...]
# expr = expr1 + expr2 + ... + ...

class EqParser:
    def __init__(self, rawExpr):
        self.encode = {}
        self.decode = {}
        normalizedExpr = self.normalize(rawExpr)
        self.expr = sympify(normalizedExpr, evaluate=False)
    def normalize(self, rawExpr):
        rawExpr = rawExpr.replace(" ", "")
        normalizedExpr = ""
        isFound = False
        var = ""
        for c in rawExpr:
            if not isFound and c.isalpha():
                isFound = True
                var = c
            elif isFound:
                if c.isalnum() or c == '_':
                    var += c
                else:
                    if var not in self.encode:
                        self.encode[var] = PREFIX + str(len(self.encode))
                        self.decode[self.encode[var]] = var
                    normalizedExpr += self.encode[var]
                    isFound = False
            if not isFound:
                normalizedExpr += c
        if isFound:
            if var not in self.encode:
                self.encode[var] = PREFIX + str(len(self.encode))
                self.decode[self.encode[var]] = var
            normalizedExpr += self.encode[var]
        return normalizedExpr

    def getSummant(self, expr):
        summant = Summant()
        if isVariable(expr):
            summant.muls.append(self.decode[str(expr)])
        elif isNumber(expr):
            summant.coef *= float(str(expr))
        elif isPow(expr):
            base, pow = expr.args
            pow = int(str(pow))
            for i in range(pow):
                summant.muls.append(self.decode[str(base)])
        elif isMul(expr):
            for arg in expr.args:
                summant = multiply(summant, self.getSummant(arg))
        return summant

    def getSummants(self, expr):
        summants = List.empty_list(summantType)
        if isAdd(expr):
            for arg in expr.args:
                summants.append(self.getSummant(arg))
        else:
            summants.append(self.getSummant(expr))
        return summants

    def getFrac(self, expr):
        numer, denom = fraction(expr)
        # Some problem when fraction does not work immediately like a*b/c = (a*b/c, 1)
        while "/" in str(numer):
            numer, denom = fraction(numer)
        frac = makeFraction(self.getSummant(numer), self.getSummants(denom))
        return frac

    def getFracs(self, expandedExpr):
        if isAdd(expandedExpr):
            return [self.getFrac(arg) for arg in expandedExpr.args]
        else:
            return [self.getFrac(expandedExpr)]

    def getFractions(self):
        if isAdd(self.expr):
            fracs = []
            for arg in self.expr.args:
                fracs += self.getFracs(expand(arg))
            return fracs
        else:
            return self.getFracs(expand(self.expr))
