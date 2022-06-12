from numba.experimental import jitclass
from numba.types import ListType, string
from numba.typed import List
from ..utility.singleton import nbFloat
from .summant import Summant, summantType, getSignature, getFullSignature

summantListType = ListType(summantType)

fractionSpec = [
    ('numer', summantType),
    ('denom', summantListType),
    ('signature', string)
]

#c * x1 * x2 * .... * xk /
#(c * x1 * x2 * .... * xk + c * x1 * x2 * .... * xk + c * x1 * x2 * .... * xk + ... + c * x1 * x2 * .... * xk)

def makeFraction(numer, denom):
    frac = Fraction(numer, denom)
    denomScale = frac.denom[0].coef
    frac.numer.coef /= denomScale
    for summant in frac.denom:
        summant.coef /= denomScale
    frac.signature = getSignature(frac.numer) + "/"
    for summant in sorted(frac.denom, key=lambda x: getSignature(x)):
        frac.signature += "+ " + getFullSignature(summant)
    return frac

@jitclass(fractionSpec)
class Fraction:
    def __init__(self, numer, denom):
        self.numer = numer
        self.denom = denom
    def getCoef(self):
        return self.numer.coef


fractionType = Fraction.class_type.instance_type
