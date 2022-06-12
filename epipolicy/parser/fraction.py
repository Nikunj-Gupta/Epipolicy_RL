from numba.experimental import jitclass
from numba.types import ListType, string

from .summant import SummantType, get_signature, get_full_signature

fraction_spec = [
    ("numer", SummantType),
    ("denom", ListType(SummantType)),
    ("signature", string)
]

#c * x1 * x2 * .... * xk /
#(c * x1 * x2 * .... * xk + c * x1 * x2 * .... * xk + c * x1 * x2 * .... * xk + ... + c * x1 * x2 * .... * xk)

@jitclass(fraction_spec)
class Fraction:

    def __init__(self, numer, denom):
        self.numer = numer
        self.denom = denom

FractionType = Fraction.class_type.instance_type


def get_fraction_from(numer, denom):
    frac = Fraction(numer, denom)
    denom_scale = frac.denom[0].coef
    frac.numer.coef /= denom_scale
    for summant in frac.denom:
        summant.coef /= denom_scale
    frac.signature = get_signature(frac.numer) + "/"
    for summant in sorted(frac.denom, key=lambda x: get_signature(x)):
        frac.signature += "+ " + get_full_signature(summant)
    return frac
