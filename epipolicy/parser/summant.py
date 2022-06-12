import numba as nb
from numba.experimental import jitclass
from numba.types import string, ListType
from numba.typed import List

summant_spec = [
    ("coef", nb.float32),
    ("muls", ListType(string))
]

# c * x1 * x2 * .... * xk
@jitclass(summant_spec)
class Summant:

    def __init__(self):
        self.coef = 1.0
        self.muls = List.empty_list(string)

SummantType = Summant.class_type.instance_type

def multiply(s1, s2):
    res = Summant()
    res.coef = s1.coef * s2.coef
    for m in s1.muls:
        res.muls.append(m)
    for m in s2.muls:
        res.muls.append(m)
    return res

def get_signature(summant):
    summant.muls.sort()
    return "*".join(summant.muls)

def get_full_signature(summant):
    return str(summant.coef) + "*" + get_signature(summant)
