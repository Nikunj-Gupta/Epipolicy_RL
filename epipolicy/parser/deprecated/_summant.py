from numba.experimental import jitclass
from numba import jit
from numba.types import string, ListType
from numba.typed import List
from ..utility.singleton import *

stringListType = ListType(string)

summantSpec = [
    ('coef', nbFloat),
    ('muls', stringListType)
]

# c * x1 * x2 * .... * xk

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def multiply(s1, s2):
    res = Summant()
    res.coef = s1.coef * s2.coef
    for m in s1.muls:
        res.muls.append(m)
    for m in s2.muls:
        res.muls.append(m)
    return res

def getSignature(summant):
    summant.muls.sort()
    return '*'.join(summant.muls)

def getFullSignature(summant):
    return str(summant.coef) + "*" + getSignature(summant)

@jitclass(summantSpec)
class Summant:
    def __init__(self):
        self.coef = 1.0
        self.muls = List.empty_list(string)

summantType = Summant.class_type.instance_type
