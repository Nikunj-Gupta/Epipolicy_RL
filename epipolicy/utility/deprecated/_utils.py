import numpy as np
from numba import jit, njit
from scipy.stats import gamma
from .singleton import *
import random, datetime

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3816335/
def infectivity(k, a=1, b=5):
    s1 = k*gamma.cdf(k, a, scale=b)+(k-2)*gamma.cdf(k-2, a, scale=b)-2*(k-1)*gamma.cdf(k-1, a, scale=b)
    s2 = a*b*(2*gamma.cdf(k-1, a+1, scale=b)-gamma.cdf(k-2, a+1, scale=b)-gamma.cdf(k, a+1, scale=b))
    return s1 + s2

Ws = [0]
while True:
    w = infectivity(len(Ws))
    if w > EPSILON:
        Ws.append(w)
    else:
        break

def getInfectivity(k):
    if 0 <= k and k < len(Ws):
        return Ws[k]
    return 0

def getInfluence(multiplier):
    if multiplier <= 1:
        return min(1-multiplier, 1)
    else:
        return 1-1/multiplier

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def linearStepFunction(x):
    # Not guarantee unit area under the curve
    return x

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def quadraticStepFunction(x):
    # Guarantee unit area under the curve
    return -3*x*x+4*x

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def logisticStepFunction(x, k):
    # Not guarantee unit area under the curve
    return 2/(1+np.exp(-k*x))-1

def parseDate(strDate):
    args = list(map(int, strDate.split('-')))
    return datetime.date(*args)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getChoice(weights):
    s = np.sum(weights)
    percentage = weights / s
    v = random.random()
    curLim = 0
    for i, per in enumerate(percentage):
        if v < curLim:
            return i
        curLim += per
    return 0

def findMatchingBracket(s, st):
    cnt = 1
    for i in range(st, len(s)):
        if s[i] == '(':
            cnt += 1
        elif s[i] == ')':
            cnt -= 1
            if cnt == 0:
                return i
    return -1

def subInject(func, search, insert, addParams):
    start = 0
    while True:
        pos = func.find(search, start)
        if pos < 0:
            break
        nextPos = findMatchingBracket(func, pos + len(search))
        dotPos = func.find('.', pos)
        func = func[:pos] + insert + func[dotPos:nextPos] + addParams + func[nextPos:]
        start = pos + 1 + len(insert)
    return func

def inject(index, func, fn):
    simFunctionNames = ['select', 'apply', 'move', 'add']
    name = "self"
    addParams = ", difParameter, itvId"
    f1 = "def {}(cp, locales)".format(fn)
    f2 = "def {}(cp)".format(fn)
    pos = max(func.find(f1), func.find(f2))
    if pos >= 0:
        nextPos = func.find(")", pos + 1)
        func = func[:pos+4+len(fn)] + str(index) + "({}, ".format(name) + func[pos+4+len(fn)+len(str(index)):nextPos] + addParams + func[nextPos:]
        for simName in simFunctionNames:
            func = subInject(func, "sim.{}(".format(simName), name, addParams)
        content = func[func.find(":")+1:].strip()
        if len(content) == 0:
            # Empty function, we just need to add return function to make it valid
            if fn == "effect":
                func = func.strip() + "\n\treturn"
            elif fn == "cost":
                func = func.strip() + "\n\treturn 0"
        return "global {}{}\n".format(fn, index) + func
    return None

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def normalize(name):
    return name.replace(' ','').lower()
