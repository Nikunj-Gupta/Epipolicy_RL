import random, datetime
import numpy as np
from numba import njit
from scipy.stats import gamma

from .singleton import FLOAT_EPSILON

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3816335/
def infectivity(k, a=1, b=5):
    s1 = k*gamma.cdf(k, a, scale=b)+(k-2)*gamma.cdf(k-2, a, scale=b)-2*(k-1)*gamma.cdf(k-1, a, scale=b)
    s2 = a*b*(2*gamma.cdf(k-1, a+1, scale=b)-gamma.cdf(k-2, a+1, scale=b)-gamma.cdf(k, a+1, scale=b))
    return s1 + s2

Ws = [0]
while True:
    w = infectivity(len(Ws))
    if w > FLOAT_EPSILON:
        Ws.append(w)
    else:
        break

def get_infectivity(k):
    if 0 <= k and k < len(Ws):
        return Ws[k]
    return 0

@njit
def get_influence(multiplier):
    if multiplier <= 1:
        return min(1-multiplier, 1)
    else:
        return 1-1/multiplier

@njit
def quadratic_step_function(x):
    # Guarantee unit area under the curve
    return (-3*x+4)*x

@njit
def get_normalized_string(name):
    if len(name) == 0:
        return ""
    return name.replace(' ','').lower()

def parse_date(date_string):
    args = list(map(int, date_string.split('-')))
    return datetime.date(*args)

def get_consecutive_difference(array):
    if len(array) > 0:
        dif_array = [array[0]]
        for i in range(len(array)-1):
            dif_array.append(array[i+1]-array[i])
        return dif_array
    return []

def true_divide(a, b, out=None):
    if out is None:
        out = a
    return np.divide(a, b, out=out, where=b!=0)
