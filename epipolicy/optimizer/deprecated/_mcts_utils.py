from ..utility.singleton import *
import psutil, math
from numba import jit

def computeNBytesPerNonRootNode(static):
    longBytes = 8*(1+static.nCompartments*static.nLocales*static.nGroups)
    uLongBytes = 8*1
    intBytes = 4*(9+static.nInterventions*MAX_CP_PER_ACTION+MAX_SIBLING_PAIRS*2+1)
    return longBytes + uLongBytes + intBytes

def computeMaxNode(static):
    perNonRootNode = computeNBytesPerNonRootNode(static)
    memory = dict(psutil.virtual_memory()._asdict())["free"]
    return int(math.floor(min(memory * MAX_RAM_PERCENTAGE / perNonRootNode, MAX_NODE)))

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def combine(c, n):
#     return compose(SPLIT_LEFT, c, SPLIT_RIGHT, n)
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def split(comb):
#     return decompose(comb, SPLIT_LEFT, SPLIT_RIGHT)
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def getFirst(nBits, v):
#     return v & ((1<<nBits)-1)
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def compose(left, leftV, right, rightV):
#     return (getFirst(left, leftV)<<right) | getFirst(right, rightV)
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def decompose(comb, left, right):
#     rightV = comb & ((1 << right) - 1)
#     leftV = comb >> right
#     return leftV, rightV
