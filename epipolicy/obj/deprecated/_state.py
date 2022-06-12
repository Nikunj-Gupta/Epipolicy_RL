# from .. import coo_matrix
from numba.experimental import jitclass
from ..utility.singleton import *
import numpy as np
from numba import jit

stateSpec = [
    ('s', nbFloat[:,:,:]),
    ('c', nbFloat[:,:]) #ixl
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeStateFrom(fullState):
    return State(fullState.s, fullState.c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeState(static, flat):
    sz1 = static.nCompartments * static.nLocales * static.nGroups
    s = flat[:sz1].reshape((static.nCompartments, static.nLocales, static.nGroups))
    c = flat[sz1:].reshape((static.nInterventions, static.nLocales))
    return State(s, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeInitialState(static, s):
    return State(s, np.zeros((static.nInterventions, static.nLocales), dtype=npFloat))

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def copyState(state):
    return State(state.s.copy(), state.c.copy())

@jitclass(stateSpec)
class State:
    def __init__(self, s, c):
        self.s = s
        self.c = c
    def flatten(self):
        return np.concatenate((self.s.copy().reshape(-1), self.c.copy().reshape(-1)))

stateType = State.class_type.instance_type
