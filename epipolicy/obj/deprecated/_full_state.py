from numba.experimental import jitclass
from ..utility.singleton import *
from .np_wrapper import Np3f, np3fType
import numpy as np
from numba import jit

fullStateSpec = [
    ('s', nbFloat[:,:,:]), #cxlxg
    ('gain', nbFloat[:,:,:]), #cxlxg
    ('lose', nbFloat[:,:,:]), #cxlxg
    ('trans', nbFloat[:,:,:,:]), #cxcxlxg
    ('c', nbFloat[:,:]) #ixl
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeFullState(static, flat):
    sz1 = static.nCompartments * static.nLocales * static.nGroups
    sz2 = (3+static.nCompartments)*sz1
    s = flat[:sz1].reshape((static.nCompartments, static.nLocales, static.nGroups))
    gain = flat[sz1:2*sz1].reshape((static.nCompartments, static.nLocales, static.nGroups))
    lose = flat[2*sz1:3*sz1].reshape((static.nCompartments, static.nLocales, static.nGroups))
    trans = flat[3*sz1:sz2].reshape((static.nCompartments, static.nCompartments, static.nLocales, static.nGroups))
    c = flat[sz2:].reshape((static.nInterventions, static.nLocales))
    return FullState(s, gain, lose, trans, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeInitialFullState(static, s):
    gain = np.zeros((static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    lose = np.zeros((static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    trans = np.zeros((static.nCompartments, static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    c = np.zeros((static.nInterventions, static.nLocales), dtype=npFloat)
    return FullState(s, gain, lose, trans, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def copyFullState(state):
    return FullState(state.s.copy(), state.gain.copy(), state.lose.copy(), state.trans.copy(), state.c.copy())

@jitclass(fullStateSpec)
class FullState:
    def __init__(self, s, gain, lose, trans, c):
        self.s = s
        self.gain = gain
        self.lose = lose
        self.trans = trans
        self.c = c
    def flatten(self):
        return np.concatenate((self.s.copy().reshape(-1), self.gain.copy().reshape(-1), self.lose.copy().reshape(-1), self.trans.copy().reshape(-1), self.c.copy().reshape(-1)))

fullStateType = FullState.class_type.instance_type
