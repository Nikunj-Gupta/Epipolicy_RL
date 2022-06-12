from ..utility.singleton import *
from ..utility.utils import quadraticStepFunction
from numba.experimental import jitclass
from numba.types import ListType, DictType
from numba.typed import List, Dict
from ..sparse.coo import *
from .np_wrapper import Np2f, np2fType, Np3f, np3fType, listNp2fType
from numba import jit
import numpy as np

parameterSpec = [
    ('csr', listNp2fType),
    ('p', nbFloat[:,:,:,:]),
    ('f', nbFloat[:,:,:]),
    ('ft', nbFloat[:,:,:,:]),
    ('cc', d3CooType),
    ('c', nbFloat[:,:])
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeParameter(static, difParameter):
    csr = List.empty_list(np2fType)
    for i in range(static.nModes):
        csrData = np.zeros((static.nGroups, len(static.csr[i].data)), dtype=npFloat)
        for g in range(static.nGroups):
            coo = copyCooMatrix(static.coo[i][g]).multiply(difParameter.coo[i][g])
            csrData[g] = getCsrMatrix(getNormalizedMatrix(static.coo[i][g], coo)).data
        csr.append(Np2f(csrData))

    f = static.normalizeFacilityMatrix(static.P.f*difParameter.f)
    ft = static.normalizeContactMatrix(static.P.ft*difParameter.ft)

    return Parameter(csr, static.P.p*difParameter.p, f, ft, difParameter.cc, difParameter.c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getGapParameter(preP, curP):
    csr = List.empty_list(np2fType)
    for i in range(len(curP.csr)):
        csr.append(Np2f(curP.csr[i].data-preP.csr[i].data))
    p = curP.p - preP.p
    f = curP.f - preP.f
    ft = curP.ft - preP.ft
    #cc = curP.cc - preP.cc
    cc = addD3(scaleD3(copyD3Coo(preP.cc), -1), curP.cc)
    c = curP.c - preP.c

    return Parameter(csr, p, f, ft, cc, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def interpolateParameter(curP, gapP, ratio):
    fx = quadraticStepFunction(ratio)
    csr = List.empty_list(np2fType)
    for i in range(len(curP.csr)):
        csr.append(Np2f(curP.csr[i].data + gapP.csr[i].data*fx))
    p = curP.p + gapP.p*fx
    f = curP.f + gapP.f*fx
    ft = curP.ft + gapP.ft*fx
    cc = addD3(scaleD3(copyD3Coo(gapP.cc), fx), curP.cc)
    c = curP.c + gapP.c*fx
    return FullParameter(csr, p, f, ft, cc, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def copyParameter(P):
    return Parameter(P.csr, P.p.copy(), P.f, P.ft, P.cc, P.c)

@jitclass(parameterSpec)
class Parameter:
    def __init__(self, csr, p, f, ft, cc, c):
        self.csr = csr
        self.p = p
        self.f = f
        self.ft = ft
        self.cc = cc
        self.c = c

parameterType = Parameter.class_type.instance_type
