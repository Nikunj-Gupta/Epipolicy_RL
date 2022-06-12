from ..utility.singleton import *
from ..utility.utils import quadraticStepFunction
from numba.experimental import jitclass
from numba.types import ListType, DictType
from numba.typed import List, Dict
from ..sparse.coo import *
from .np_wrapper import Np2f, np2fType, Np3f, np3fType, listNp2fType
from numba import jit
import numpy as np

fullParameterSpec = [
    ('csr', listNp2fType),
    ('p', nbFloat[:,:,:,:]),
    ('f', nbFloat[:,:,:]),
    ('ft', nbFloat[:,:,:,:]),
    ('cc', d3CooType),
    ('c', nbFloat[:,:])
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def initializeFullParameter(nLocales, nParameters, nGroups, nFacilities, nInterventions):
    csrData = List.empty_list(np2fType)
    p = np.zeros((nParameters, nLocales, nFacilities, nGroups), dtype=npFloat)
    # If facility mobility is empty, then by default makeup value such that group shares equally time with each other
    f = np.ones((nLocales, nFacilities, nGroups), dtype=npFloat) / nFacilities
    ft = np.ones((nLocales, nFacilities, nGroups, nGroups), dtype=npFloat) / nGroups
    cc = Dict.empty(nbInt, d2CooType)
    c = np.zeros((nInterventions, nLocales), dtype=npFloat)
    return FullParameter(csrData, p, f, ft, cc, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeFullParameter(static, difParameter):
    csr = List.empty_list(np2fType)
    for i in range(static.nModes):
        csrData = np.zeros((static.nGroups, len(static.csr[i].data)), dtype=npFloat)
        for g in range(static.nGroups):
            coo = copyCooMatrix(static.coo[i][g]).multiply(difParameter.coo[i][g])
            csrData[g] = getCsrMatrix(getNormalizedMatrix(static.coo[i][g], coo)).data
        csr.append(Np2f(csrData))

    f = static.normalizeFacilityMatrix(static.P.f*difParameter.f)
    ft = static.normalizeContactMatrix(static.P.ft*difParameter.ft)
    return FullParameter(csr, static.P.p*difParameter.p, f, ft, difParameter.cc, difParameter.c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getGapFullParameter(preP, curP):
    csr = List.empty_list(np2fType)
    for i in range(len(curP.csr)):
        csr.append(Np2f(curP.csr[i].data-preP.csr[i].data))
    p = curP.p - preP.p
    f = curP.f - preP.f
    ft = curP.ft - preP.ft
    #cc = curP.cc - preP.cc
    cc = addD3(scaleD3(copyD3Coo(preP.cc), -1), curP.cc)
    c = curP.c - preP.c

    return FullParameter(csr, p, f, ft, cc, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def interpolateFullParameter(curP, gapP, ratio):
    fx = quadraticStepFunction(ratio)
    csr = List.empty_list(np2fType)
    for i in range(len(curP.csr)):
        csr.append(Np2f(curP.csr[i].data + gapP.csr[i].data*ratio))
    p = curP.p + gapP.p*ratio
    f = curP.f + gapP.f*ratio
    ft = curP.ft + gapP.ft*ratio
    cc = addD3(scaleD3(copyD3Coo(gapP.cc), fx), curP.cc)
    c = curP.c + gapP.c*fx
    return FullParameter(csr, p, f, ft, cc, c)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def copyFullParameter(P):
    return FullParameter(P.csr, P.p.copy(), P.f, P.ft, P.cc, P.c)

@jitclass(fullParameterSpec)
class FullParameter:
    def __init__(self, csr, p, f, ft, cc, c):
        self.csr = csr
        self.p = p
        self.f = f
        self.ft = ft
        self.cc = cc
        self.c = c

fullParameterType = FullParameter.class_type.instance_type
