from ..sparse.coo import CooMatrix, cooMatrixType, listCooType, d1CooType, d2CooType, d3CooType, makeCooMatrix2DList
from ..utility.singleton import *
from .np_wrapper import Np3f, np3fType
from numba.types import DictType, ListType
from numba.typed import Dict, List
import numpy as np

from numba.experimental import jitclass
from numba import jit

difParameterSpec = [
    ('coo', ListType(listCooType)),
    ('p', nbFloat[:,:,:,:]), # pxlxfxg
    ('f', nbFloat[:,:,:]),
    ('ft', nbFloat[:,:,:,:]),
    ('cc', d3CooType), # gxcxcxlxl
    ('c', nbFloat[:,:]), #ixl
    # ('fc', nbFloat[:,:,:,:]),
    ('inf', nbFloat[:,:,:]) # ixlxg
    # ('finf', nbFloat[:,:,:,:])
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def initializeDifParameter(static):
    coo = makeCooMatrix2DList(static.nModes, static.nGroups, static.shape)
    p = np.ones((static.nParameters, static.nLocales, static.nFacilities, static.nGroups), dtype=npFloat)
    f = np.ones((static.nLocales, static.nFacilities, static.nGroups), dtype=npFloat)
    ft = np.ones((static.nLocales, static.nFacilities, static.nGroups, static.nGroups), dtype=npFloat)
    #cc = np.zeros((static.nCompartments, static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    cc = Dict.empty(nbInt, d2CooType)
    c = np.zeros((static.nInterventions, static.nLocales), dtype=npFloat)
    # fc = np.zeros((static.nInterventions, static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    inf = np.zeros((static.nInterventions, static.nLocales, static.nGroups), dtype=npFloat)
    # finf = np.zeros((static.nInterventions, static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    return DifParameter(coo, p, f, ft, cc, c, inf)

@jitclass(difParameterSpec)
class DifParameter:
    def __init__(self, coo, p, f, ft, cc, c, inf):
        self.coo = coo
        self.p = p
        self.f = f
        self.ft = ft
        self.cc = cc
        self.c = c
        self.inf = inf

    def addMove(self, static, inf, g, c1, c2, l1, l2, amount):
        if g not in self.cc:
            self.cc[g] = Dict.empty(nbInt, d1CooType)
        if c1 not in self.cc[g]:
            self.cc[g][c1] = Dict.empty(nbInt, cooMatrixType)
        if c2 not in self.cc[g][c1]:
            self.cc[g][c1][c2] = CooMatrix(static.shape)
        #print(g, c1, c2, l1, l2, amount)
        self.cc[g][c1][c2].elementAdd(l1, l2, amount)
        inf[l1, g] += amount/static.sN[l1, g]

    def combine(self, other):
        nModes = len(self.coo)
        nGroups = len(self.coo[0])
        nCompartments = len(self.cc[0])
        for i in range(nModes):
            for j in range(nGroups):
                self.coo[i][j].multiply(other.coo[i][j])

        self.p *= other.p
        self.f *= other.f
        self.ft *= other.ft
        for g in other.cc:
            if g not in self.cc:
                self.cc[g] = other.cc[g]
            else:
                for c1 in other.cc[g]:
                    if c1 not in self.cc[g]:
                        self.cc[g][c1] = other.cc[g][c1]
                    else:
                        for c2 in other.cc[g][c1]:
                            if c2 not in self.cc[g][c1]:
                                self.cc[g][c1][c2] = other.cc[g][c1][c2]
                            else:
                                self.cc[g][c1][c2].add(other.cc[g][c1][c2])
        # for i in range(nGroups):
        #     for j in range(nCompartments):
        #         for k in range(nCompartments):
        #             self.cc[i][j][k].multiply(other.cc[i][j][k])
        self.c += other.c
        self.inf = 1-(1-self.inf)*(1-other.inf)

difParameterType = DifParameter.class_type.instance_type
