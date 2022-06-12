import numpy as np
import time
from numba import jit
from numba.types import ListType, DictType
from numba.typed import List, Dict
from ..utility.singleton import *
from ..obj.state import State, stateType, makeState
from ..obj.full_state import FullState, fullStateType, makeFullState
from ..obj.parameter import copyParameter
from ..obj.full_parameter import copyFullParameter

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def computeSummant(static, s, P, N, summant):
    res = np.ones((static.nLocales, static.nGroups), dtype=npFloat)
    for var in summant.muls:
        if static.hasPropName("parameter", var):
            #print(var, P.p[static.getPropIndex("parameter", var), 0, 0])
            res *= P.p[static.getPropIndex("parameter", var), :, 0]
        elif var == ALIVE_COMPARTMENT:
            res *= N
        elif static.hasPropName("compartment", var):
            res *= s[static.getPropIndex("compartment", var)]
    return res

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def computeFraction(static, s, P, N, frac):
    numer, denom = np.zeros((2, static.nLocales, static.nGroups), dtype=npFloat)
    numer += computeSummant(static, s, P, N, frac.numer)
    for summant in frac.denom:
        denom += summant.coef * computeSummant(static, s, P, N, summant)
    for i in range(static.nLocales):
        for j in range(static.nGroups):
            if denom[i,j] == 0:
                denom[i,j] = 1
    return numer / denom

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def computeDS(static, s, P):
    N = getSumNotTagMatrix(static, s, DIE_TAG)
    dS = np.zeros((static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    csrC, cscC = getCombinedMobility(static, P.csr)

    for edge in static.edges.values():
        dEdge = computeFraction(static, s, P, N, edge.frac)
        for comId, coef in edge.coms.items():
            dS[comId] += dEdge*coef

    for infEdge in static.infEdges.values():
        D = getFoiMatrix(static, s, P, N, infEdge.infId, cscC)
        dEdge = computeInfectionEdge(static, s, P, D, infEdge.susId, infEdge.paramId, csrC, infEdge.numerParamIds, infEdge.denomParamIds)
        for comId, coef in infEdge.coms.items():
            dS[comId] += dEdge*coef

    newS = s + dS
    for g in P.cc:
        for c1 in P.cc[g]:
            for c2 in P.cc[g][c1]:
                for coo in P.cc[g][c1][c2]:
                    for l1 in coo._from:
                        for l2, v in coo._from[l1].items():
                            newV = min(v, newS[c1, l1, g])
                            newS[c1, l1, g] -= newV
                            newS[c2, l2, g] += newV
    # for c1 in range(static.nCompartments):
    #     for c2 in range(c1+1, static.nCompartments):
    #         dC12 = P.cc[c1,c2] - P.cc[c2,c1]
    #         if np.sum(np.abs(dC12)) < EPSILON:
    #             continue
    #
    #         for l in range(static.nLocales):
    #             for g in range(static.nGroups):
    #                 if abs(dC12[l,g]) > EPSILON:
    #                     coef = 1.0
    #                     if dC12[l,g] > 0:
    #                         coef = min(coef, newS[c1,l,g] / dC12[l,g])
    #                     else:
    #                         coef = min(coef, - newS[c2,l,g] / dC12[l,g])
    #                     newS[c1,l,g] -= dC12[l,g]*coef
    #                     newS[c2,l,g] += dC12[l,g]*coef

    return newS - s

intFloatDictType = DictType(nbInt, nbFloat)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def setTrans(trans, c1, c2, v):
    if c1 not in trans:
        trans[c1] = Dict.empty(nbInt, nbFloat)
    trans[c1][c2] = v

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def computeTransition(coms):
    gain = Dict.empty(nbInt, nbFloat)
    lose = Dict.empty(nbInt, nbFloat)
    trans = Dict.empty(nbInt, intFloatDictType)
    pos = Dict.empty(nbInt, nbFloat)
    neg = Dict.empty(nbInt, nbFloat)
    sumPos = 0
    sumNeg = 0
    for comId, coef in coms.items():
        if coef > 0:
            pos[comId] = coef
            sumPos += coef
        else:
            neg[comId] = -coef
            sumNeg -= coef
    if abs(sumPos-sumNeg) < EPSILON:
        for negId, negCoef in neg.items():
            for posId, posCoef in pos.items():
                setTrans(trans, negId, posId, negCoef*posCoef/sumPos)
    elif sumPos > sumNeg:
        dif = sumPos - sumNeg
        for posId, posCoef in pos.items():
            v = dif*posCoef/sumPos
            gain[posId] = v
            pos[posId] -= v
        if sumNeg > 0:
            for negId, negCoef in neg.items():
                for posId, posCoef in pos.items():
                    setTrans(trans, negId, posId, negCoef*posCoef/sumNeg)
    else:
        dif = sumNeg - sumPos
        for negId, negCoef in neg.items():
            v = dif*negCoef/sumNeg
            lose[negId] = v
            neg[negId] -= v
        if sumPos > 0:
            for negId, negCoef in neg.items():
                for posId, posCoef in pos.items():
                    setTrans(trans, negId, posId, negCoef*posCoef/sumPos)
    return gain, lose, trans

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def computeFullDS(static, s, P):
    #print(np.sum(s[2]))
    N = getSumNotTagMatrix(static, s, DIE_TAG)
    dS, dGain, dLose = np.zeros((3, static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    dTrans = np.zeros((static.nCompartments, static.nCompartments, static.nLocales, static.nGroups), dtype=npFloat)
    csrC, cscC = getCombinedMobility(static, P.csr)

    for edge in static.edges.values():
        dEdge = computeFraction(static, s, P, N, edge.frac)
        for comId, coef in edge.coms.items():
            dS[comId] += dEdge*coef
        gain, lose, trans = computeTransition(edge.coms)
        for comId, coef in gain.items():
            dGain[comId] += dEdge*coef
        for comId, coef in lose.items():
            dLose[comId] += dEdge*coef
        for c1 in trans:
            for c2, coef in trans[c1].items():
                #print("edge", edge.frac.signature, c1, c2, coef, np.sum(dEdge))
                dTrans[c1, c2] += dEdge*coef

    for infEdge in static.infEdges.values():
        D = getFoiMatrix(static, s, P, N, infEdge.infId, cscC)
        #print(np.sum(D))
        dEdge = computeInfectionEdge(static, s, P, D, infEdge.susId, infEdge.paramId, csrC, infEdge.numerParamIds, infEdge.denomParamIds)
        #print(np.sum(dEdge))
        # for numerParamId in infEdge.numerParamIds:
        #     dEdge *= P.p[numerParamId,:,0,:]
        for comId, coef in infEdge.coms.items():
            #print(coef, np.sum(dEdge))
            dS[comId] += dEdge*coef
        gain, lose, trans = computeTransition(infEdge.coms)
        for comId, coef in gain.items():
            dGain[comId] += dEdge*coef
        for comId, coef in lose.items():
            dLose[comId] += dEdge*coef
        for c1 in trans:
            for c2, coef in trans[c1].items():
                #print("infEdge", infEdge.frac.signature, c1, c2, coef, np.sum(dEdge))
                dTrans[c1, c2] += dEdge*coef

    newS = s + dS
    #print("HERE", np.sum(s[1]), np.sum(dS[1]), np.sum(newS[1]))
    for g in P.cc:
        for c1 in P.cc[g]:
            for c2 in P.cc[g][c1]:
                coo = P.cc[g][c1][c2]
                for l1 in coo._from:
                    for l2, v in coo._from[l1].items():
                        newV = min(v, newS[c1, l1, g])
                        #print("PRE", newV, newS[c1, l1, g], newS[c2, l2, g])
                        newS[c1, l1, g] -= newV
                        newS[c2, l2, g] += newV
                        #print("AFT", newV, newS[c1, l1, g], newS[c2, l2, g])
                        if l1 == l2:
                            dTrans[c1, c2, l1, g] += newV
    # for c1 in range(static.nCompartments):
    #     for c2 in range(c1+1, static.nCompartments):
    #         dC12 = P.cc[c1,c2] - P.cc[c2,c1]
    #         if np.sum(np.abs(dC12)) < EPSILON:
    #             continue
    #
    #         for l in range(static.nLocales):
    #             for g in range(static.nGroups):
    #                 if abs(dC12[l,g]) > EPSILON:
    #                     coef = 1.0
    #                     if dC12[l,g] > 0:
    #                         coef = min(coef, newS[c1,l,g] / dC12[l,g])
    #                     else:
    #                         coef = min(coef, - newS[c2,l,g] / dC12[l,g])
    #                     newS[c1,l,g] -= dC12[l,g]*coef
    #                     newS[c2,l,g] += dC12[l,g]*coef
    #
    #                     if dC12[l,g] > 0:
    #                         dNew[c2,l,g] += dC12[l,g] * coef
    #                     else:
    #                         dNew[c1,l,g] -= dC12[l,g] * coef
    return newS - s, dGain, dLose, dTrans

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def computeDC(static, scale, s, P):
#     return np.sum(P.c*s + P.fc) * scale
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def computeFullDC(static, scale, s, P):
#     dC = np.zeros((static.nInterventions, static.nLocales), dtype=npFloat)
#     for i in range(static.nInterventions):
#         dC[i] += P.c[i]*s
#     dC += P.fc
#     return dC

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def forward(static, scale, S, P):
    state = makeState(static, S)
    dS = computeDS(static, state.s, P)
    # dC = computeDC(static, scale, state.s, P)
    return State(dS, P.c).flatten()

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def fullForward(t, static, scale, S, P):
    fullState = makeFullState(static, S)
    #print("TIME", t, np.sum(fullState.s[1]))
    #print(np.sum(fullState.s[2]))
    dS, dGain, dLose, dTrans = computeFullDS(static, fullState.s, P)
    # dC = computeFullDC(static, scale, fullState.s, P)
    return FullState(dS, dGain, dLose, dTrans, P.c).flatten()

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def forward_b(dSc, static, scale, S, P):
#     state = makeState(static, S)
#     dS = computeDS(static, state.s, P)
#     dC = computeDC(static, scale, state.s, P)
#     flat = State(dS, dC).flatten()
#     for i in range(len(flat)):
#         dSc[i] = flat[i]
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def fullForward_b(dSc, static, scale, S, P):
#     fullState = makeFullState(static, S)
#     dS, dNew = computeFullDS(static, fullState.s, P)
#     dC = computeFullDC(static, scale, fullState.s, P)
#     flat = FullState(dS, dNew, dC).flatten()
#     for i in range(len(flat)):
#         dSc[i] = flat[i]

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def f_b(dSc, S, p, t):
#     static, scheduledParameter = p
#     _scale = scheduledParameter.tSpan[1] - scheduledParameter.tSpan[0]
#     unscaledT = t * _scale
#     P = copyParameter(scheduledParameter.measure(unscaledT))
#     P.p *= _scale
#     forward_b(dSc, static, _scale, S, P)
#
# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def fullF_b(dSc, S, p, t):
#     static, scheduledParameter = p
#     _scale = scheduledParameter.tSpan[1] - scheduledParameter.tSpan[0]
#     unscaledT = t * _scale
#     P = copyFullParameter(scheduledParameter.measure(unscaledT))
#     P.p *= _scale
#     fullForward_b(dSc, static, _scale, S, P)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getFoiEntry(static, s, P, N, c2, u0, j, v, data):
    numer = 0
    denom = 0
    indices = static.sumCsc.indices
    indptr = static.sumCsc.indptr
    for u in range(static.nGroups):
        numerSummant = 0
        denomSummant = 0
        for rowIndex in range(indptr[j], indptr[j+1]):
            i = indices[rowIndex]
            val = data[u][rowIndex]
            #print(s[c2,i,u], c2, i, u, np.sum(s[c2]))
            numerSummant += s[c2,i,u]*val
            denomSummant += N[i,u]*val
            #print("FACTOR", P.f[j,v,u], P.ft[j,v,u0,u], P.ft[j,v,u,u0], val)
        factor = P.f[j,v,u]*P.ft[j,v,u0,u]*P.ft[j,v,u,u0]
        #print(v, np.sum(P.ft[j,v,u0]))
        numer += numerSummant*factor
        denom += denomSummant*factor
    if denom <= EPSILON:
        denom = 1.0
    #print(numer, denom)
    return numer / denom

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getFoiMatrix(static, s, P, N, c2, data):
    D = np.zeros((static.nGroups, static.nLocales, static.nFacilities), dtype=npFloat)
    for u0 in range(static.nGroups):
        for j in range(static.nLocales):
            for v in range(static.nFacilities):
                D[u0,j,v] = getFoiEntry(static, s, P, N, c2, u0, j, v, data)
    return D

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def computeInfectionEdge(static, s, P, D, c0, p0, data, numerParams, denomParams):
    res = np.zeros((static.nLocales, static.nGroups), dtype=npFloat)
    indices = static.sumCsr.indices
    indptr = static.sumCsr.indptr
    for i0 in range(static.nLocales):
        for u0 in range(static.nGroups):
            entry = 0
            for colIndex in range(indptr[i0], indptr[i0+1]):
                j = indices[colIndex]
                val = data[u0][colIndex]
                tmp = 0

                # for v in range(static.nFacilities):
                #     if v == 0 and P.p[p0,j,u0] > 0:
                #         tmp += 0.28*P.f[j,v,u0]*D[u0,j,v]/10
                #     else:
                #         tmp += P.p[p0,j,u0]*P.f[j,v,u0]*D[u0,j,v]
                # tmp *= val

                for v in range(static.nFacilities):
                    #print("OMGHERE", P.f[j,v,u0], D[u0,j,v], P.p[p0,j,v,u0])
                    coef = 1.0
                    for p1 in numerParams:
                        coef *= P.p[p1,j,v,u0]
                    for p1 in denomParams:
                        if abs(P.p[p1,j,v,u0]) > 0:
                            coef /= P.p[p1,j,v,u0]
                    tmp += P.f[j,v,u0]*D[u0,j,v]*P.p[p0,j,v,u0]*coef
                #print("VAL", val)
                tmp *= val

                entry += tmp
            #print("HERE", res[i0,u0], entry, s[c0,i0,u0])
            res[i0,u0] = entry*s[c0,i0,u0]
    return res

# Ref: https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
# csr to csc
@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getCombinedMobility(static, csr):
    nnz = static.sumCsr.nnz
    csrC, cscC = np.zeros((2, static.nGroups, nnz), dtype=npFloat)
    indptrC = static.sumCsc.indptr.copy()
    for r in range(static.nLocales):
        movingIndex = np.zeros(static.nModes, dtype=npInt)
        for i in range(static.nModes):
            movingIndex[i] = static.csr[i].indptr[r]
        for colIndex in range(static.sumCsr.indptr[r], static.sumCsr.indptr[r+1]):
            c = static.sumCsr.indices[colIndex]
            v = np.zeros(static.nGroups, dtype=npFloat)
            for i in range(static.nModes):
                if movingIndex[i] < static.csr[i].indptr[r+1]:
                    movingC = static.csr[i].indices[movingIndex[i]]
                    if c == movingC:
                        for g in range(static.nGroups):
                            v[g] += static.bias[i]*csr[i].data[g, movingIndex[i]]
                        movingIndex[i] += 1

            dest = indptrC[c]
            for g in range(static.nGroups):
                csrC[g, colIndex] = v[g]
                cscC[g, dest] = v[g]
            indptrC[c] += 1
    return csrC, cscC
