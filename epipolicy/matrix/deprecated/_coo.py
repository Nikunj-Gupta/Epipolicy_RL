from ..utility.singleton import *
from .sparse import SparseMatrix
from numba.types import DictType, UniTuple, ListType
from numba.typed import Dict, List
from numba.experimental import jitclass
from numba import jit
import numpy as np

innerDictType = DictType(nbInt, nbFloat)
cooMatrixSpec = [
    ('_from', DictType(nbInt, innerDictType)),
    ('shape', UniTuple(nbInt, 2))
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def copyCooMatrix(cooMatrix):
    cm = CooMatrix(cooMatrix.shape)
    for r in cooMatrix._from:
        cm._from[r] = Dict.empty(nbInt, nbFloat)
        for c, v in cooMatrix._from[r].items():
            cm._from[r][c] = v
    return cm

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getCsrMatrix(cooMatrix):
    sz = cooMatrix.getSize()
    nRows = cooMatrix.shape[0]
    data = np.zeros(sz, dtype=npFloat)
    indices = np.zeros(sz, dtype=npInt)
    indptr = np.zeros(nRows+1, dtype=npInt)
    for r in range(nRows):
        indptr[r+1] = indptr[r]
        if r in cooMatrix._from:
            indptr[r+1] += len(cooMatrix._from[r])
            i = indptr[r]
            for c in cooMatrix._from[r]:
                indices[i] = c
                i += 1
            indices[indptr[r]:indptr[r+1]].sort()
            for i in range(indptr[r], indptr[r+1]):
                data[i] = cooMatrix._from[r][indices[i]]
    return SparseMatrix(data, indices, indptr)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getCscMatrix(cooMatrix):
    sz = cooMatrix.getSize()
    nCols = cooMatrix.shape[1]
    data = np.zeros(sz, dtype=npFloat)
    indices = np.zeros(sz, dtype=npInt)
    indptr = np.zeros(nCols+1, dtype=npInt)
    _to = cooMatrix.getTo()
    for c in range(nCols):
        indptr[c+1] = indptr[c]
        if c in _to:
            indptr[c+1] += len(_to[c])
            i = indptr[c]
            for r in _to[c]:
                indices[i] = r
                i += 1
            indices[indptr[c]:indptr[c+1]].sort()
            for i in range(indptr[c], indptr[c+1]):
                data[i] = _to[c][indices[i]]
    return SparseMatrix(data, indices, indptr)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getNormalizedMatrix(origin, cooMatrix):
    normalized = CooMatrix(cooMatrix.shape)
    for r in origin._from:
        originSum = origin.getRowSum(r)
        rowSum = cooMatrix.getRowSum(r)
        if rowSum > EPSILON:
            for c in origin._from[r]:
                normalized.set(r, c, cooMatrix.get(r, c)/rowSum*originSum) # Preserving the origin sum of mobility
        else:
            # By default if every mobility is 0 then the within mobility is originSum
            for c in origin._from[r]:
                if c == r:
                    normalized.set(r, c, originSum)
                else:
                    normalized.set(r, c, 0)
    return normalized

@jitclass(cooMatrixSpec)
class CooMatrix:
    def __init__(self, shape):
        self._from = Dict.empty(nbInt, innerDictType)
        self.shape = shape
    def getRowSum(self, r):
        res = 0
        if r in self._from:
            for c, v in self._from[r].items():
                res += v
        return res
    def getTo(self):
        _to = Dict.empty(nbInt, innerDictType)
        for r in self._from:
            for c, v in self._from[r].items():
                if c not in _to:
                    _to[c] = Dict.empty(nbInt, nbFloat)
                _to[c][r] = v
        return _to
    def set(self, r, c, v):
        if r not in self._from:
            self._from[r] = Dict.empty(nbInt, nbFloat)
        self._from[r][c] = v
    def elementMultiply(self, r, c, v):
        if r not in self._from:
            self._from[r] = Dict.empty(nbInt, nbFloat)
        if c not in self._from[r]:
            self._from[r][c] = 1.0
        self._from[r][c] *= v
    def elementAdd(self, r, c, a):
        if r not in self._from:
            self._from[r] = Dict.empty(nbInt, nbFloat)
        if c not in self._from[r]:
            self._from[r][c] = 0
        self._from[r][c] += a
    def setCsrData(self, data, indices, indptr):
        for r in range(len(indptr)-1):
            for i in range(indptr[r], indptr[r+1]):
                c = indices[i]
                v = data[i]
                self.set(r, c, v)
    def setCooData(self, data, row, col):
        for i, val in enumerate(data):
            self.set(row[i], col[i], val)
        return self
    def getSum(self):
        res = 0
        for r in self._from:
            for val in self._from[r].values():
                res += val
        return res
    def has(self, r, c):
        if r not in self._from:
            return False
        return c in self._from[r]
    def get(self, r, c):
        return 0 if not self.has(r, c) else self._from[r][c]
    def getSize(self):
        sz = 0
        for r in self._from:
            sz += len(self._from[r])
        return sz
    def getCooData(self):
        sz = self.getSize()
        data = np.zeros(sz, dtype=npFloat)
        row = np.zeros(sz, dtype=npInt)
        col = np.zeros(sz, dtype=npInt)
        ind = 0
        for r in self._from:
            for c, v in self._from[r].items():
                data[ind] = v
                row[ind] = r
                col[ind] = c
                ind += 1
        return data, row, col

    def makeChange(self, dA, multiplier, froms, tos):
        for st in froms:
            if st in self._from:
                for fn in self._from[st]:
                    if fn in tos:
                        dA.elementMultiply(st, fn, multiplier)

    # def interChange(self, dA, multiplier, froms, tos):
    #     for st in froms:
    #         if st in self._from:
    #             for fn in self._from[st]:
    #                 if st != fn and fn in tos:
    #                     dA.elementMultiply(st, fn, multiplier)
    #
    # def intraChange(self, dA, multiplier, froms):
    #     for st in froms:
    #         if st in self._from and st in self._from[st]:
    #             dA.elementMultiply(st, st, multiplier)

    def setChange(self, curA, dA):
        for r in curA._from:
            for c, cur in curA._from[r].items():
                change = 0
                if dA.has(r, c):
                    change = self._from[r][c]*dA.get(r, c) - cur
                else:
                    change = self._from[r][c] - cur
                if abs(change) > EPSILON:
                    dA.set(r, c, change)

    def multiply(self, A):
        for r in A._from:
            for c, val in A._from[r].items():
                self.elementMultiply(r, c, val)
        return self

    def add(self, A):
        for r in A._from:
            for c, val in A._from[r].items():
                self.elementAdd(r, c, val)
        return self

    def scale(self, v):
        for r in self._from:
            for c in self._from[r]:
                self._from[r][c] *= v
        return self

    def addSignature(self, As):
        for A in As:
            for r in A._from:
                for c in A._from[r]:
                    if not self.has(r, c):
                        self.set(r, c, 0)
        return self

cooMatrixType = CooMatrix.class_type.instance_type
listCooType = ListType(cooMatrixType)
d1CooType = DictType(nbInt, cooMatrixType)
d2CooType = DictType(nbInt, d1CooType)
d3CooType = DictType(nbInt, d2CooType)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def copyD3Coo(d3):
    d3_ = Dict.empty(nbInt, d2CooType)
    for g in d3:
        d3_[g] = Dict.empty(nbInt, d1CooType)
        for c1 in d3[g]:
            d3_[g][c1] = Dict.empty(nbInt, cooMatrixType)
            for c2 in d3[g][c1]:
                d3_[g][c1][c2] = copyCooMatrix(d3[g][c1][c2])
    return d3_

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def scaleD3(d3, v):
    for g in d3:
        for c1 in d3[g]:
            for c2 in d3[g][c1]:
                d3[g][c1][c2].scale(v)
    return d3

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def addD3(d3A, d3B):
    for g in d3B:
        if g not in d3A:
            d3A[g] = d3B[g]
        else:
            for c1 in d3B[g]:
                if c1 not in d3A[g]:
                    d3A[g][c1] = d3B[g][c1]
                else:
                    for c2 in d3B[g][c1]:
                        if c2 not in d3A[g][c1]:
                            d3A[g][c1][c2] = d3B[g][c1][c2]
                        else:
                            d3A[g][c1][c2].add(d3B[g][c1][c2])
    return d3A

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeCooMatrix2DList(first, second, shape):
    coo = List.empty_list(listCooType)
    for i in range(first):
        coo.append(List.empty_list(cooMatrixType))
        for j in range(second):
            coo[i].append(CooMatrix(shape))
    return coo
