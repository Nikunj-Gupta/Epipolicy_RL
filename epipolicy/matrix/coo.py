import numpy as np
import numba as nb
from numba.types import DictType, UniTuple, ListType
from numba.typed import Dict, List
from numba.experimental import jitclass
from numba import njit

from .sparse import SparseMatrix
from ..utility.singleton import FLOAT_EPSILON
from ..utility.numba_type import IntFloatDictType

coo_matrix_spec = [
    ("mat", DictType(nb.int32, IntFloatDictType)),
    ("shape", UniTuple(nb.int32, 2))
]

@jitclass(coo_matrix_spec)
class CooMatrix:

    def __init__(self, shape):
        self.mat = Dict.empty(nb.int32, IntFloatDictType)
        self.shape = shape

    def get_row_sum(self, row):
        row_sum = 0
        if row in self.mat:
            for val in self.mat[row].values():
                row_sum += val
        return row_sum

    def get_transpose(self):
        transpose_mat = Dict.empty(nb.int32, IntFloatDictType)
        for row in self.mat:
            for col, val in self.mat[row].items():
                if col not in transpose_mat:
                    transpose_mat[col] = Dict.empty(nb.int32, nb.float32)
                transpose_mat[col][row] = val
        return transpose_mat

    def set(self, row, col, val):
        if row not in self.mat:
            self.mat[row] = Dict.empty(nb.int32, nb.float32)
        self.mat[row][col] = val

    def element_multiply(self, row, col, val):
        if row not in self.mat:
            self.mat[row] = Dict.empty(nb.int32, nb.float32)
        if col not in self.mat[row]:
            self.mat[row][col] = 1.0
        self.mat[row][col] *= val

    def element_add(self, row, col, val):
        if row not in self.mat:
            self.mat[row] = Dict.empty(nb.int32, nb.float32)
        if col not in self.mat[row]:
            self.mat[row][col] = 0.0
        self.mat[row][col] += val

    def set_csr_matrix(self, data, indices, indptr):
        for row in range(len(indptr)-1):
            for i in range(indptr[row], indptr[row+1]):
                col = indices[i]
                val = data[i]
                self.set(row, col, val)
        return self

    def set_coo_matrix(self, data, rows, cols):
        for i, val in enumerate(data):
            self.set(rows[i], cols[i], val)
        return self

    def get_sum(self):
        mat_sum = 0
        for row in self.mat:
            for val in self.mat[row].values():
                mat_sum += val
        return mat_sum

    def has(self, row, col):
        if row not in self.mat:
            return False
        return col in self.mat[row]

    def get(self, row, col):
        return 0 if not self.has(row, col) else self.mat[row][col]

    def get_size(self):
        size = 0
        for row in self.mat:
            size += len(self.mat[row])
        return size

    def get_coo_matrix(self):
        size = self.get_size()
        data = np.zeros(size, dtype=np.float32)
        rows = np.zeros(size, dtype=np.int32)
        cols = np.zeros(size, dtype=np.int32)
        index = 0
        for row in self.mat:
            for col, val in self.mat[row].items():
                data[index] = val
                rows[index] = row
                cols[index] = col
                index += 1
        return data, rows, cols

    def pairwise_multiply(self, base_coo, multiplier, from_list, to_list):
        for row in from_list:
            if row in base_coo.mat:
                for col in base_coo.mat[row]:
                    if col in to_list:
                        self.element_multiply(row, col, multiplier)

    # Deprecated because interpolation no longer goes through coo but reduced_csr
    # def set_delta(self, base_coo, current_coo):
    #     for row in current_coo.mat:
    #         for col, current_val in current_coo.mat[row].items():
    #             delta = 0
    #             if self.has(row, col):
    #                 delta = base_coo.mat[row][col]*self.get(row, col) - current_val
    #             else:
    #                 delta = base_coo.mat[row][col] - current_val
    #             if abs(delta) > FLOAT_EPSILON:
    #                 self.set(row, col, delta)

    def multiply(self, coo):
        for row in coo.mat:
            for col, val in coo.mat[row].items():
                self.element_multiply(row, col, val)
        return self

    def add(self, coo):
        for row in coo.mat:
            for col, val in coo.mat[row].items():
                self.element_add(row, col, val)
        return self

    def scale(self, val):
        for row in self.mat:
            for col in self.mat[row]:
                self.mat[row][col] *= val
        return self

    def add_signature(self, coo_list):
        for coo in coo_list:
            for row in coo.mat:
                for col in coo.mat[row]:
                    if not self.has(row, col):
                        self.set(row, col, 0)
        return self

CooMatrixType = CooMatrix.class_type.instance_type
CooMatrixListType = ListType(CooMatrixType)

@njit
def get_coo_matrix_llist(first, second, shape):
    coo_llist = List.empty_list(CooMatrixListType)
    for i in range(first):
        coo_llist.append(List.empty_list(CooMatrixType))
        for j in range(second):
            coo_llist[i].append(CooMatrix(shape))
    return coo_llist

@njit
def copy_coo_matrix(coo):
    copied_coo = CooMatrix(coo.shape)
    for row in coo.mat:
        copied_coo.mat[row] = Dict.empty(nb.int32, nb.float32)
        for col, val in coo.mat[row].items():
            copied_coo.mat[row][col] = val
    return copied_coo

@njit
def get_normalized_matrix(origin_coo, coo):
    normalized_coo = CooMatrix(coo.shape)
    for row in origin_coo.mat:
        origin_row_sum = origin_coo.get_row_sum(row)
        row_sum = coo.get_row_sum(row)
        if row_sum > FLOAT_EPSILON:
            for col in origin_coo.mat[row]:
                normalized_coo.set(row, col, coo.get(row, col)/row_sum*origin_row_sum) # Preserving the origin sum of mobility
        else:
            # By default if every mobility is 0 then the within mobility is originSum
            normalized_coo.set(row, row, origin_row_sum)
            for col in origin_coo.mat[row]:
                if col != row:
                    normalized_coo.set(row, col, 0)
    return normalized_coo

D1CooType = DictType(nb.int32, CooMatrixType)
D2CooType = DictType(nb.int32, D1CooType)

@njit
def copy_d3_coo(d3):
    copied_d3 = Dict.empty(nb.int32, D2CooType)
    for g in d3:
        copied_d3[g] = Dict.empty(nb.int32, D1CooType)
        for c1 in d3[g]:
            copied_d3[g][c1] = Dict.empty(nb.int32, CooMatrixType)
            for c2 in d3[g][c1]:
                copied_d3[g][c1][c2] = copy_coo_matrix(d3[g][c1][c2])
    return copied_d3

@njit
def scale_d3(d3, val):
    for g in d3:
        for c1 in d3[g]:
            for c2 in d3[g][c1]:
                d3[g][c1][c2].scale(val)
    return d3

@njit
def add_d3(d3A, d3B):
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
