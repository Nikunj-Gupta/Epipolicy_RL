import numba as nb
import numpy as np
from numba.experimental import jitclass
from numba.types import UniTuple
from numba import njit

sparse_matrix_spec = [
    ("data", nb.float32[:]),
    ("indices", nb.int32[:]),
    ("indptr", nb.int32[:]),
    ("nnz", nb.int32),
    ("shape", UniTuple(nb.int32, 2))
]

@jitclass(sparse_matrix_spec)
class SparseMatrix:

    def __init__(self, data, indices, indptr):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        self.shape = (len(indptr)-1, len(indptr)-1)

SparseMatrixType = SparseMatrix.class_type.instance_type

@njit
def get_csr_matrix_from_coo(coo):
    size = coo.get_size()
    row_count = coo.shape[0]
    data = np.zeros(size, dtype=np.float32)
    indices = np.zeros(size, dtype=np.int32)
    indptr = np.zeros(row_count+1, dtype=np.int32)
    for row in range(row_count):
        indptr[row+1] = indptr[row]
        if row in coo.mat:
            indptr[row+1] += len(coo.mat[row])
            i = indptr[row]
            for col in coo.mat[row]:
                indices[i] = col
                i += 1
            indices[indptr[row]:indptr[row+1]].sort()
            for i in range(indptr[row], indptr[row+1]):
                data[i] = coo.mat[row][indices[i]]
    return SparseMatrix(data, indices, indptr)

@njit
def get_csc_matrix_from_coo(coo):
    size = coo.get_size()
    col_count = coo.shape[1]
    data = np.zeros(size, dtype=np.float32)
    indices = np.zeros(size, dtype=np.int32)
    indptr = np.zeros(col_count+1, dtype=np.int32)
    transpose_mat = coo.get_transpose()
    for col in range(col_count):
        indptr[col+1] = indptr[col]
        if col in transpose_mat:
            indptr[col+1] += len(transpose_mat[col])
            i = indptr[col]
            for row in transpose_mat[col]:
                indices[i] = row
                i += 1
            indices[indptr[col]:indptr[col+1]].sort()
            for i in range(indptr[col], indptr[col+1]):
                data[i] = transpose_mat[col][indices[i]]
    return SparseMatrix(data, indices, indptr)
