from numba.experimental import jitclass
from numba.types import UniTuple
from ..utility.singleton import nbInt, nbFloat

sparseMatrixSpec = [
    ('data', nbFloat[:]),
    ('indices', nbInt[:]),
    ('indptr', nbInt[:]),
    ('nnz', nbInt),
    ('shape', UniTuple(nbInt, 2))
]

@jitclass(sparseMatrixSpec)
class SparseMatrix:
    def __init__(self, data, indices, indptr):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        self.shape = (len(self.indptr)-1, len(self.indptr)-1)

sparseMatrixType = SparseMatrix.class_type.instance_type
