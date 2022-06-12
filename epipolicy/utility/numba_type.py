# Ref: https://github.com/numba/numba/issues/6191
import numba as nb
from numba.types import ListType, DictType, string, Array

SetType = DictType(nb.int32, nb.boolean)
StringIntDictType = DictType(string, nb.int32)
StringListType = ListType(string)
IntFloatDictType = DictType(nb.int32, nb.float32)
Float2DArrayType = Array(nb.float32, 2, 'C')
Int1DArrayType = Array(nb.int32, 1, 'C')
