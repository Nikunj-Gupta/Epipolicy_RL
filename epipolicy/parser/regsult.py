import numba as nb
from numba.experimental import jitclass
from numba.types import DictType, string, Array
from numba.typed import Dict

from ..utility.numba_type import Int1DArrayType, SetType

regsult_spec = [
    ("locale_result", DictType(string, SetType)),
    ("index_result", DictType(string, Int1DArrayType)),
    ("string_result", DictType(string, string))
]

@jitclass(regsult_spec)
class Regsult:

    def __init__(self):
        self.locale_result = Dict.empty(string, SetType)
        self.index_result = Dict.empty(string, Int1DArrayType)
        self.string_result = Dict.empty(string, string)

RegsultType = Regsult.class_type.instance_type
