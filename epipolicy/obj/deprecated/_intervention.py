from ..utility.singleton import *
from numba.experimental import jitclass
from numba.types import string, ListType
from numba.typed import List
from .control_parameter import ControlParameter, listControlParameterType, controlParameterType

interventionSpec = [
    ('id', nbInt),
    ('name', string),
    ('cps', listControlParameterType),
    # ('hashLimit', nbHashedInt),
    ('isCost', nbBool)
]

@jitclass(interventionSpec)
class Intervention:
    def __init__(self, id, name, isCost):
        self.id = id
        self.name = name
        self.cps = List.empty_list(controlParameterType)
        self.isCost = isCost

    # def setHashLimit(self):
    #     self.hashLimit = 1
    #     for cp in self.cps:
    #         self.hashLimit *= cp.nBuckets

interventionType = Intervention.class_type.instance_type
listInterventionType = ListType(interventionType)
