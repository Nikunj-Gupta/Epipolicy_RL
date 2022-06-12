from ..utility.singleton import nbInt, nbFloat
from numba.experimental import jitclass
from numba.types import ListType
from numba.types import string

controlParameterSpec = [
    ('id', nbInt),
    ('name', string),
    ('defaultValue', nbFloat)
    # ('nBuckets', nbInt),
    # ('low', nbFloat),
    # ('high', nbFloat)
]

@jitclass(controlParameterSpec)
class ControlParameter:
    def __init__(self, id, name, defaultValue):
        # if high <= low:
        #     high = low
        #     nBuckets = 1
        self.id = id
        self.name = name
        self.defaultValue = defaultValue
        # self.low = low
        # self.high = high
        # self.nBuckets = nBuckets

controlParameterType = ControlParameter.class_type.instance_type
listControlParameterType = ListType(controlParameterType)
