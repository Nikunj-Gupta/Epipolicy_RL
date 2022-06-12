from ..utility.singleton import nbInt, nbFloat
from numba.experimental import jitclass
from numba.types import ListType, string

np1iSpec = [
    ('data', nbInt[:])
]

@jitclass(np1iSpec)
class Np1i:
    def __init__(self, data):
        self.data = data

np1iType = Np1i.class_type.instance_type

np1fSpec = [
    ('data', nbFloat[:])
]

@jitclass(np1fSpec)
class Np1f:
    def __init__(self, data):
        self.data = data

np1fType = Np1f.class_type.instance_type


np2fSpec = [
    ('data', nbFloat[:,:])
]

@jitclass(np2fSpec)
class Np2f:
    def __init__(self, data):
        self.data = data

np2fType = Np2f.class_type.instance_type
listNp2fType = ListType(np2fType)

np3fSpec = [
    ('data', nbFloat[:,:,:])
]

@jitclass(np3fSpec)
class Np3f:
    def __init__(self, data):
        self.data = data

np3fType = Np3f.class_type.instance_type
