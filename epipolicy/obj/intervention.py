import numba as nb
from numba.experimental import jitclass
from numba.types import string, ListType
from numba.typed import List

from .control_parameter import ControlParameterType

intervention_spec = [
    ("name", string),
    ("cp_list", ListType(ControlParameterType)),
    ("is_cost", nb.boolean)
]

@jitclass(intervention_spec)
class Intervention:

    def __init__(self, name, is_cost):
        self.name = name
        self.cp_list = List.empty_list(ControlParameterType)
        self.is_cost = is_cost

InterventionType = Intervention.class_type.instance_type
