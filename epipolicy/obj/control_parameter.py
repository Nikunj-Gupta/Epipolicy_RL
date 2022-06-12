import numba as nb
from numba.experimental import jitclass
from numba.types import string

control_parameter_spec = [
    ("name", string),
    ("default_value", nb.float32),
    ("min_value", nb.float32),
    ("max_value", nb.float32)
]

@jitclass(control_parameter_spec)
class ControlParameter:

    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value

    def set_interval(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

ControlParameterType = ControlParameter.class_type.instance_type
