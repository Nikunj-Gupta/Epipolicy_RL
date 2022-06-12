from numba.experimental import jitclass
from numba import njit
import numba as nb

from .observable import ObservableType, copy_observable, get_initial_observable
from .parameter import ParameterType, copy_parameter, get_initial_parameter

state_spec = [
    ("t", nb.int32),
    ("obs", ObservableType),
    ("unobs", ParameterType)
]

@jitclass(state_spec)
class State:

    def __init__(self, t, obs, unobs):
        self.t = t
        self.obs = obs
        self.unobs = unobs

StateType = State.class_type.instance_type

@njit
def copy_state(state):
    return State(state.t, copy_observable(state.obs), copy_parameter(state.unobs))

@njit
def get_initial_state(static):
    return State(0, get_initial_observable(static), get_initial_parameter(static))
