import numpy as np
import numba as nb
from numba.types import ListType
from numba.typed import List
from numba.experimental import jitclass
from numba import njit

from .state import StateType
from .act import ActType, ActListType

schedule_spec = [
    ("horizon", nb.int32),
    ("initial_state", StateType),
    ("action_list", ListType(ActListType)),
]

@jitclass(schedule_spec)
class Schedule:
    def __init__(self, horizon, initial_state):
        self.horizon = horizon
        self.initial_state = initial_state
        self.action_list = List.empty_list(ActListType)
        for i in range(horizon):
            self.action_list.append(List.empty_list(ActType))
    def add_act(self, time_step, act):
        if 0 <= time_step and time_step < self.horizon:
            self.action_list[time_step].append(act)

ScheduleType = Schedule.class_type.instance_type
