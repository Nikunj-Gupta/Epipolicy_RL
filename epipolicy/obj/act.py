import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba.types import string, ListType

act_spec = [
    ("index", nb.int32),
    ("locale_regex", string),
    ("cpv_list", nb.float32[:])
]

def construct_act(index, cpv_list, locale_regex="*"):
    np_cpv_list = np.array(cpv_list, dtype=np.float32)
    return Act(index, locale_regex, np_cpv_list)

@jitclass(act_spec)
class Act:

    def __init__(self, index, locale_regex, cpv_list):
        self.index = index
        self.locale_regex = locale_regex
        self.cpv_list = cpv_list

ActType = Act.class_type.instance_type
ActListType = ListType(ActType)

def get_act_repr(static, act):
    args = ", ".join([str(round(v,2)) for v in act.cpv_list])
    return "{}({}, {})".format(static.interventions[act.index].name, act.locale_regex, args)

def get_action_repr(static, action):
    repr = ""
    for act in action:
        repr += get_act_repr(static, act) + '\n'
    return repr

@njit
def get_default_act(static, act_index):
    itv = static.interventions[act_index]
    cpv_list = np.zeros(len(itv.cp_list), np.float32)
    for i, cp in enumerate(itv.cp_list):
        cpv_list[i] = cp.default_value
    return Act(act_index, "*", cpv_list)

def get_normalized_action(static, action):
    cost_exists = {}
    for itv_id, itv in enumerate(static.interventions):
        if itv.is_cost:
            cost_exists[itv_id] = False
    for act in action:
        if act.index in cost_exists:
            cost_exists[act.index] = True
    for cost_id in cost_exists:
        if not cost_exists[cost_id]:
            action.append(get_default_act(static, cost_id))
    return action
