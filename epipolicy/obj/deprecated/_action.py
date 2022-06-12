from ..utility.singleton import *
from ..utility.utils import getChoice
from .np_wrapper import Np1f, np1fType
from numba.experimental import jitclass
from numba.types import DictType, string, ListType
from numba.typed import List
from numba import jit
import numpy as np

actionSpec = [
    ('id', nbInt),
    # ('cpi', nbInt[:]),
    ('cpv', nbFloat[:]),
    ('locale', string)
    # ('hash', nbHashedInt)
]

def printAction(static, action):
    args = ', '.join([str(v) for v in action.cpv])
    return "{}({}, {})".format(static.interventions[action.id].name, args, action.locale)

def printActions(static, preState, curState, actions):
    res = "Actions:\n"
    for itvId in range(static.nInterventions):
        for action in actions[itvId]:
            totalCost = np.sum(curState.c[itvId])
            incurredCost = totalCost if preState is None else totalCost - np.sum(preState.c[itvId])
            res += "\t{}, ic:{} tc:{}\n".format(printAction(static, action), incurredCost, totalCost)
    return res

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def generateUniformRandomAction(static, itvId):
#     weights = List.empty_list(np1fType)
#     itv = static.interventions[itvId]
#     for cp in itv.cps:
#         weights.append(Np1f(np.ones(cp.nBuckets, dtype=npFloat)))
#     return generateRandomAction(static, itvId, weights)

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def generateZeroAction(static, itvId):
    itv = static.interventions[itvId]
    #print(len(itv.cps))
    cpv = np.zeros(len(itv.cps), npFloat)
    return Action(itvId, cpv, "*")

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def generateDefaultAction(static, itvId):
    itv = static.interventions[itvId]
    #print(len(itv.cps))
    cpv = np.zeros(len(itv.cps), npFloat)
    for i, cp in enumerate(itv.cps):
        cpv[i] = cp.defaultValue
    return Action(itvId, cpv, "*")

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def generateRandomAction(static, itvId, weights):
#     itv = static.interventions[itvId]
#     cpi = np.zeros(len(itv.cps), dtype=npInt)
#     for i, cp in enumerate(itv.cps):
#         cpi[i] = getChoice(weights[i].data)
#     return Action(static, itvId, cpi)

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def makeActionFromValues(static, id, cpv):
#     cps = static.interventions[id].cps
#     cpi = np.zeros(len(cps), dtype=npInt)
#     for cpId, cp in enumerate(cps):
#         if cp.nBuckets == 1:
#             index = 0
#         else:
#             index = round((cpv[cpId] - cp.low) / (cp.high - cp.low) * (cp.nBuckets - 1))
#         cpi[cpId] = npInt(min(cp.nBuckets-1, max(0, index)))
#     return Action(static, id, cpi)

@jitclass(actionSpec)
class Action:
    def __init__(self, id, cpv, locale):
        self.id = id
        self.cpv = cpv
        self.locale = locale
        # self.cpi = cpi
        # self.hash = 0
        # itv = static.interventions[id]
        # coef = 1.0
        # self.cpv = np.zeros(len(self.cpi), dtype=npFloat)
        # for i, cp in enumerate(itv.cps):
        #     # Compute hash
        #     self.hash += self.cpi[i]*coef
        #     coef *= cp.nBuckets
        #     # Compute cpv
        #     if cp.nBuckets == 1:
        #         self.cpv[i] = cp.low
        #     else:
        #         self.cpv[i] = cp.low + (cp.high-cp.low)/(cp.nBuckets-1)*self.cpi[i]

actionType = Action.class_type.instance_type
listActionType = ListType(actionType)
actionsType = ListType(ListType(actionType))

# def makePyAction(action):
#     pyAction = PyAction()
#     pyAction.id = action.id
#     pyAction.cpi = action.cpi.tolist()
#     return pyAction
#
# def loadPyAction(dump):
#     pyAction = PyAction()
#     pyAction.id = nbInt(dump['id'])
#     pyAction.cpi = list(dump['cpi'])
#     return pyAction
#
# class PyAction:
#     def __init__(self):
#         pass
#     def getAction(self, static):
#         return Action(static, self.id, np.array(self.cpi, dtype=npInt))
#     def dumps(self):
#         return {
#             "id": self.id,
#             "cpi": self.cpi
#         }
