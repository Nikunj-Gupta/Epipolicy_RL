from numba.types import DictType, UniTuple
from numba.typed import Dict
from numba.experimental import jitclass
from numba import jit

from ..utility.singleton import *
from .action import *
import numpy as np
import random, math, datetime

scheduleSpec = [
    ('tSpan', UniTuple(nbFloat, 2)),
    ('A0', actionsType),
    ('ticks', DictType(nbFloat, actionsType))
]

# @jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
# def generateRandomSchedule(T, static, interval=MCTS_INTERVAL):
#     A0 = Dict.empty(nbInt, actionType)
#     schedule = Schedule((0, T), A0)
#     for t in np.arange(0, T, interval, dtype=npFloat):
#         for itvId, itv in enumerate(static.interventions):
#             if itv.isCost or random.random() < 0.5:
#                 schedule.addAction(t, generateUniformRandomAction(static, itvId))
#     return schedule

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def getFinerSchedule(static, schedule, tSpan):
    if tSpan[0] == 0:
        A0 = schedule.A0
    else:
        if tSpan[0]-1 in schedule.ticks:
            A0 = schedule.ticks[tSpan[0]-1]
        else:
            A0 = static.generateEmptyActions()
    res = Schedule(tSpan, A0)
    for i in np.arange(tSpan[0], tSpan[1]):
        if i in schedule.ticks:
            for listAction in schedule.ticks[i]:
                for action in listAction:
                    res.addAction(static, i, action)
    return res

def printSchedule(static, schedule):
    res = ("({},{})\n".format(schedule.tSpan[0], schedule.tSpan[1]))
    res += printActions(static, schedule.A0)
    for tick, actions in schedule.ticks.items():
        res += "{}:{}".format(tick, printActions(static, actions))
    return res

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeUnitSchedule(static, A0, t, actions, interval=1):
    schedule = Schedule((t, t+interval), A0)
    for listAction in actions:
        for action in listAction:
            schedule.addAction(static, t, action)
    return schedule

@jitclass(scheduleSpec)
class Schedule:
    def __init__(self, tSpan, A0):
        self.tSpan = tSpan
        self.A0 = A0
        self.ticks = Dict.empty(nbFloat, actionsType)
    def addAction(self, static, tick, action):
        if self.tSpan[0] <= tick and tick < self.tSpan[1]:
            if tick not in self.ticks:
                self.ticks[tick] = List.empty_list(listActionType)
                for i in range(static.nInterventions):
                    self.ticks[tick].append(List.empty_list(actionType))
            self.ticks[tick][action.id].append(action)

scheduleType = Schedule.class_type.instance_type

# def getPyActions(A):
#     pyActions = {}
#     for actId, action in A.items():
#         pyActions[actId] = makePyAction(action)
#     return pyActions
#
# def getActions(static, pyActions):
#     actions = Dict.empty(nbInt, actionType)
#     for actId, pyAction in pyActions.items():
#         actions[nbInt(actId)] = pyAction.getAction(static)
#     return actions
#
# def dumpActions(A):
#     res = {}
#     for actId, act in A.items():
#         res[actId] = act.dumps()
#     return res
#
# def loadPyActions(dump):
#     res = {}
#     for actId, actDump in dump.items():
#         res[actId] = loadPyAction(actDump)
#     return res
#
# def loadPySchedule(dump):
#     pySchedule = PySchedule()
#     pySchedule.tSpan = (float(dump["tSpan"][0]), float(dump["tSpan"][1]))
#     pySchedule.A0 = loadPyActions(dump["A0"])
#     pySchedule.ticks = {}
#     for tick, dumpActs in dump["ticks"].items():
#         pySchedule.ticks[float(tick)] = loadPyActions(dumpActs)
#     return pySchedule
#
# def makePySchedule(schedule):
#     pySchedule = PySchedule()
#     pySchedule.tSpan = (schedule.tSpan[0], schedule.tSpan[1])
#     pySchedule.A0 = getPyActions(schedule.A0)
#     pySchedule.ticks = {}
#     for tick, A in schedule.ticks.items():
#         pySchedule.ticks[tick] = getPyActions(A)
#     return pySchedule
#
# class PySchedule:
#     def __init__(self):
#         pass
#
#     def getSchedule(self, static):
#         schedule = Schedule(self.tSpan, getActions(static, self.A0))
#         for tick, pyActions in self.ticks.items():
#             for pyAction in pyActions.values():
#                 schedule.addAction(tick, pyAction.getAction(static))
#         return schedule
#     def dumps(self):
#         ticks = {tick:dumpActions(acts) for tick, acts in self.ticks.items()}
#         res = {
#             "tSpan": list(self.tSpan),
#             "A0": dumpActions(self.A0),
#             "ticks": ticks
#         }
#         return res
