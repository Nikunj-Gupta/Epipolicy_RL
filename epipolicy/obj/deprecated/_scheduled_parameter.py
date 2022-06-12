from numba.types import ListType, UniTuple
from numba.typed import List
from numba.experimental import jitclass
from numba import jit

from ..utility.singleton import *
from .parameter import Parameter, parameterType, makeParameter, interpolateParameter, getGapParameter

import numpy as np

scheduledParameterSpec = [
    ('tSpan', UniTuple(nbFloat, 2)),
    ('P0', parameterType),
    ('transPeriod', nbFloat),
    ('ticks', nbFloat[:]),
    ('parameters', ListType(parameterType)),
    ('gapParameters', ListType(parameterType))
]

@jit(nopython=NO_PYTHON, nogil=NO_GIL, fastmath=FAST_MATH)
def makeScheduledParameter(epi, schedule, transPeriod=1.0):
    P0 = makeParameter(epi.static, epi.executeActions(schedule.A0))
    ticks = np.zeros(len(schedule.ticks), dtype=npFloat)
    i = 0
    for tick in schedule.ticks:
        ticks[i] = tick
        i += 1
    ticks.sort()
    parameters = List.empty_list(parameterType)
    for tick in ticks:
        actions = schedule.ticks[tick]
        parameters.append(makeParameter(epi.static, epi.executeActions(actions)))
    return ScheduledParameter(schedule.tSpan, P0, transPeriod, ticks, parameters)

@jitclass(scheduledParameterSpec)
class ScheduledParameter:
    def __init__(self, tSpan, P0, transPeriod, ticks, parameters):
        self.tSpan = tSpan
        self.P0 = P0
        self.transPeriod = transPeriod
        self.ticks = ticks
        self.parameters = parameters
        self.gapParameters = List.empty_list(parameterType)
        preP = P0
        for i, p in enumerate(self.parameters):
            self.gapParameters.append(getGapParameter(preP, p))
            preP = p

    def measure(self, tick):
        if len(self.ticks) == 0:
            return self.P0
        left = 0
        right = len(self.ticks)-1
        while left < right:
            mid = (left+right+1)//2
            if tick < self.ticks[mid]:
                right = mid - 1
            else:
                left = mid

        curP = self.parameters[left]
        preP = None
        ind = None
        if left == 0:
            if tick < self.ticks[0]:
                return self.P0
            preP = self.P0
            ind = -2
        else:
            preP = self.parameters[left-1]
            ind = left-1

        curTick = self.ticks[left]
        distance = tick - curTick
        gap = 0
        if left == len(self.ticks)-1:
            gap = self.tSpan[1] - curTick
        else:
            gap = self.ticks[left+1] - curTick
        interpolateGap = min(gap, self.transPeriod)
        if distance >= interpolateGap:
            return curP
        # distance < interpolateGap
        return interpolateParameter(preP, self.gapParameters[left], distance/interpolateGap)

scheduledParameterType = ScheduledParameter.class_type.instance_type
