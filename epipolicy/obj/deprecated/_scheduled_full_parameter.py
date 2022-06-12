from numba.types import ListType, UniTuple
from numba.typed import List
from numba.experimental import jitclass
from numba import jit

from ..utility.singleton import *
from .full_parameter import FullParameter, fullParameterType, makeFullParameter, interpolateFullParameter, getGapFullParameter

import numpy as np

scheduledFullParameterSpec = [
    ('tSpan', UniTuple(nbFloat, 2)),
    ('P0', fullParameterType),
    ('transPeriod', nbFloat),
    ('ticks', nbFloat[:]),
    ('parameters', ListType(fullParameterType)),
    ('gapParameters', ListType(fullParameterType))
]

def makeScheduledFullParameter(epi, schedule, transPeriod=1.0):
    D0 = epi.executeActions(schedule.A0)
    P0 = makeFullParameter(epi.static, D0)
    #print("HERE", schedule.tSpan, np.sum(P0.c[0]))
    ticks = np.zeros(len(schedule.ticks), dtype=npFloat)
    i = 0
    for tick in schedule.ticks:
        ticks[i] = tick
        i += 1
    ticks.sort()
    parameters = List.empty_list(fullParameterType)
    D1 = None
    P1 = None
    for tick in ticks:
        actions = schedule.ticks[tick]
        D1 = epi.executeActions(actions)
        P1 = makeFullParameter(epi.static, D1)
        parameters.append(P1)
    if D1 is None:
        D1 = D0
        P1 = P0
    return ScheduledFullParameter(schedule.tSpan, P0, transPeriod, ticks, parameters), D1, P1

@jitclass(scheduledFullParameterSpec)
class ScheduledFullParameter:
    def __init__(self, tSpan, P0, transPeriod, ticks, parameters):
        self.tSpan = tSpan
        self.P0 = P0
        self.transPeriod = transPeriod
        self.ticks = ticks
        self.parameters = parameters
        self.gapParameters = List.empty_list(fullParameterType)
        preP = P0
        for i, p in enumerate(self.parameters):
            self.gapParameters.append(getGapFullParameter(preP, p))
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
        return interpolateFullParameter(preP, self.gapParameters[left], distance/interpolateGap)

scheduledFullParameterType = ScheduledFullParameter.class_type.instance_type
