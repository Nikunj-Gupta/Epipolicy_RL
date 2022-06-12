import json, os, time, re, types
# from diffeqpy import de
# from julia import Main
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
import random, dill, math
from numba.typed import Dict, List
from numba import njit, jit, types, typed
from ..obj.static import Static
from ..obj.dif_parameter import DifParameter, difParameterType, initializeDifParameter
from ..obj.parameter import copyParameter, makeParameter, Parameter, parameterType
from ..obj.full_parameter import copyFullParameter, makeFullParameter, FullParameter, fullParameterType
from ..obj.action import Action, actionType
from ..obj.schedule import *
from ..obj.scheduled_parameter import makeScheduledParameter
from ..obj.scheduled_full_parameter import makeScheduledFullParameter
from ..obj.np_wrapper import Np3f, np3fType, Np1i
from ..obj.state import makeState, makeInitialState, makeStateFrom
from ..obj.full_state import makeFullState, makeInitialFullState
from ..obj.action import printActions
from ..obj.history import History
from ..optimizer.mcts import MCTS
from ..parser.regex_parser import RegexParser
from ..parser.regex_builder import Expr
from .core_optimize import *
from .core_utils import makeStatic, getNLocales, getNTags
from ..utility.singleton import *
from ..utility.utils import inject, parseDate, normalize, getInfectivity, getInfluence
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from datetime import timedelta

class Epidemic:

    def __init__(self, conn, session, runMode=RUN_NORMAL):
        self.conn = conn
        self.session = session["session"]
        self.runMode = runMode
        self.simulationId = session["id"]

        # # Temporary ignore group "other"
        # if len(self.session['groups']) > 3:
        #     self.session['groups'].pop()

        self.model = self.session["model"]
        self.locales = self.session["locales"]
        self.compartments = self.model["compartments"]
        self.parameters = self.model["parameters"]
        self.leaf_locales = getNLocales(self.locales)

        self.static = Static(
            nLocales=len(self.truncatedLocales),
            nCompartments=len(self.compartments),
            nParameters=len(self.parameters),
            nGroups=max(len(self.session["groups"]), 1),
            nFacilities=max(len(self.session["facilities"]), 1),
            nInterventions=len(self.session["interventions"])+len(self.session["costs"]),
            nModes=len(MODES),
            nCompartmentTags=getNTags(self.compartments),
            nParameterTags=getNTags(self.parameters)
        )
        self.parser = RegexParser(self.static)
        makeStatic(self.session, self.static, self.parser)
        # For debug
        # for sig, edge in self.static.infEdges.items():
        #     print(sig)
        #     for comId, coef in edge.coms.items():
        #         print("\t", self.compartments[comId]["name"], coef)
        # print("SEPARATE")
        # for sig, edge in self.static.edges.items():
        #     print(sig)
        #     for comId, coef in edge.coms.items():
        #         print("\t", self.compartments[comId]["name"], coef)
        # print("VEC")
        # print([(self.compartments[i]["name"], x) for i, x in enumerate(self.static.ivec)])
        # print([(self.compartments[i]["name"], x) for i, x in enumerate(self.static.avec)])
        # print([(self.compartments[i]["name"], x) for i, x in enumerate(self.static.svec)])
        # print(self.static.initId)
        self.setInterventionFunctions()
        #self.precomputeInterventions()

        if self.runMode == RUN_MCTS:
            self.mcts = MCTS(self)

    def registerFunction(self, index, func, fn):
        # print("ORIGINAL")
        # print(func)
        func = inject(index, func, fn)
        # print("INJECTED")
        # print(func)
        exec(func)
        exec("self.{}{} = {}{}.__get__(self)".format(fn, index, fn, index))

    def setInterventionFunctions(self):
        interventions = self.session["interventions"]
        costs = self.session["costs"]

        for i, intervention in enumerate(interventions):
            self.registerFunction(i, intervention["effect"], "effect")
            self.registerFunction(i, intervention["cost"], "cost")

        for i, cost in enumerate(costs):
            self.registerFunction(len(interventions) + i, cost["func"], "cost")

    def executeActions(self, actions):
        difParameter = initializeDifParameter(self.static)
        for listAction in actions:
            for action in listAction:
                actDict = {}
                #print(action.id, action.cpv)
                for cpId, cpv in enumerate(action.cpv):
                    actDict[self.static.interventions[action.id].cps[cpId].name] = cpv
                #print("HERE",actDict)
                if not self.static.interventions[action.id].isCost:
                    #print(action.id)
                    exec("self.effect{}(actDict, action.locale, difParameter, action.id)".format(action.id))
                    exec("self.cost{}(actDict, action.locale, difParameter, action.id)".format(action.id))
                else:
                    exec("self.cost{}(actDict, difParameter, action.id)".format(action.id))
        return difParameter

    # def recurseChoice(self, itvId, index, choices):
    #     itv = self.static.interventions[itvId]
    #     if index >= len(itv.cps):
    #         actions = Dict.empty(nbInt, actionType)
    #         actions[itvId] = Action(self.static, itvId, choices)
    #         self.static.precomputedInterventions[itvId][actions[itvId].hash] = self.executeActions(actions)
    #     else:
    #         cp = itv.cps[index]
    #         for i in range(cp.nBuckets):
    #             choices[index] = i
    #             self.recurseChoice(itvId, index+1, choices)
    #
    # def precomputeInterventions(self):
    #     for itvId, itv in enumerate(self.static.interventions):
    #         self.static.precomputedInterventions[itvId] = Dict.empty(nbHashedInt, difParameterType)
    #         if len(itv.cps) == 0:
    #             actions = Dict.empty(nbInt, actionType)
    #             actions[itvId] = Action(self.static, itvId, np.zeros(0, dtype=npInt))
    #             self.static.precomputedInterventions[itvId][actions[itvId].hash] = self.executeActions(actions)
    #         else:
    #             choices = np.zeros(len(itv.cps), dtype=npInt)
    #             self.recurseChoice(itvId, 0, choices)

    def strictenLocale(self, func, res):
        if "locale-from" not in res.lResult:
            localeFrom = Dict.empty(nbInt, nbBool)
            for l in range(self.static.nLocales):
                localeFrom[l] = True
            res.lResult["locale-from"] = localeFrom

    def strictenGroup(self, func, res):
        if "group-from" not in res.iResult:
            res.iResult["group-from"] = Np1i(np.arange(self.static.nGroups, dtype=npInt))

    def strictenFacility(self, func, res):
        if "facility" not in res.iResult:
            res.iResult["facility"] = Np1i(np.arange(self.static.nFacilities, dtype=npInt))

    def strictenCompartment(self, func, res):
        if "compartment-from" not in res.iResult:
            if func == "move":
                aliveCompartments = []
                if self.hasPropName("compartmentTag", DIE_TAG):
                    for c in range(self.static.nCompartments):
                        if self.static.compartmentTags[c, DIE_TAG] == 0:
                            aliveCompartments.append(c)
                else:
                    for c in range(self.static.nCompartments):
                        aliveCompartments.append(c)
                res.iResult["compartment-from"] = Np1i(np.array(aliveCompartments, dtype=npInt))

    def strictenResult(self, func, res):
        if func == "select" or func == "apply":
            if "parameter" in res.iResult:
                self.strictenLocale(func, res)
                self.strictenFacility(func, res)
                self.strictenGroup(func, res)
            elif "locale-from" in res.lResult and "locale-to" in res.lResult:
                self.strictenGroup(func, res)
            elif "group-from" in res.iResult and "group-to" in res.iResult:
                self.strictenLocale(func, res)
                self.strictenFacility(func, res)
            elif "compartment-from" in res.iResult:
                self.strictenLocale(func, res)
                self.strictenGroup(func, res)
            elif "facility" in res.iResult:
                self.strictenLocale(func, res)
                self.strictenGroup(func, res)
        elif func == "add":
            if "intervention" in res.iResult:
                self.strictenLocale(func, res)
        elif func == "move":
                self.strictenCompartment(func, res)
                self.strictenLocale(func, res)
                self.strictenGroup(func, res)
        return res

    def getRegsult(self, func, regex):
        # if isinstance(regex, str):
        #     res = self.parser.parseRegex(regex)
        # elif isinstance(regex, dict):
        nDictRegex = Dict.empty(types.string, types.string)
        for k, v in regex.items():
            nDictRegex[str(k)] = str(v)
        res = self.parser.parseDictRegex(nDictRegex)
        # else:
        #     res = self.parser.parseRegex(str(regex))
        return self.strictenResult(func, res)

    def getLocaleName(self, index):
        return self.truncatedLocales[index]["name"]

    def select(self, regex, difParameter, itvId):
        res = self.getRegsult("select", regex)
        rows = []
        valueMode = "current"
        if "value-mode" in res.sResult:
            valueMode = res.sResult["value-mode"]
        if "locale-from" in res.lResult and "locale-to" in res.lResult:
            modeIndex = self.static.getPropIndex("mode", "border")
            if "mode" in res.sResult:
                modeIndex = self.static.getPropIndex("mode", res.sResult["mode"])
            coo = self.static.coo[modeIndex]
            if valueMode == "current":
                indices = self.static.csr[modeIndex].indices
                indptr = self.static.csr[modeIndex].indptr
                data = self.P.csr[modeIndex].data
                for g in res.iResult["group-from"].data:
                    for l1 in res.lResult["locale-from"]:
                        if l1 in coo[g]._from:
                            for colIndex in range(indptr[l1], indptr[l1+1]):
                                l2 = indices[colIndex]
                                if l2 in res.lResult["locale-to"]:
                                    val = data[g][colIndex]
                                    rows.append([self.static.getPropName("group", g), self.getLocaleName(l1), self.getLocaleName(l2), val])
            elif valueMode == "default":
                for g in res.iResult["group-from"].data:
                    for l1 in res.lResult["locale-from"]:
                        if l1 in coo[g]._from:
                            for l2 in coo[g]._from[l1]:
                                if l2 in res.lResult["locale-to"]:
                                    rows.append([self.static.getPropName("group", g), self.getLocaleName(l1), self.getLocaleName(l2), coo[g]._from[l1][l2]])
            ret = pd.DataFrame(rows, columns=['Group', 'Locale From', 'Locale To', 'Value'])
        elif "parameter" in res.iResult:
            if valueMode == "current":
                for p in res.iResult["parameter"].data:
                    for l in res.lResult["locale-from"]:
                        for f in res.iResult["facility"].data:
                            for g in res.iResult["group-from"].data:
                                rows.append([self.static.getPropName("parameter", p), self.getLocaleName(l), self.static.getPropName("facility", f), self.static.getPropName("group", g), self.P.p[p, l, f, g]])
            elif valueMode == "default":
                for p in res.iResult["parameter"].data:
                    for l in res.lResult["locale-from"]:
                        for f in res.iResult["facility"].data:
                            for g in res.iResult["group-from"].data:
                                rows.append([self.static.getPropName("parameter", p), self.getLocaleName(l), self.static.getPropName("facility", f), self.static.getPropName("group", g), self.static.P.p[p, l, f, g]])
            ret = pd.DataFrame(rows, columns=['Parameter', 'Locale', 'Facility', 'Group', 'Value'])
        elif "group-from" in res.iResult and "group-to" in res.iResult:
            if valueMode == "current":
                for l in res.lResult["locale-from"]:
                    for f in res.iResult["facility"].data:
                        for g1 in res.iResult["group-from"].data:
                            for g2 in res.iResult["group-to"].data:
                                rows.append([self.getLocaleName(l), self.static.getPropName("facility", f), self.static.getPropName("group", g1), self.static.getPropName("group", g2), self.P.ft[l, f, g1, g2]])
            elif valueMode == "default":
                for l in res.lResult["locale-from"]:
                    for f in res.iResult["facility"].data:
                        for g1 in res.iResult["group-from"].data:
                            for g2 in res.iResult["group-to"].data:
                                rows.append([self.getLocaleName(l), self.static.getPropName("facility", f), self.static.getPropName("group", g1), self.static.getPropName("group", g2), self.static.P.ft[l, f, g1, g2]])
            ret = pd.DataFrame(rows, columns=['Locale', 'Facility', 'Group From', 'Group To', 'Value'])
        elif "compartment-from" in res.iResult:
            if valueMode == "current":
                for c in res.iResult["compartment-from"].data:
                    for l in res.lResult["locale-from"]:
                        for g in res.iResult["group-from"].data:
                            rows.append([self.static.getPropName("compartment", c), self.getLocaleName(l), self.static.getPropName("group", g), self.curState.s[c, l, g]])
            elif valueMode == "default":
                for c in res.iResult["compartment-from"].data:
                    for l in res.lResult["locale-from"]:
                        for g in res.iResult["group-from"].data:
                            rows.append([self.static.getPropName("compartment", c), self.getLocaleName(l), self.static.getPropName("group", g), self.static.s[c, l, g]])
            elif valueMode == "change":
                for c in res.iResult["compartment-from"].data:
                    for l in res.lResult["locale-from"]:
                        for g in res.iResult["group-from"].data:
                            change = self.curState.s[c, l, g]
                            if len(self.history.states) > 1:
                                change -= self.history.states[-2].s[c, l, g]
                            rows.append([self.static.getPropName("compartment", c), self.getLocaleName(l), self.static.getPropName("group", g), change])
            ret = pd.DataFrame(rows, columns=['Compartment', 'Locale', 'Group', 'Value'])
            #print(ret)
        elif "facility" in res.iResult:
            if valueMode == "current":
                for l in res.lResult["locale-from"]:
                    for f in res.iResult["facility"].data:
                        for g in res.iResult["group-from"].data:
                            rows.append([self.getLocaleName(l), self.static.getPropName("facility", f), self.static.getPropName("group", g), self.P.f[l, f, g]])
            elif valueMode == "default":
                for l in res.lResult["locale-from"]:
                    for f in res.iResult["facility"].data:
                        for g in res.iResult["group-from"].data:
                            rows.append([self.getLocaleName(l), self.static.getPropName("facility", f), self.static.getPropName("group", g), self.static.P.f[l, f, g]])
            ret = pd.DataFrame(rows, columns=['Locale', 'Facility', 'Group', 'Value'])
        #print(ret)
        return ret

    def apply(self, regex, multiplier, difParameter, itvId):
        #print(regex)
        res = self.getRegsult("apply", regex)
        infScore = getInfluence(multiplier)
        inf = np.zeros((self.static.nLocales, self.static.nGroups), dtype=npFloat)
        if "locale-from" in res.lResult and "locale-to" in res.lResult:
            modeIndex = self.static.getPropIndex("mode", "border")
            if "mode" in res.sResult:
                modeIndex = self.static.getPropIndex("mode", res.sResult["mode"])
            for g in res.iResult["group-from"].data:
                self.static.coo[modeIndex][g].makeChange(difParameter.coo[modeIndex][g], multiplier, res.lResult["locale-from"], res.lResult["locale-to"])

            coo = self.static.coo[modeIndex]
            for g in res.iResult["group-from"].data:
                for l1 in res.lResult["locale-from"]:
                    if l1 in coo[g]._from:
                        for l2, val in coo[g]._from[l1].items():
                            if l2 in res.lResult["locale-to"]:
                                inf[l1, g] += self.static.pN[l1, g]*val*infScore

        elif "parameter" in res.iResult:
            for p in res.iResult["parameter"].data:
                for l in res.lResult["locale-from"]:
                    for f in res.iResult["facility"].data:
                        for g in res.iResult["group-from"].data:
                            difParameter.p[p, l, f, g] *= multiplier
            for l in res.lResult["locale-from"]:
                for f in res.iResult["facility"].data:
                    for g in res.iResult["group-from"].data:
                        inf[l, g] += self.static.P.f[l, f, g]*infScore
        elif "group-from" in res.iResult and "group-to" in res.iResult:
            for l in res.lResult["locale-from"]:
                for f in res.iResult["facility"].data:
                    for g1 in res.iResult["group-from"].data:
                        for g2 in res.iResult["group-to"].data:
                            difParameter.ft[l, f, g1, g2] *= multiplier
                            inf[l, g1] += self.static.P.f[l, f, g1]*self.static.P.ft[l, f, g1, g2]*infScore
                            inf[l, g2] += self.static.P.f[l, f, g2]*self.static.P.ft[l, f, g2, g1]*infScore

        elif "facility" in res.iResult:
            for l in res.lResult["locale-from"]:
                for f in res.iResult["facility"].data:
                    for g in res.iResult["group-from"].data:
                        difParameter.f[l, f, g] *= multiplier
                        inf[l, g] += self.static.P.f[l, f, g]*infScore
                        #difParameter.inf[itvId, l, g] = 1-(1-difParameter.inf[itvId, l, g])*(1-self.P.f[l, f, g]*inf)
        #print("HERE", np.sum(inf))
        difParameter.inf[itvId] = 1-(1-difParameter.inf[itvId])*(1-inf)

    def move(self, regex1, regex2, value, difParameter, itvId):
        res1 = self.getRegsult("move", regex1)
        res2 = self.getRegsult("move", regex2)
        inf = np.zeros((self.static.nLocales, self.static.nGroups), dtype=npFloat)
        departSum = 0
        for g in range(self.static.nGroups):
            if g in res1.iResult["group-from"].data and g in res2.iResult["group-from"].data:
                for c1 in res1.iResult["compartment-from"].data:
                    for l1 in res1.lResult["locale-from"]:
                        departSum += self.curState.s[c1, l1, g]
        if departSum > 0:
            c2Set = set()
            for c2 in res2.iResult["compartment-from"].data:
                c2Set.add(c2)
            for g in range(self.static.nGroups):
                if g in res1.iResult["group-from"].data and g in res2.iResult["group-from"].data:
                    for c1 in res1.iResult["compartment-from"].data:
                        for l1 in res1.lResult["locale-from"]:
                            departAmount = self.curState.s[c1, l1, g]/departSum*value
                            if c1 in c2Set:
                                if l1 not in res2.lResult["locale-from"]:
                                    arriveSum = 0
                                    for l2 in res2.lResult["locale-from"]:
                                        arriveSum += self.curState.s[c1, l2, g]
                                    if arriveSum > 0:
                                        for l2 in res2.lResult["locale-from"]:
                                            moveAmount = self.curState.s[c1, l2, g]/arriveSum*departAmount
                                            difParameter.addMove(self.static, inf, g, c1, c1, l1, l2, moveAmount)
                                    else:
                                        for l2 in res2.lResult["locale-from"]:
                                            moveAmount = 1/len(res2.lResult["locale-from"])*departAmount
                                            difParameter.addMove(self.static, inf, g, c1, c1, l1, l2, moveAmount)
                            else:
                                if l1 in res2.lResult["locale-from"]:
                                    arriveSum = 0
                                    for c2 in res2.iResult["compartment-from"].data:
                                        arriveSum += self.curState.s[c2, l1, g]
                                    if arriveSum > 0:
                                        for c2 in res2.iResult["compartment-from"].data:
                                            moveAmount = self.curState.s[c2, l1, g]/arriveSum*departAmount
                                            difParameter.addMove(self.static, inf, g, c1, c2, l1, l1, moveAmount)
                                    else:
                                        for c2 in res2.iResult["compartment-from"].data:
                                            moveAmount = 1/len(res2.iResult["compartment-from"].data)*departAmount
                                            difParameter.addMove(self.static, inf, g, c1, c2, l1, l1, moveAmount)
                                else:
                                    arriveSum = 0
                                    for c2 in res2.iResult["compartment-from"].data:
                                        for l2 in res2.lResult["locale-from"]:
                                            arriveSum += self.curState.s[c2, l2, g]
                                    if arriveSum > 0:
                                        for c2 in res2.iResult["compartment-from"].data:
                                            for l2 in res2.lResult["locale-from"]:
                                                moveAmount = self.curState.s[c2, l2, g]/arriveSum*departAmount
                                                difParameter.addMove(self.static, inf, g, c1, c2, l1, l2, moveAmount)
                                    else:
                                        nPartitions = len(res2.iResult["compartment-from"].data)*len(res2.lResult["locale-from"])
                                        for c2 in res2.iResult["compartment-from"].data:
                                            for l2 in res2.lResult["locale-from"]:
                                                moveAmount = 1/nPartitions*departAmount
                                                difParameter.addMove(self.static, inf, g, c1, c2, l1, l2, moveAmount)
        difParameter.inf[itvId] = 1-(1-difParameter.inf[itvId])*(1-np.minimum(inf, 1))

    def add(self, regex, value, difParameter, itvId):
        res = self.getRegsult("add", regex)
        if "intervention" in res.iResult:
            nPartitions = len(res.iResult["intervention"].data)*len(res.lResult["locale-from"])
            if nPartitions > 0:
                for i in res.iResult["intervention"].data:
                    for l in res.lResult["locale-from"]:
                        difParameter.c[i, l] += (value/nPartitions)

    def getResponses(self):
        T = len(self.history.states)
        totalInitInf = getSumTagMatrix(self.static, self.static.s, INF_TAG)
        totalIncidences = [self.static.getTotalIncidenceMatrix(state)+totalInitInf for state in self.history.states]
        totalNews = [self.static.getTotalNewMatrix(state) for state in self.history.states]

        incidences = [totalIncidences[0]]
        news = [totalNews[0]]
        for t in range(1, T):
            incidences.append(totalIncidences[t] - totalIncidences[t-1])
            #print("HERE", t, np.sum(incidences[-1]))
            news.append(totalNews[t] - totalNews[t-1])

        # New way
        # maxBeta = 0
        # if self.static.hasPropName("parameterTag", TRANS_TAG):
        #     for p0 in range(self.static.nParameters):
        #         #print(self.static.parameterTags)
        #         if self.static.parameterTags[p0, self.static.getPropIndex("parameterTag", TRANS_TAG)] > 0:
        #             maxBeta = max(maxBeta, np.amax(self.static.P.p[p0]))
        # betaWindow = int(math.ceil(1.0/maxBeta)) if maxBeta > 0 else T

        totalInfTime = 0
        totalIncidence = 0
        averageInfDuration = np.zeros(T, dtype=npFloat)
        for t in range(T):
            # Old way
            totalInfTime += np.sum(getSumTagMatrix(self.static, self.history.states[t].s, INF_TAG))
            totalIncidence += np.sum(incidences[t])
            if totalIncidence > 0:
                averageInfDuration[t] = totalInfTime / totalIncidence
            # New way
            # if t >= betaWindow:
            #     totalInfTime += np.sum(getSumTagMatrix(self.static, self.history.states[t].s-self.history.states[t-betaWindow].s, INF_TAG))
            #     totalIncidence += np.sum(incidences[t] - incidences[t-betaWindow])
            # else:
            #     totalInfTime += np.sum(getSumTagMatrix(self.static, self.history.states[t].s, INF_TAG))
            #     totalIncidence += np.sum(incidences[t])
            # if totalIncidence > 0:
            #     averageInfDuration[t] = totalInfTime / totalIncidence
            #print("HERE2", t, totalInfTime,'-', np.sum(getSumTagMatrix(self.static, self.history.states[t].s, INF_TAG)), '-', totalIncidence, np.sum(incidences[t]), '-', averageInfDuration[t])

        #print([round(np.sum(x),1) for x in totalIncidences])
        #print([round(np.sum(x),1) for x in incidences])

        lIncidences = []
        lNews = []
        lCounts = []
        lItvCosts = []
        lSus = []
        lAlives = []
        lInfs = []
        lParameters = []
        lAveragePs = []
        lTrans = []

        for t in range(T):
            if self.history.schedules[t] is None:
                actions = self.static.generateEmptyActions()
            else:
                actions = self.getActions(self.history.schedules[t])

            lIncidence = np.zeros((len(self.locales), self.static.nGroups), dtype=npFloat)
            lNew = np.zeros((self.static.nCompartments, len(self.locales), self.static.nGroups), dtype=npFloat)
            lCount = np.zeros((self.static.nCompartments, len(self.locales), self.static.nGroups), dtype=npFloat)
            lItvCost = np.zeros((self.static.nInterventions, len(self.locales)), dtype=npFloat)
            lSu = np.zeros((len(self.locales), self.static.nGroups), dtype=npFloat)
            lAlive = np.zeros((len(self.locales), self.static.nGroups), dtype=npFloat)
            lInf = np.zeros((self.static.nInterventions, len(self.locales), self.static.nGroups), dtype=npFloat)
            lParameter = [[[0 for k in range(len(self.static.interventions[i].cps))] for i in range(self.static.nInterventions)] for j in range(len(self.locales))]
            lParamCount = [[[0 for k in range(len(self.static.interventions[i].cps))] for i in range(self.static.nInterventions)] for j in range(len(self.locales))]
            lAverageP = np.zeros((self.static.nParameters, len(self.locales)), dtype=npFloat)
            lSumP = np.zeros((self.static.nParameters, len(self.locales)), dtype=npFloat)
            lTran = np.zeros((self.static.nCompartments, self.static.nCompartments, len(self.locales)), dtype=npFloat)

            sus = getSumTagMatrix(self.static, self.history.states[t].s, SUS_TAG)
            alive = getSumNotTagMatrix(self.static, self.history.states[t].s, DIE_TAG)

            for locId, locale in enumerate(self.locales):
                contains = self.static.localeHierarchy[normalize(locale["name"])]
                for l in contains:
                    lIncidence[locId] += incidences[t][l]
                    lNew[:,locId] += news[t][:,l]
                    lCount[:,locId] += self.history.states[t].s[:,l]
                    lItvCost[:,locId] += self.history.states[t].c[:,l]
                    lSu[locId] += sus[l]
                    lAlive[locId] += alive[l]
                    lInf[:,locId] += self.history.difParameters[t].inf[:,l]*alive[l]
                    coef = self.history.parameters[t].f[l]*alive[l][np.newaxis,:]
                    # if t == 0:
                    #     print(l, self.history.parameters[t].p[:,l], coef)
                    lAverageP[:,locId] += np.sum(self.history.parameters[t].p[:,l]*coef[np.newaxis,:], axis=(1,2))
                    lSumP[:,locId] += np.sum(coef)
                    lTran[:,:,locId] = np.sum(self.history.states[t].trans[:,:,l], axis=2)

                for i in range(self.static.nInterventions):
                    for action in actions[i]:
                        #print("HERE", action.locale)
                        locales = self.parser.parseLRegex(action.locale)
                        #print("OK")
                        for l in locales:
                            if l in contains:
                                totalAlive = np.sum(alive[l])
                                for k, v in enumerate(action.cpv):
                                    lParameter[locId][i][k] += v*totalAlive
                                    lParamCount[locId][i][k] += totalAlive
                for i in range(self.static.nInterventions):
                    for k in range(len(self.static.interventions[i].cps)):
                        if lParamCount[locId][i][k] > 0:
                            lParameter[locId][i][k] /= lParamCount[locId][i][k]
                        else:
                            lParameter[locId][i][k] = None
                lInf[:,locId] /= lAlive[locId]

            # if t == 0:
            #     print(lAverageP, lSumP)
            lAverageP = np.divide(lAverageP, lSumP, out=lAverageP.copy(), where=lSumP!=0)

            lIncidences.append(lIncidence)
            lNews.append(lNew)
            lCounts.append(lCount)
            lItvCosts.append(lItvCost)
            lSus.append(lSu)
            lAlives.append(lAlive)
            lInfs.append(lInf)
            lParameters.append(lParameter)
            lAveragePs.append(lAverageP)
            lTrans.append(lTran)

        # totalInfTime += np.sum(getInfectiousMatrix(self.static, self.curState.s))
        # averageDuration = totalInfTime/totalInfNew
        # totalInfNew += np.sum(getInfectiousMatrix(self.static, nextState.new-self.curState.new))
        #     loc["r_eff"] = 0 if scount < EPSILON else snew / scount * convergedDuration

        # New way
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325187/
        # instantRs = []
        # for t in range(T):
        #     instantR = np.zeros(len(self.locales), dtype=npFloat)
        #     for s in range(1, t+1):
        #         instantR += getInfectivity(s)*np.sum(lIncidences[t-s], axis=1)
        #     It = np.sum(lIncidences[t], axis=1)
        #     for l in range(len(self.locales)):
        #         if instantR[l] > 0:
        #             instantR[l] = It[l] / instantR[l]
        #     instantRs.append(instantR)

        # Old way
        instantRs = []
        lWindowIncidence = np.zeros(len(self.locales), dtype=npFloat)
        lWindowInfCount = np.zeros(len(self.locales), dtype=npFloat)
        windowInfDuration = 0
        # New way v1
        # baseRs = []
        # lWindowSusCount = np.zeros(len(self.locales), dtype=npFloat)
        # lWindowAlive = np.zeros(len(self.locales), dtype=npFloat)
        # New way v2
        # Ref: https://web.stanford.edu/~jhj1/teachingdocs/Jones-on-R0.pdf
        baseRs = []

        for t in range(T):
            instantR = np.zeros(len(self.locales), dtype=npFloat)
            baseR = np.zeros(len(self.locales), dtype=npFloat)
            # New way v1
            # si = np.sum(getSumTagMatrix(self.static, lCounts[t], SUS_TAG) * getSumTagMatrix(self.static, lCounts[t], INF_TAG), axis=1)
            # beta = np.sum(lIncidences[t]*lAlives[t], axis=1)

            if t >= R_WINDOW:
                lCount = lCounts[t] - lCounts[t-R_WINDOW]
                lWindowIncidence += np.sum(lIncidences[t], axis=1) - np.sum(lIncidences[t-R_WINDOW], axis=1)
                lWindowInfCount += np.sum(getSumTagMatrix(self.static, lCount, INF_TAG), axis=1)
                windowInfDuration += averageInfDuration[t] - averageInfDuration[t-R_WINDOW]
                # New way v1
                # lWindowSusCount += np.sum(getSumTagMatrix(self.static, lCount, SUS_TAG), axis=1)
                # lWindowAlive += np.sum(lAlives[t], axis=1) - np.sum(lAlives[t-R_WINDOW], axis=1)
            else:
                lWindowIncidence += np.sum(lIncidences[t], axis=1)
                #print(lWindowInfCount.shape, lCounts[t].shape)
                lWindowInfCount += np.sum(getSumTagMatrix(self.static, lCounts[t], INF_TAG), axis=1)
                windowInfDuration += averageInfDuration[t]
                # New way v1
                # lWindowSusCount += np.sum(getSumTagMatrix(self.static, lCounts[t], SUS_TAG), axis=1)
                # lWindowAlive += np.sum(lAlives[t], axis=1)

            # New way v2
            sumWeights = 0
            finalBeta = np.zeros(len(self.locales), dtype=npFloat)
            for h in self.static.hashedNewInfEdges:
                c1 = h // self.static.nCompartments
                c2 = h % self.static.nCompartments
                weight = 1 + lTrans[t][c1, c2, l]
                beta = np.zeros(len(self.locales), dtype=npFloat)
                for infEdge in self.static.infEdges.values():
                    for comId, coef in infEdge.coms.items():
                        if infEdge.susId == c1 and comId == c2:
                            add = lAveragePs[t][infEdge.paramId]
                            #print("ADD1", add)
                            for p0 in infEdge.numerParamIds:
                                add *= lAveragePs[t][p0]
                            #print("ADD2", add)
                            for p0 in infEdge.denomParamIds:
                                for l in range(len(self.locales)):
                                    if lAveragePs[t][p0, l] > 0:
                                        add[l] /= lAveragePs[t][p0, l]
                            #print("ADD3", add, coef)
                            beta += add*coef
                finalBeta += beta*weight
                sumWeights += weight
            if sumWeights > 0:
                finalBeta /= sumWeights

            for l in range(len(self.locales)):
                if lWindowInfCount[l] > 0:
                    instantR[l] = max(lWindowIncidence[l] / lWindowInfCount[l] * windowInfDuration / min(t+1, R_WINDOW), 0)
                # New way v1
                # if lWindowInfCount[l]*lWindowSusCount[l] > 0:
                #     baseR[l] = max(lWindowIncidence[l] * lWindowAlive[l] / (lWindowInfCount[l]*lWindowSusCount[l]) * windowInfDuration / min(t+1, R_WINDOW), 0)
                    #print("OK",beta[l] / si[l])
                    #baseR[l] = beta[l] / si[l] * windowInfDuration / min(t+1, R_WINDOW)
                # New way v2
                baseR[l] = finalBeta[l] * windowInfDuration / min(t+1, R_WINDOW)

            instantRs.append(instantR)
            baseRs.append(baseR)
            #print("HERE1", t, averageInfDuration[t], [round(x,10) for x in instantR], [round(x,10) for x in baseR], [round(x,10) for x in lWindowIncidence], [round(x,10) for x in lWindowInfCount], [round(x,10) for x in finalBeta])

        caseRs = []
        for t in range(T):
            caseR = np.zeros(len(self.locales), dtype=npFloat)
            for u in range(t, T):
                caseR += instantRs[u]*getInfectivity(u-t)
            caseRs.append(caseR)

        #print(np.array(instantRs).reshape(-1))
        #print(np.array(caseRs).reshape(-1))

        # Old way
        # baseRs = []
        # for t in range(T):
        #     baseR = instantRs[t].copy()
        #     for l in range(len(self.locales)):
        #         if np.sum(lSus[t][l]) > 0:
        #             baseR[l] = instantRs[t][l]*np.sum(lAlives[t][l])/np.sum(lSus[t][l])
        #     print("HERE2", t, baseR)
        #     baseRs.append(baseR)

        responses = []
        for t in range(T):
            result = []
            for l, locale in enumerate(self.locales):
                # Add initial infection as incidence!
                # incidenceValue = np.sum(lIncidences[t][l])
                # if t == 0:
                #     totalInf = getSumTagMatrix(self.static, self.static.s, INF_TAG)
                #     for leafId in contains:
                #         incidenceValue += np.sum(totalInf[leafId])
                loc = {
                    "id":locale["id"],
                    "name":locale["name"],
                    "parent_id":locale["parent_id"],
                    "incidence":np.sum(lIncidences[t][l]),
                    "r_instant":instantRs[t][l],
                    "r_case":caseRs[t][l],
                    "r_0":baseRs[t][l],
                    "compartments":[],
                    "interventions":[]
                }
                for c in range(self.static.nCompartments):
                    com = {
                        "compartment_name":self.static.getPropName("compartment", c),
                        "groups":[]
                    }
                    for g in range(self.static.nGroups):
                        group = {
                            "group_name": self.static.getPropName("group", g),
                            "count": lCounts[t][c, l, g],
                            "new": lNews[t][c, l, g]
                        }
                        com["groups"].append(group)
                    loc["compartments"].append(com)

                for i in range(self.static.nInterventions):
                    #print("HERE", t, i, np.sum(lItvCosts[t][i]))
                    intervention = {
                        "intervention_name":self.static.getPropName("intervention", i),
                        "cost": lItvCosts[t][i, l] if t == 0 else lItvCosts[t][i, l] - lItvCosts[t-1][i, l],
                        "groups":[],
                        "parameter":[]
                    }
                    for g in range(self.static.nGroups):
                        ginf = lInfs[t+1][i,l,g] if t < T-1 else 0
                        group = {
                            "group_name": self.static.getPropName("group", g),
                            "inf_count": ginf*lAlives[t][l,g],
                            "inf": ginf
                        }
                        intervention["groups"].append(group)

                    for k, cp in enumerate(self.static.interventions[i].cps):
                        if lParameters[t][l][i][k] is not None:
                            parameter = {
                                "name": cp.name,
                                "value": lParameters[t][l][i][k]
                            }
                            intervention["parameter"].append(parameter)

                    loc["interventions"].append(intervention)

                result.append(loc)

            response = {
                "day":t,
                "simulation_id":self.simulationId,
                "result":result,
            }
            # if t > 60 and t < 70:
            #     print(response)
            #print(response)
            responses.append(response)

        return responses

        # difParameter = self.static.combineActions(actions)
        # convergedDuration = self.history.get(-1)[2]

        # N = getAliveMatrix(self.static, curState.s)
        # N[np.where(N == 0)] = 1
        # Inf = difParameter.inf*curState.s[np.newaxis, :] + difParameter.finf
        # Inf = np.minimum(Inf, curState.s[np.newaxis, :])

        # result = []
        # for locId, locale in enumerate(self.locales):
        #     loc = {}
        #     loc["id"] = locale["id"]
        #     loc["name"] = locale["name"]
        #     loc["parent_id"] = locale["parent_id"]
        #
        #     contains = self.static.localeHierarchy[normalize(loc["name"])]
        #
        #     n = np.zeros(self.static.nGroups, dtype=npFloat)
        #     for i in contains:
        #         n += N[i]
        #
        #     snew = 0
        #     scount = 0
        #     s = 0
        #
        #     coms = []
        #     for j in range(self.static.nCompartments):
        #         com = {}
        #         com["compartment_id"] = self.compartments[j]["id"]
        #         com["compartment_name"] = self.compartments[j]["name"]
        #         com["groups"] = []
        #         for name, g in self.static.groupToIndex.items():
        #             count = 0
        #             new = 0
        #             for i in contains:
        #                 count += curState.s[j,i,g]
        #                 new += (0 if preState is None else curState.new[j,i,g]-preState.new[j,i,g])
        #             dict = {
        #                 "group_name": name,
        #                 "count": count,
        #                 "new": new
        #             }
        #             if self.static.ivec[j] > 0:
        #                 scount += count
        #                 snew += new
        #             if self.static.svec[j] > 0:
        #                 s += count
        #             com["groups"].append(dict)
        #         coms.append(com)
        #
        #     loc["compartments"] = coms
        #
        #     loc["r_eff"] = 0 if scount < EPSILON else snew / scount * convergedDuration
        #     loc["r_0"] = 0 if s < EPSILON else loc["r_eff"] * np.sum(n) / s

            # interventions = []
            # for j in range(self.static.nInterventions):
            #     if j in actions:
            #         groups = []
            #         for groupName, groupId in self.static.groupToIndex.items():
            #             cost = 0
            #             inf = 0
            #             for locId in contains:
            #                 localeCost = np.sum(curState.c[j, :, locId, groupId])
            #                 if preState is not None:
            #                     localeCost -= np.sum(preState.c[j, :, locId, groupId])
            #                 cost += localeCost
            #                 inf += np.sum(Inf[j, :, locId, groupId])
            #             group = {
            #                 "group_name":groupName,
            #                 "cost":cost,
            #                 "inf":0 if n[groupId] < EPSILON else min(inf/n[groupId], 1),
            #                 "inf_count": 0 if n[groupId] < EPSILON else inf
            #             }
            #             groups.append(group)
            #         parameters = []
            #         cps = self.static.interventions[j].cps
            #         for cpId, cp in enumerate(actions[j].cpv):
            #             parameters.append(
            #                 {
            #                     "id": cpId+1,
            #                     "name": cps[cpId].name,
            #                     "value": cp
            #                 }
            #             )
            #         intervention = {
            #             "intervention_name":self.static.interventions[j].name,
            #             "groups":groups,
            #             "parameters":parameters
            #         }
            #         interventions.append(intervention)
            # loc["interventions"] = interventions

        #     result.append(loc)
        #
        # response = {
        #     "day": q,
        #     "simulation_id":self.simulationId,
        #     "result":result,
        # }

    def getLocaleName(self, index):
        for loc in self.locales:
            contains = self.static.localeHierarchy[normalize(loc["name"])]
            if len(contains) == 1:
                for i in contains:
                    if i == index:
                        return loc["name"]

    # def combineActions(self, actions):
    #     difParameter = initializeDifParameter(self.static)
    #     for action in actions.values():
    #         difParameter.combine()
    #     return difParameter

    def getNextStateIteratively(self, curState, schedule, method='sp_RK23'):
        state = curState
        for t in np.arange(schedule.tSpan[0], schedule.tSpan[1]):
            finerSchedule = getFinerSchedule(schedule, (t, t+1))
            state = self.getNextState(state, finerSchedule, method)
        return state

    def getNextState(self, curState, schedule, method='sp_RK23'):
        scheduledParameter = makeScheduledParameter(self.static, schedule)
        scale = schedule.tSpan[1] - schedule.tSpan[0]
        scaledTSpan = (schedule.tSpan[0]/scale, schedule.tSpan[1]/scale)
        package, solver = method.split('_')
        if package == 'sp':
            def f(t, S):
                unscaledT = t * scale
                P = copyParameter(scheduledParameter.measure(unscaledT))
                P.p *= scale
                dS = forward(self.static, scale, S, P)
                return dS

            res = solve_ivp(f, scaledTSpan, curState.flatten(), method=solver)
            #res = solve_ivp(f, scaledTSpan, curState.flatten(), method='BDF')
            nextState = makeState(self.static, res.y[:, -1])
            return nextState
        elif package == 'jl':
            p = (self.static, scheduledParameter)
            prob = de.ODEProblem(f_b, curState.flatten(), scaledTSpan, p)
            sol = de.solve(prob, de.BS3(), abstol = 1e-1, reltol = 1e-1, save_everystep=False)
            nextState = makeState(self.static, sol.u[-1])
            return nextState

    def getNextFullState(self, curState, schedule, isMain=False, method='sp_RK45'):
        scheduledParameter, D1, P1 = makeScheduledFullParameter(self, schedule)
        self.P = scheduledParameter.P0
        if isMain:
            self.history.difParameters.append(D1)
            self.history.parameters.append(P1)
            #print(len(self.history.difParameters), np.sum(D0.inf))
        scale = schedule.tSpan[1] - schedule.tSpan[0]
        scaledTSpan = (schedule.tSpan[0]/scale, schedule.tSpan[1]/scale)
        package, solver = method.split('_')
        if package == 'sp':
            def fullF(t, S):
                unscaledT = t * scale
                P = copyFullParameter(scheduledParameter.measure(unscaledT))
                P.p *= scale
                dS = fullForward(t, self.static, scale, S, P)
                return dS
            #print("WTFx")
            res = solve_ivp(fullF, scaledTSpan, curState.flatten(), method=solver)
            #print("WTFy")
            nextState = makeFullState(self.static, res.y[:, -1])
            return nextState
        elif package == 'jl':
            p = (self.static, scheduledParameter)
            prob = de.ODEProblem(fullF_b, curState.flatten(), scaledTSpan, p)
            sol = de.solve(prob, de.BS3(), abstol = 1e-1, reltol = 1e-1, save_everystep=False)
            nextState = makeFullState(self.static, sol.u[-1])
            return nextState

    def reset(self):
        D0 = initializeDifParameter(self.static)
        self.P = makeFullParameter(self.static, D0)
        self.q = 0
        self.curState = makeInitialFullState(self.static, self.static.s.copy())
        self.prevActions = self.static.A0
        return self.curState

    def step(self, actions):
        finerSchedule = makeUnitSchedule(self.static, self.prevActions, self.q, actions)
        nextState = self.getNextFullState(self.curState, finerSchedule, isMain=False, method=DEBUG_METHOD)
        r = - np.sum(nextState.c - self.curState.c)
        self.prevActions = actions
        self.q += 1
        self.curState = nextState
        return nextState, r, self.q > self.T

    def run(self, T=None, debugMode=NORMAL, interval=DEBUG_INTERVAL, method=DEBUG_METHOD):
        #print(self.static.localeToIndex)
        D0 = initializeDifParameter(self.static)
        self.P = makeFullParameter(self.static, D0)
        if T is None:
            T = self.T
        self.currentFolder = Path(__file__).parent
        self.basePath = Path(__file__).parent.parent.parent
        self.history = History()
        self.q = 0
        self.interval = interval
        self.curState = makeInitialFullState(self.static, self.static.s.copy())
        self.history.difParameters.append(D0)
        self.history.parameters.append(self.P)
        self.history.add(None, self.curState)

        # self.tempRes = {}
        # self.tempRes["interventions"] = self.session["interventions"]
        # schedules = []
        #
        # if debugMode == DEBUG:
        #     self.infs = []
        #     self.hosp = []
        #     self.deaths = []
        #

        #totalInfTime = 0
        #totalInfNew = np.sum(getSumTagMatrix(self.static, self.curState.s, INF_TAG))

        if debugMode == DEBUG:
            self.filePath = (self.basePath / "test/core/results/debug.json").resolve()
            self.debugFile = open(self.filePath, 'w')
            self.initialDebug(self.curState)
            #
            # self.infs.append(round(np.sum(getInfectiousMatrix(self.static, self.curState.s)[3]), 2))
            # self.hosp.append(round(np.sum(getHospitalizedMatrix(self.static, self.curState.s)[3]), 2))
            # self.deaths.append(round(np.sum(getDeathMatrix(self.static, self.curState.s)[3]), 2))
            #
        elif debugMode == NORMAL:
            self.conn.update(self.simulationId, "running")
        elif debugMode == WARMUP:
            pass

        for q in np.arange(0, T - self.interval + 1, self.interval):
            tSpan = (q, q + self.interval)
            if debugMode == NORMAL:
                # temporarily
                finerSchedule = getFinerSchedule(self.static, self.static.schedule, tSpan)
            elif debugMode == DEBUG:
                if self.runMode == RUN_MCTS:
                    #MCTS
                    prevSchedule, prevState, _ = self.history.get(q)
                    prevActions = self.static.A0 if q == 0 else self.getActionsInf(prevSchedule)[0]
                    if q % self.mcts.interval == 0:
                        actions = self.mcts.makeDecision(q, prevActions, makeStateFrom(prevState))
                        finerSchedule = makeUnitSchedule(self.static, prevActions, q, actions)
                    else:
                        finerSchedule = makeUnitSchedule(self.static, prevActions, q, prevActions)
                else:
                    #Normal
                    finerSchedule = getFinerSchedule(self.static, self.static.schedule, tSpan)
            elif debugMode == WARMUP:
                finerSchedule = getFinerSchedule(self.static, self.static.schedule, tSpan)
            #print("WTF1")
            #TempRes
#             actions, _ = self.getActionsInf(finerSchedule)
#             for itvId, act in actions.items():
#                 index = len(schedules) + 1
#                 itv = self.static.interventions[itvId]
#                 today = self.startDate + timedelta(days=int(q))
#                 repToday = "{}-{}-{}".format(today.year, today.month, today.day)
#                 cps = []
#                 for v in act.cpv:
#                     cps.append({
#                         "id":1,
#                         "name": "degree",
#                         "value": v
#                     })
#                 mySchedule = {
#                     "id": index,
#                     "name": "Schedule " + str(index),
#                     "intervention": itvId+1,
#                     "intervention_name": itv.name,
#                     "type": "fixed",
#                     "notes": "...",
#                     "days": [repToday, repToday],
#                     "condition": "def isActive():\n\n",
#                     "control_params": cps
#                 }
#                 schedules.append(mySchedule)
            #TempRes

            # totalInfTime += np.sum(getInfectiousMatrix(self.static, self.curState.s))
            # averageDuration = totalInfTime/totalInfNew
            nextState = self.getNextFullState(self.curState, finerSchedule, isMain=True, method=method)
            # totalInfNew += np.sum(getInfectiousMatrix(self.static, nextState.new-self.curState.new))
            #
            self.history.add(finerSchedule, nextState)
            self.q += self.interval

            if debugMode != WARMUP:
                self.recurringDebug(self.q, self.curState, nextState, finerSchedule)
                #
                # if debugMode == DEBUG:
                #     self.infs.append(round(np.sum(getInfectiousMatrix(self.static, nextState.s)[3]), 2))
                #     self.hosp.append(round(np.sum(getHospitalizedMatrix(self.static, nextState.s)[3]), 2))
                #     self.deaths.append(round(np.sum(getDeathMatrix(self.static, nextState.s)[3]), 2))
                #
                # print(self.curState.s[0, :, 2])
                # print(self.curState.s[-1, :, 2])
                # print([self.getLocaleName(i) for i in range(self.static.nLocales)])
                print(self.q, np.sum(self.curState.trans[0,1]), np.sum(self.curState.s), [(self.compartments[i]['name'], np.sum(self.curState.s[i])) for i in range(self.static.nCompartments)])

            self.curState = nextState
            #print("WTF5")


        if debugMode == DEBUG:
            responses = self.getResponses()
            # self.dout = np.sum(self.curState.s[7])
            # self.iout = np.sum(self.curState.new[1])
            # self.cout = np.sum(self.curState.c[:,:])
            # locId = "ARE"
            # for q in range(1, self.history.getSize()):
            #     preState = self.history.get(q-1)[1]
            #     finerSchedule, curState, _ = self.history.get(q)
            #     response = self.getRecurringResponse(q, preState, curState, finerSchedule)
            #     for loc in response["result"]:
            #         if loc["id"] == locId:
            #             print(loc['r_eff'], loc['r_0'])
            #self.tempRes["schedules"] = schedules
            #json.dump(self.tempRes, self.debugFile)
            #os.rename(self.filePath, (self.basePath / "test/core/results/{}-{}.txt".format(method, interval)).resolve())
            self.debugFile.close()
            return responses
        elif debugMode == NORMAL:
            responses = self.getResponses()
            threads = [self.conn.post(response) for response in responses]
            # for q in range(1, self.history.getSize()):
            #     preState = self.history.get(q-1)[1]
            #     finerSchedule, curState = self.history.get(q)
            #     response = self.getRecurringResponse(q, preState, curState, finerSchedule)
            #     # if q == 350 or q == 351:
            #     #     print(response)
            #     threads.append(self.conn.post(response))
            for thread in threads:
                if thread is not None:
                    thread.join()
            print("All responses sent")
            self.conn.update(self.simulationId, "finished")
        elif debugMode == WARMUP:
            pass

        if self.runMode == RUN_MCTS:
            self.mcts.close()

    def getActions(self, finerSchedule):
        q = finerSchedule.tSpan[0]
        if q in finerSchedule.ticks:
            actions = finerSchedule.ticks[q]
        else:
            actions = finerSchedule.A0
        return actions

    # def getInitialResponse(self, curState):
    #     return self.getResponse(0, None, curState, self.static.generateEmptyActions())
    #
    # def getRecurringResponse(self, q, preState, curState, finerSchedule):
    #     actions = self.getActions(finerSchedule)
    #     return self.getResponse(q, preState, curState, actions)

    def initialDebug(self, curState):
        self.debug(0, None, curState, self.static.generateEmptyActions())

    def recurringDebug(self, q, preState, curState, finerSchedule):
        actions = self.getActions(finerSchedule)
        self.debug(q, preState, curState, actions)

    def debug(self, q, preState, curState, actions):
        print(printActions(self.static, preState, curState, actions))
