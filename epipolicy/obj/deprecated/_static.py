from numba.experimental import jitclass
from numba.types import ListType, DictType, string, UniTuple
from numba.typed import Dict, List
from .np_wrapper import Np2f, np2fType, listNp2fType
from ..utility.singleton import *
from ..sparse.sparse import SparseMatrix, sparseMatrixType
from ..sparse.coo import CooMatrix, cooMatrixType, listCooType, makeCooMatrix2DList
from .edge import Edge, edgeType, InfEdge, infEdgeType
from .intervention import Intervention, interventionType, listInterventionType
from .full_parameter import FullParameter, fullParameterType, initializeFullParameter
from .dif_parameter import DifParameter, difParameterType, initializeDifParameter
from .action import *
from .schedule import scheduleType
from .regsult import setType
import numpy as np

precomputedDictType = DictType(nbHashedInt, difParameterType)
stringIntDictType = DictType(string, nbInt)
stringListType = ListType(string)

staticSpec = [
    ('localeHierarchy', DictType(string, setType)),
    ('properties', stringListType),
    ('toIndex', DictType(string, stringIntDictType)),
    ('toName', DictType(string, stringListType)),
    # ('parameterToIndex', DictType(string, nbInt)),
    # ('compartmentToIndex', DictType(string, nbInt)),
    # ('localeToIndex', DictType(string, nbInt)),
    # ('modeToIndex', DictType(string, nbInt)),
    # ('groupToIndex', DictType(string, nbInt)),
    # ('facilityToIndex', DictType(string, nbInt)),
    # ('interventionToIndex', DictType(string, nbInt)),
    # ('tagToIndex', DictType(string, nbInt)),
    ('coo', ListType(listCooType)),
    # ('sumCoo', cooMatrixType), # Uninitialized
    ('csr', ListType(sparseMatrixType)),
    ('sumCsr', sparseMatrixType), # Uninitialized
    ('sumCsc', sparseMatrixType), # Uninitialized
    #('jacSparsity', sparseMatrixType), # Uninitialized
    ('bias', nbFloat[:]),
    ('compartmentTags', nbInt[:,:]),
    ('parameterTags', nbInt[:,:]),
    # ('avec', nbFloat[:]),
    # ('ivec', nbFloat[:]),
    # ('svec', nbFloat[:]),
    # ('hvec', nbFloat[:]),
    # ('initId', nbInt), # Unitialized
    #('csrData', listNp2fType),
    ('P', fullParameterType),
    ('s', nbFloat[:,:,:]),
    #('p', nbFloat[:,:,:,:]),
    #('f', nbFloat[:,:,:]),
    ('A0', actionsType),
    #('ft', nbFloat[:,:,:,:]),
    ('sN', nbFloat[:,:]),
    ('pN', nbFloat[:,:]),
    ('schedule', scheduleType), # Unitialized
    ('nLocales', nbInt),
    ('nCompartments', nbInt),
    ('nParameters', nbInt),
    ('nGroups', nbInt),
    ('nFacilities', nbInt),
    ('nInterventions', nbInt),
    ('nModes', nbInt),
    ('nCompartmentTags', nbInt),
    ('nParameterTags', nbInt),
    ('shape', UniTuple(nbInt, 2)),
    ('edges', DictType(string, edgeType)),
    ('infEdges', DictType(string, infEdgeType)),
    ('hashedNewInfEdges', setType),
    ('interventions', listInterventionType),
    #('precomputedInterventions', DictType(nbInt, precomputedDictType)),
]

@jitclass(staticSpec)
class Static:
    def __init__(self, nLocales, nCompartments, nParameters, nGroups, nFacilities, nInterventions, nModes, nCompartmentTags, nParameterTags):
        self.localeHierarchy = Dict.empty(string, setType)

        self.properties = List.empty_list(string)
        self.properties.append('parameter')
        self.properties.append('compartment')
        self.properties.append('locale')
        self.properties.append('mode')
        self.properties.append('group')
        self.properties.append('facility')
        self.properties.append('intervention')
        self.properties.append('compartmentTag')
        self.properties.append('parameterTag')

        self.nLocales = nLocales
        self.nCompartments = nCompartments
        self.nParameters = nParameters
        self.nGroups = nGroups
        self.nFacilities = nFacilities
        self.nInterventions = nInterventions
        self.nModes = nModes
        self.nCompartmentTags = nCompartmentTags
        self.nParameterTags = nParameterTags
        self.shape = (self.nLocales, self.nLocales)

        self.toIndex = Dict.empty(string, stringIntDictType)
        self.toName = Dict.empty(string, stringListType)
        for prop in self.properties:
            self.toIndex[prop] = Dict.empty(string, nbInt)
            self.toName[prop] = List.empty_list(string)
        # self.parameterToIndex = Dict.empty(string, nbInt)
        # self.compartmentToIndex = Dict.empty(string, nbInt)
        # self.localeToIndex = Dict.empty(string, nbInt)
        # self.modeToIndex = Dict.empty(string, nbInt)
        # self.groupToIndex = Dict.empty(string, nbInt)
        # self.facilityToIndex = Dict.empty(string, nbInt)
        # self.interventionToIndex = Dict.empty(string, nbInt)
        # self.tagToIndex = Dict.empty(string, nbInt)

        self.coo = makeCooMatrix2DList(self.nModes, self.nGroups, self.shape)

        self.csr = List.empty_list(sparseMatrixType)
        # self.csrData = List.empty_list(np2fType)

        self.bias = np.zeros(self.nModes, dtype=npFloat)
        self.compartmentTags = np.zeros((self.nCompartments, self.nCompartmentTags), dtype=npInt)
        self.parameterTags = np.zeros((self.nParameters, self.nParameterTags), dtype=npInt)
        # self.avec = np.ones(self.nCompartments, dtype=npFloat)
        # self.ivec = np.zeros(self.nCompartments, dtype=npFloat)
        # self.svec = np.zeros(self.nCompartments, dtype=npFloat)
        # self.hvec = np.zeros(self.nCompartments, dtype=npFloat)
        self.s = np.zeros((self.nCompartments, self.nLocales, self.nGroups), dtype=npFloat)
        self.P = initializeFullParameter(self.nLocales, self.nParameters, self.nGroups, self.nFacilities, self.nInterventions)
        self.sN = np.zeros((self.nLocales, self.nGroups), dtype=npFloat)
        self.pN = np.zeros((self.nLocales, self.nGroups), dtype=npFloat)
        # self.p = np.zeros((self.nParameters, self.nLocales, self.nFacilities, self.nGroups), dtype=npFloat)
        self.A0 = List.empty_list(listActionType)
        for i in range(self.nInterventions):
            self.A0.append(List.empty_list(actionType))

        # self.f = np.ones((self.nLocales, self.nFacilities, self.nGroups), dtype=npFloat) / self.nFacilities
        # self.ft = np.ones((self.nLocales, self.nFacilities, self.nGroups, self.nGroups), dtype=npFloat) / self.nGroups

        self.edges = Dict.empty(string, edgeType)
        self.infEdges = Dict.empty(string, infEdgeType)
        self.hashedNewInfEdges = Dict.empty(nbInt, nbBool)

        self.interventions = List.empty_list(interventionType)
        #self.precomputedInterventions = Dict.empty(nbInt, precomputedDictType)

    def addProp(self, prop, name):
        if name not in self.toIndex[prop]:
            index = len(self.toName[prop])
            self.toName[prop].append(name)
            self.toIndex[prop][name] = index
        return self.toIndex[prop][name]

    def getPropName(self, prop, index):
        return self.toName[prop][index]

    def getPropIndex(self, prop, name):
        return self.toIndex[prop][name]

    def hasPropName(self, prop, name):
        return name in self.toIndex[prop]

    # def compartmentHasTag(self, comId, tag):
    #     if self.hasPropName("compartmentTag", tag):
    #         return self.compartmentTags[comId, self.getPropIndex("compartmentTag", tag)] > 0
    #     return False
    #
    # def parameterHasTag(self, paramId, tag):
    #     if self.hasPropName("parameterTag", tag):
    #         return self.paramaterTags[paramId, self.getPropIndex("parameterTag", tag)] > 0
    #     return False

    # def lookupAction(self, action):
    #     if action.hash in self.precomputedInterventions[action.id]:
    #         return self.precomputedInterventions[action.id][action.hash]
    #     return initializeDifParameter(self)

    def getNRealInterventions(self):
        count = 0
        for itv in self.interventions:
            if not itv.isCost:
                count += 1
        return count

    # def hashActions(self, actions):
    #     hash = 0
    #     for itvId in range(len(self.interventions)-1,-1,-1):
    #         itv = self.interventions[itvId]
    #         if itvId in actions:
    #             hash += actions[itvId].hash
    #             if itvId == 0:
    #                 return hash
    #         hash *= itv.hashLimit
    #
    def generateZeroActions(self):
        actions = List.empty_list(listActionType)
        for itvId, itv in enumerate(self.interventions):
            actions.append(List.empty_list(actionType))
            actions[itvId].append(generateZeroAction(self, itvId))
        return actions

    def generateDefaultActions(self):
        actions = List.empty_list(listActionType)
        for itvId, itv in enumerate(self.interventions):
            actions.append(List.empty_list(actionType))
            actions[itvId].append(generateDefaultAction(self, itvId))
        return actions

    def generateEmptyActions(self):
        actions = List.empty_list(listActionType)
        for itvId, itv in enumerate(self.interventions):
            actions.append(List.empty_list(actionType))
            if itv.isCost:
                actions[itvId].append(generateDefaultAction(self, itvId))
        return actions
    #
    # def isActionsEqual(self, A1, A2):
    #     for itvId, itv in enumerate(self.interventions):
    #         h1 = 0
    #         h2 = 0
    #         if itvId in A1:
    #             h1 = A1[itvId].hash
    #         if itvId in A2:
    #             h2 = A2[itvId].hash
    #         if h1 != h2:
    #             return False
    #     return True

    def normalizeFacilityMatrix(self, F):
        normalizedF = np.zeros((self.nLocales, self.nFacilities, self.nGroups), dtype=npFloat)
        sumFacility = np.sum(F, axis=1)
        for l in range(self.nLocales):
            for g in range(self.nGroups):
                if sumFacility[l, g] > EPSILON:
                    for f in range(self.nFacilities):
                        normalizedF[l, f, g] = F[l, f, g] / sumFacility[l, g]
                        #print(l, f, g, normalizedF[l, f, g], flush=True)
                else:
                    # By default if everything is 0, the first facility gets all timespent
                    normalizedF[l, 0, g] = 1.0
        return normalizedF

    def normalizeContactMatrix(self, Ft):
        normalizedFt = Ft.copy()
        sumContact = np.sum(Ft, axis=3)
        for l in range(self.nLocales):
            for f in range(self.nFacilities):
                for g1 in range(self.nGroups):
                    #print(l, f, g1, sumContact[l, f, g1], flush=True)
                    if sumContact[l, f, g1] > 1.0:
                        for g2 in range(self.nGroups):
                            normalizedFt[l, f, g1, g2] = Ft[l, f, g1, g2] / sumContact[l, f, g1]
        return normalizedFt

    def getTotalIncidenceMatrix(self, fullState): # l x g
        res = np.zeros((self.nLocales, self.nGroups), dtype=npFloat)
        for h in self.hashedNewInfEdges:
            c1 = h // self.nCompartments
            c2 = h % self.nCompartments
            #print(fullState.trans[c1, c2], flush=True)
            res += fullState.trans[c1, c2]
            res -= fullState.trans[c2, c1]
        return res

    def getTotalNewMatrix(self, fullState): # c x l x g
        res = fullState.gain
        for c1 in range(self.nCompartments):
            for c2 in range(self.nCompartments):
                if c1 != c2:
                    add = fullState.trans[c2, c1] - fullState.trans[c1, c2]
                    if np.sum(add) > 0:
                        res[c1] += add
        return res

    def getTotalChangeMatrix(self, fullState): # c x l x g
        res = fullState.gain - fullState.lose
        for c1 in range(self.nCompartments):
            for c2 in range(self.nCompartments):
                if c1 != c2:
                    res[c1] -= fullState.trans[c1, c2]
                    res[c2] += fullState.trans[c1, c2]
        return res

staticType = Static.class_type.instance_type
