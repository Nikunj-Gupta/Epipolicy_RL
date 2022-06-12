from .mcts_utils import *
from ..obj.state import State, makeState, makeInitialState
from ..obj.schedule import makeUnitSchedule, Schedule
from ..obj.action import *
from ..utility.singleton import *
from ..utility.utils import getChoice
from numba.typed import Dict
from math import sqrt, log, inf
import random, os, time
from multiprocessing.sharedctypes import RawValue, RawArray, Value
from ctypes import c_int, c_longlong, c_ulonglong, byref, CDLL

STAGE_ONE = 1
"""
Condition: nExpandingChildren < nProjectedChildren
Description: We can create more children for this node.
Action:
    Assign workers to create children for this node
    If nExpandingChildren == nProjectedChildren:
        Next Stage
"""

STAGE_TWO = 2
"""
Condition: nExpandingChildren >= nProjectedChildren && nExpandedChildren == 0
Description: All projected children are expanding. None of them are finished yet.
Action:
    If current node is root then:
        wait until the node reach stage three
    else:
        start random playout at this node
"""

STAGE_THREE = 3
"""
Condition: nExpandedChildren > 0 && nEvaluatedChildren < nProjectedChildren
Description: At least one children is expanded but not all of them are evaluated.
Action:
    Randomly choose a child among the expanded ones
"""

STAGE_FOUR = 4
"""
Condition: nEvaluatedChildren >= nProjectedChildren
Description: All children are evaluated
Action:
    Use UCT to choose which child to go next
"""

STATUS_UNINITIALIZED = 0
STATUS_EXPANDED = 1
STATUS_REPEATED = 2
"""
The node has been detected to be repeated by one of its sibling.
"""


"""
Shared Memory Representation
Modes:
    =: Accessed parallel
    o: Set once and constant throughout
Format:
<Mode>:<Size>:<Name>:<Description>

# ULong
    =:1:w:TotalCost

# Int
    o:1:t:Day
    o:1:parent:Parent index
    o:1:nProjectedChildren:Number of projected children
    =:1:nExpandingChildren:Number of children being created by workers
    =:1:nExpandedChildren:Number of children had been created
    =:1:nEvaluatedChildren:Number of children had received playout
    =:1:nVisits:Number of visits
    :1:stage:Node's children construction stage
    :1:status:Node's running status
    :MaxChildren:children:Indices of children
    o:nInterventions*MaxCPPerAction:actions:Actions that results to this node

# State
    o:nCompartments*nLocales*nGroups:state:State of the node
    o:1:cost:Total cost so far at the node
"""

class NodeManager:
    def __init__(self, mcts):
        self.mcts = mcts
        self.epi = self.mcts.epi
        self.static = self.epi.static
        self.maxNode = computeMaxNode(self.static)
        self.initSharedMemory()
        self.initLibrary()
        self.bestCost = Value(c_ulonglong, MAX_UCOST, lock=True)
        self.nNodes = RawValue(c_int, 0)
        self.nPlayouts = RawValue(c_int, 0)
        self.nRequiredPlayouts = RawValue(c_int, -1)
        self.stopBy = RawValue(c_longlong, -1)
        self.clear()

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['static']
        del attributes['epi']
        del attributes['libint']
        del attributes['liblong']
        del attributes['libulong']
        return attributes

    def clear(self):
        self.nNodes.value = 0
        self.nPlayouts.value = 0
        self.nRequiredPlayouts.value = self.mcts.iBudget
        self.stopBy.value = -1
        if self.mcts.tBudget >= 0:
            self.stopBy.value = time.time() + self.mcts.tBudget
        self.ulong[:] = np.zeros(self.ulongSize*self.maxNode, dtype=np.uint64)
        self.int[:] = np.zeros(self.intSize*self.maxNode, dtype=np.int32)
        self.long[:] = np.zeros(self.longSize*self.maxNode, dtype=np.int64)

    def initLibrary(self):
        dirPath = os.path.dirname(os.path.realpath(__file__))
        self.libint = CDLL(os.path.join(dirPath, "atomic/libint.so"))
        self.liblong = CDLL(os.path.join(dirPath, "atomic/liblong.so"))
        self.libulong = CDLL(os.path.join(dirPath, "atomic/libulong.so"))

    def initSharedMemory(self):
        self.ulongSize = 1
        self.ulong = RawArray(c_ulonglong, self.ulongSize*self.maxNode)

        self.maxChildren = max(self.computeNProjectedChildrenForRoot(), self.computeNProjectedChildrenForNonRoot())
        self.intSize = 9 + self.maxChildren + self.static.nInterventions * MAX_CP_PER_ACTION
        self.int = RawArray(c_int, self.intSize*self.maxNode)

        self.longSize = self.static.nCompartments*self.static.nLocales*self.static.nGroups+1
        self.long = RawArray(c_longlong, self.longSize*self.maxNode)

    def updateBestCost(self, cost):
        ucost = int(cost * (10 ** ULONG_PRECISION))
        with self.bestCost.get_lock():
            if self.bestCost.value > ucost:
                self.bestCost.value = ucost

    def getBestCost(self):
        return self.bestCost.value / (10 ** ULONG_PRECISION)

    def getNNodes(self):
        return self.nNodes.value

    def getNPlayouts(self):
        nPlayouts = c_int(0)
        self.libint.load(byref(self.nPlayouts), byref(nPlayouts))
        return nPlayouts.value

    def isPlayoutFinished(self):
        startTime = time.time()
        if self.stopBy.value >= 0 and startTime > self.stopBy.value:
            return True
        if self.nRequiredPlayouts.value >= 0 and self.nRequiredPlayouts.value <= self.getNPlayouts():
            return True
        return False

    def addNNodes(self):
        return self.libint.fetch_add(byref(self.nNodes), byref(c_int(1)))

    def addNPlayouts(self):
        return self.libint.fetch_add(byref(self.nPlayouts), byref(c_int(1)))

    def getState(self, index):
        longFlat = np.frombuffer(self.long, dtype=npLong, count=self.longSize, offset=index * self.longSize * LONG_SIZE)
        flat = longFlat.astype(npFloat) / (10**LONG_PRECISION)
        return makeState(self.static, flat)

    def setState(self, index, state):
        src = np.frombuffer(self.long, dtype=npLong, count=self.longSize, offset=index * self.longSize * LONG_SIZE)
        dest = state.flatten() * (10**LONG_PRECISION)
        src[:] = dest.astype(npLong)

    def getT(self, index):
        return self.int[index * self.intSize]

    def setT(self, index, t):
        self.int[index * self.intSize] = t

    def getParent(self, index):
        return self.int[index * self.intSize + 1]

    def setParent(self, child, parent):
        self.int[child * self.intSize + 1] = parent

    def setStage(self, index, stage):
        self.int[index * self.intSize + 7] = stage

    def getStage(self, index):
        return self.int[index * self.intSize + 7]

    def setStatus(self, index, status):
        self.int[index * self.intSize + 8] = status

    def getStatus(self, index):
        return self.int[index * self.intSize + 8]

    def getChildToParent(self, parent, childIndex):
        return self.int[parent * self.intSize + 9 + childIndex]

    def addChildToParent(self, child, parent, childIndex):
        self.int[parent * self.intSize + 9 + childIndex] = child

    def getNProjectedChildren(self, index):
        return self.int[index * self.intSize + 2]

    def setNProjectedChildren(self, index, nProjectedChildren):
        self.int[index * self.intSize + 2] = nProjectedChildren

    def getActions(self, index):
        actions = Dict.empty(nbInt, actionType)
        offset = index * self.intSize + 9 + self.maxChildren
        for itvId, itv in enumerate(self.static.interventions):
            actOffset = offset + itvId * MAX_CP_PER_ACTION
            cpi = np.zeros(len(itv.cps), dtype=npInt)
            for cpId in range(len(cpi)):
                cpi[cpId] = self.int[actOffset + cpId]
            actions[itvId] = Action(self.static, itvId, cpi)
        return actions

    def setActions(self, index, actions):
        offset = index * self.intSize + 9 + self.maxChildren
        for itvId in range(self.static.nInterventions):
            if itvId in actions:
                actOffset = offset + itvId * MAX_CP_PER_ACTION
                for cpId, v in enumerate(actions[itvId].cpi):
                    self.int[actOffset + cpId] = v

    def createRoot(self, epochT, prevActions, prevState):
        self.epochT = epochT
        self.addNNodes()
        self.createChild(-1, 0, -1, prevActions, prevState)

#     def getDistanceFromRoot(self, index):
#         dist = -1
#         while index != -1:
#             index = self.getParent(index)
#             dist += 1
#         return dist

    def computeNProjectedChildrenForRoot(self):
        return 1 + 2**self.static.getNRealInterventions()

    def computeNProjectedChildrenForNonRoot(self):
        return 1 + 2*MAX_SIBLING_PAIRS

    def computeNProjectedChildren(self, index):
        if index == 0:
            return self.computeNProjectedChildrenForRoot()
        else:
            return self.computeNProjectedChildrenForNonRoot()

    def createChild(self, childIndex, child, parent, actions, childState=None):
        self.setNProjectedChildren(child, self.computeNProjectedChildren(child))
        childT = self.epochT
        if parent >= 0:
            parentT = self.getT(parent)
            childT = parentT + self.mcts.interval
            A0 = self.getActions(parent)
            schedule = makeUnitSchedule(self.static, A0, parentT, actions, self.mcts.interval)
            parentState = self.getState(parent)
            childState = self.epi.getNextState(parentState, schedule)
        self.setT(child, childT)
        self.setParent(child, parent)
        self.setState(child, childState)
        self.setActions(child, actions)
        self.setStatus(child, STATUS_EXPANDED)
        self.setStage(child, STAGE_ONE)
        if parent >= 0:
            self.addChildToParent(child, parent, childIndex)
            nExpandedChildren = self.addNExpandedChildren(parent)
            nExpandingChildren = self.getNExpandingChildren(parent)
            nProjectedChildren = self.getNProjectedChildren(parent)
            if self.getStage(parent) == STAGE_TWO:
                self.setStage(parent, STAGE_THREE)
            if nExpandedChildren == nProjectedChildren - 1:
                # Check repeated children since all children have been expanded
                actionHashDict = {}
                for childIndex_ in range(nProjectedChildren):
                    child_ = self.getChildToParent(parent, childIndex_)
                    actions_ = self.getActions(child_)
                    hashed = self.static.hashActions(actions_)
                    if hashed not in actionHashDict:
                        actionHashDict[hashed] = []
                    isRepeated = False
                    for otherChild in actionHashDict[hashed]:
                        otherActions = self.getActions(otherChild)
                        if self.static.isActionsEqual(actions_, otherActions):
                            isRepeated = True
                            break
                    if isRepeated:
                        self.setStatus(child_, STATUS_REPEATED)
                    else:
                        actionHashDict[hashed].append(child_)

    def addNExpandingChildren(self, index):
        return self.libint.fetch_add(byref(self.int, (index * self.intSize + 3) * INT_SIZE), byref(c_int(1)))

    def getNExpandingChildren(self, index):
        ret = c_int()
        self.libint.load(byref(self.int, (index * self.intSize + 3) * INT_SIZE), byref(ret))
        return ret.value

    def addNExpandedChildren(self, index):
        return self.libint.fetch_add(byref(self.int, (index * self.intSize + 4) * INT_SIZE), byref(c_int(1)))

    def addNEvaluatedChildren(self, index):
        return self.libint.fetch_add(byref(self.int, (index * self.intSize + 5) * INT_SIZE), byref(c_int(1)))

    def GET(self, index):
        c = c_ulonglong()
        self.libulong.load(byref(self.ulong, (index * self.ulongSize) * LONG_SIZE), byref(c))
        n = c_int()
        self.libint.load(byref(self.int, (index * self.intSize + 6) * INT_SIZE), byref(n))
        return c.value / (10 ** ULONG_PRECISION), n.value

    def SET(self, index, cost):
        c = c_ulonglong(int(cost * (10 ** ULONG_PRECISION)))
        self.libulong.fetch_add(byref(self.ulong, (index * self.ulongSize) * LONG_SIZE), byref(c))
        n = self.libint.fetch_add(byref(self.int, (index * self.intSize + 6) * INT_SIZE), byref(c_int(1)))
        if n == 0:
            parent = self.getParent(index)
            if parent >= 0:
                nEvaluatedChildren = self.addNEvaluatedChildren(parent)
                if nEvaluatedChildren == self.getNProjectedChildren(parent) - 1:
                    self.setStage(parent, STAGE_FOUR)

    def UCT(self, child, n, bestC, C):
        c_, n_ = self.GET(child)
        if n_ == 0 or abs(c_) < EPSILON:
            return inf
        exploit = bestC / (c_ / n_)
        explore = C * sqrt(2 * log(n) / n_)

        # Debug
        if C == 0:
            print(os.getpid(), child, n, c_, n_, bestC, exploit, explore)

        return exploit + explore

#     # Assess state of node with index to figure best action to explore in the tree
#     def smartSelect(self, index):
#         distance = self.getT(index) - self.epochT
#         degree = min(max(MIN_DEGREE, MAX_CHILDREN - distance), self.static.possibility)
#         choices = getSubset(degree, self.static.possibility)
#         As = []
#         for choice in choices:
#             As.append(self.static.inverseChoice(choice))
#         return As

#     # Assess state of node with index to figure a schedule to terminal time
#     def randomSelect(self, index):
#         tSpan = (self.getT(index), self.epi.T)
#         schedule = Schedule(tSpan, self.getActions(index))
#         for t in np.arange(tSpan[0], tSpan[1], self.mcts.interval, dtype=npFloat):
#             for itvId, itv in enumerate(self.static.interventions):
#                 if itv.name.startswith(ID_ALL) or (itv.name.startswith(ID_OP) and random.random() < RANDOM_APPLY_PROB):
#                     schedule.addAction(t, generateUniformRandomAction(self.static, itvId))
#         return schedule

#     def CREATE_CHILDREN(self, index):
#         if not self.isParent_exchange(index, True):
#             As = self.smartSelect(index)
#             j = self.fetchAddNNodes(len(As))
#             for A in As:
#                 self.createChild(j, index, A)
#                 j += 1
#             self.nNonexpandedChildren_store(index, len(As))
#             self.isExpandable_store_release(index, True)

    def modifyAction(self, action, option, changeDistance):
        if option == UNCHANGE:
            return action
        else:
            itv = self.static.interventions[action.id]
            cpi = np.zeros(len(itv.cps), dtype=npInt)
            if option == POSITIVE_CHANGE:
                for cpId, cp in enumerate(itv.cps):
                    cpi[cpId] = min(action.cpi[cpId] + changeDistance, cp.nBuckets - 1)
            else:
                for cpId, cp in enumerate(itv.cps):
                    cpi[cpId] = max(action.cpi[cpId] - changeDistance, 0)
            return Action(self.static, action.id, cpi)

    def coreSelect(self, parentActions, isBetter, changeDistance):
        weights = np.array([BIAS_PROBABILITY, UNBIAS_PROBABILITY, 1 - BIAS_PROBABILITY - UNBIAS_PROBABILITY])
        actions = Dict.empty(nbInt, actionType)
        for itvId, itv in enumerate(self.static.interventions):
            if itv.isCost:
                actions[itvId] = generateZeroAction(self.static, itvId)
            else:
                choice = getChoice(weights) - 1
                if isBetter:
                    choice *= -1
                parentAction = generateZeroAction(self.static, itvId) if itvId not in parentActions else parentActions[itvId]
                actions[itvId] = self.modifyAction(parentAction, choice, changeDistance)
        return actions

    def smartSelect(self, parent, childIndex):
        if childIndex == 0:
            return self.getActions(parent)
        else:
            if parent == 0:
                actions = Dict.empty(nbInt, actionType)
                # Select children for root
                binary = childIndex - 1
                for itvId, itv in enumerate(self.static.interventions):
                    if itv.isCost:
                        actions[itvId] = generateZeroAction(self.static, itvId)
                    else:
                        if binary % 2 == 0:
                            actions[itvId] = generateZeroAction(self.static, itvId)
                        else:
                            actions[itvId] = generateOneAction(self.static, itvId)
                        binary //= 2
                return actions
            else:
                # Select children for non-root
                return self.coreSelect(self.getActions(parent), childIndex%2==0, SMART_DISTANCE)

    def randomSelect(self, index):
        tSpan = (self.getT(index), self.epi.T)
        actions = self.getActions(index)
        schedule = Schedule(tSpan, actions)
        for t in np.arange(tSpan[0], tSpan[1], self.mcts.interval, dtype=npFloat):
            if random.random() < RANDOM_APPLY_PROBABILITY:
                isBetter = random.random() < 0.5
                actions = self.coreSelect(actions, isBetter, RANDOM_DISTANCE)
                for actId, action in actions.items():
                    schedule.addAction(t, action)
        return schedule

    def addChild(self, index):
        nProjectedChildren = self.getNProjectedChildren(index)
        if self.getStage(index) ==  STAGE_ONE:
            nExpandingChildren = self.addNExpandingChildren(index)
            if nExpandingChildren < nProjectedChildren:
                if nExpandingChildren == nProjectedChildren - 1:
                    self.setStage(index, STAGE_TWO)
                # Assign this worker to create a children from parent index
                actions = self.smartSelect(index, nExpandingChildren)
                child = self.addNNodes()
                self.createChild(nExpandingChildren, child, index, actions)
                return child
            return self.addChild(index)
        elif self.getStage(index) == STAGE_TWO:
            if index == 0:
                while self.getStage(index) == STAGE_TWO:
                    pass
                return self.addChild(index)
            else:
                return index
        else:
            choices = []
            for childIndex in range(nProjectedChildren):
                child = self.getChildToParent(index, childIndex)
                if self.getStatus(child) == STATUS_EXPANDED:
                    choices.append(child)
            return random.choice(choices)

    def isRepeated(self, index):
        return self.getStatus(index) == STATUS_REPEATED

    def isFullyExpanded(self, index):
        return self.getStage(index) == STAGE_FOUR

    def isTreeTerminal(self, index):
        return self.getT(index) - self.epochT > TREE_TERMINAL
