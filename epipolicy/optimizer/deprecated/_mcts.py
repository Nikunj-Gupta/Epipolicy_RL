from ..utility.singleton import *
from .node_manager import NodeManager
from multiprocessing import Process, Queue
from ctypes import byref
import numpy as np
import os, time, random, pickle

def f(self, pIndex, conn, session):
    random.seed()
    st = time.time()
    self.reinitialize(conn, session)
    fn = time.time()
    print(pIndex, "finished reinitialization in", fn-st, "s", flush=True)
    while True:
        epochT = self.start[pIndex].get()
        self.manager.epochT = epochT
        if epochT < 0:
            break
        st = time.time()
        self.UCT_SEARCH()
        self.done.put(pIndex)
        fn = time.time()
        print(pIndex, "finished in", fn-st, "s", flush=True)

class MCTS:
    def __init__(self, epi, iBudget=I_BUDGET, tBudget=T_BUDGET, interval=MCTS_INTERVAL):
        self.epi = epi
        self.static = epi.static
        self.iBudget = iBudget
        self.tBudget = tBudget
        self.interval = interval
        self.manager = NodeManager(self)
        #self.nProcesses = os.cpu_count()
        self.nProcesses = 2
        self.done = Queue()
        self.start = [Queue() for i in range(self.nProcesses)]
        processes = []
        for i in range(self.nProcesses):
            p = Process(target=f, args=(self, i, self.epi.conn, self.epi.session))
            processes.append(p)
        for p in processes:
            p.start()

    def reinitialize(self, conn, session):
        from ..core.epidemic import Epidemic
        epi = Epidemic(conn, session, False)
        self.epi = epi
        self.static = epi.static
        self.manager.epi = epi
        self.manager.static = epi.static
        self.manager.initLibrary()

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['static']
        del attributes['epi']
        return attributes

    def getBestChild(self, index, C=MCTS_C):
        bestCost = self.manager.getBestCost()
        nChildren = self.manager.getNProjectedChildren(index)
        _, n = self.manager.GET(index)
        bestChild = -1
        maxUct = 0
        for i in range(nChildren):
            child = self.manager.getChildToParent(index, i)
            if self.manager.isRepeated(child):
                continue
            uct = self.manager.UCT(child, n, bestCost, C)
            if maxUct < uct:
                maxUct = uct
                bestChild = child
        return bestChild

    def makeDecision(self, epochT, prevActions, prevState):
        self.manager.clear()
        self.manager.createRoot(epochT, prevActions, prevState)
        st = time.time()
        for i in range(self.nProcesses):
            self.start[i].put(epochT)
        for i in range(self.nProcesses):
            pid = self.done.get()
        fn = time.time()
        print("Finished", self.iBudget, "iterations in", fn-st, "s -- avg:", self.iBudget / (fn-st), "iterations per sec", flush=True)
        bestChild = self.getBestChild(0, C=0)
        return self.manager.getActions(bestChild)

    def close(self):
        for i in range(self.nProcesses):
            self.start[i].put(-1)

    def UCT_SEARCH(self):
        while not self.manager.isPlayoutFinished():
            playout = self.manager.addNPlayouts()
            selectIndex = self.SELECT(0)
            expandIndex = self.EXPAND(selectIndex)
            cost = self.PLAYOUT(expandIndex)
            self.BACKUP(expandIndex, cost)
            print(os.getpid(), "finished", playout, "-th iteration", flush=True)

    def SELECT(self, index):
        #print(os.getpid(), "SELECT", index)
        while self.manager.isFullyExpanded(index):
            index = self.getBestChild(index)
        return index

    def EXPAND(self, index):
        #print(os.getpid(), "EXPAND", index, flush=True)
        if self.manager.isTreeTerminal(index):
            return index
        return self.manager.addChild(index)

    def PLAYOUT(self, index):
        #print(os.getpid(), "PLAYOUT", index, flush=True)
        schedule = self.manager.randomSelect(index)
        curState = self.manager.getState(index)
        #terminalState = self.epi.getNextState(curState, schedule)
        terminalState = self.epi.getNextStateIteratively(curState, schedule)
        self.manager.updateBestCost(terminalState.c)
        return terminalState.c

    def BACKUP(self, index, cost):
        #print(os.getpid(), "BACKUP", index, cost, flush=True)
        while index != -1:
            self.manager.SET(index, cost)
            index = self.manager.getParent(index)
