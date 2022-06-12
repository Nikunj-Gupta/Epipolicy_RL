from ..utility.singleton import *
from ..utility.utils import normalize
from ..obj.np_wrapper import Np1i, np1iType
from ..obj.static import Static, staticType
from ..obj.regsult import Regsult, setType
from numba.types import DictType, string
from numba.typed import Dict, List
from numba.experimental import jitclass
from numba import generated_jit, njit
import numpy as np

regexParserSpec = [
    ('static', staticType)
]

@njit
def _getAttVal(s, atts):
    pos = s.find(':')
    atts[s[1:pos-1]] = s[pos+2:-1]

@njit
def _updateGroupers(lvs, c):
    if c == '(':
        lvs[0] += 1
    elif c == ')':
        lvs[0] -= 1
    elif c == '{':
        lvs[1] += 1
    elif c == '}':
        lvs[1] -= 1
    elif c == '"':
        lvs[2] = 1 - lvs[2]
    elif c == "'":
        lvs[3] = 1 - lvs[3]

@njit
def _parseMaskRegexRecurse(static, maskRegex, maskProp, mat):
    maskMap = static.toIndex[maskProp]
    res = np.zeros(len(mat), dtype=nbBool)
    st = 0
    end = st
    lvs = np.zeros(4, dtype=nbInt)
    preOp = OR
    while end < len(maskRegex):
        _updateGroupers(lvs, maskRegex[end])
        if (maskRegex[end] == OR or maskRegex[end] == AND) and np.sum(lvs) == 0:
            if preOp == OR:
                res = np.logical_or(res, _parseMaskRegexRecurse(static, maskRegex[st:end], maskProp, mat))
            elif preOp == AND:
                res = np.logical_and(res, _parseMaskRegexRecurse(static, maskRegex[st:end], maskProp, mat))
            preOp = maskRegex[end]
            st = end+1
        end += 1
    if st > 0:
        if preOp == OR:
            res = np.logical_or(res, _parseMaskRegexRecurse(static, maskRegex[st:end], maskProp, mat))
        elif preOp == AND:
            res = np.logical_and(res, _parseMaskRegexRecurse(static, maskRegex[st:end], maskProp, mat))
    else:
        if maskRegex[st] == ALL:
            res = np.sum(mat, axis=1) > 0
        elif maskRegex[st] == NOT:
            return np.logical_not(_parseMaskRegexRecurse(static, maskRegex[1:], maskProp, mat))
        elif maskRegex[st] == '(':
            return _parseMaskRegexRecurse(static, maskRegex[1:-1], maskProp, mat)
        else:
            res = mat[:, maskMap[maskRegex]] > 0
    return res

@njit
def _parseDRegex(static, dRegex, prop):
    map = static.toIndex[prop]
    st = 0
    end = st
    lvs = np.zeros(4, dtype=nbInt)
    atts = Dict.empty(string, string)
    while end < len(dRegex):
        _updateGroupers(lvs, dRegex[end])
        if dRegex[end] == ',' and np.sum(lvs) == 0:
            _getAttVal(dRegex[st:end], atts)
            st = end+1
        end += 1
    _getAttVal(dRegex[st:end], atts)
    for att, val in atts.items():
        if prop == 'compartment':
            if att == 'tag':
                return _parseMaskRegexRecurse(static, val, "compartmentTag", static.compartmentTags)
        elif prop == 'parameter':
            if att == 'tag':
                return _parseMaskRegexRecurse(static, val, "parameterTag", static.parameterTags)
    return np.zeros(len(map), dtype=nbBool)

@njit
def _parseXRegexRecurse(static, xRegex, prop):
    map = static.toIndex[prop]
    res = np.zeros(len(map), dtype=nbBool)
    st = 0
    end = st
    lvs = np.zeros(4, dtype=nbInt)
    preOp = OR
    while end < len(xRegex):
        _updateGroupers(lvs, xRegex[end])
        if (xRegex[end] == OR or xRegex[end] == AND) and np.sum(lvs) == 0:
            if preOp == OR:
                res = np.logical_or(res, _parseXRegexRecurse(static, xRegex[st:end], prop))
            elif preOp == AND:
                res = np.logical_and(res, _parseXRegexRecurse(static, xRegex[st:end], prop))
            preOp = xRegex[end]
            st = end+1
        end += 1
    if st > 0:
        if preOp == OR:
            res = np.logical_or(res, _parseXRegexRecurse(static, xRegex[st:end], prop))
        elif preOp == AND:
            res = np.logical_and(res, _parseXRegexRecurse(static, xRegex[st:end], prop))
    else:
        if xRegex[st] == ALL:
            return np.ones(len(map), dtype=nbBool)
        elif xRegex[st] == NOT:
            return np.logical_not(_parseXRegexRecurse(static, xRegex[1:], prop))
        elif xRegex[st] == '(':
            return _parseXRegexRecurse(static, xRegex[1:-1], prop)
        elif xRegex[st] == '{':
            return _parseDRegex(static, xRegex[1:-1], prop)
        else:
            res[map[xRegex]] = True
    return res

@jitclass(regexParserSpec)
class RegexParser:
    def __init__(self, static):
        self.static = static

    def parseLRegex(self, lRegex):
        isFound = lRegex.find(ALL)
        name = lRegex
        if isFound >= 0:
            name = lRegex[:isFound]
            if len(name) > 0:
                name = name[:-1]
            else:
                res = Dict.empty(nbInt, nbBool)
                for i in range(self.static.nLocales):
                    res[i] = True
                return res
        name = normalize(name)
        isNot = False
        if name[0] == NOT:
            isNot = True
            name = name[1:]
        res = self.static.localeHierarchy[name]
        if isNot:
            notRes = Dict.empty(nbInt, nbBool)
            for i in range(self.static.nLocales):
                if i not in res:
                    notRes[i] = True
            return notRes
        return res

    def _parseXRegex(self, xRegex, prop):
        res = _parseXRegexRecurse(self.static, xRegex.replace(" ", ""), prop)
        #print(res, flush=True)
        dictRes = Dict.empty(nbInt, nbInt)
        for i, v in enumerate(res):
            if v:
                dictRes[i] = len(dictRes)
        liRes = np.zeros(len(dictRes), dtype=nbInt)
        for v, i in dictRes.items():
            liRes[i] = v
        return Np1i(liRes)

    def parseGRegex(self, gRegex):
        return self._parseXRegex(gRegex, "group")

    def parseFRegex(self, fRegex):
        return self._parseXRegex(fRegex, "facility")

    def parseCRegex(self, cRegex):
        return self._parseXRegex(cRegex, "compartment")

    def parsePRegex(self, pRegex):
        return self._parseXRegex(pRegex, "parameter")

    def parseIRegex(self, iRegex):
        return self._parseXRegex(iRegex, "intervention")

    def parseDictRegex(self, keywords):
        res = Regsult()
        for att, val in keywords.items():
            #print(att, val, flush=True)
            if att == "locale" or att == "locale-from":
                res.lResult["locale-from"] = self.parseLRegex(val)
            elif att == "locale-to":
                res.lResult["locale-to"] = self.parseLRegex(val)
            elif att == "group" or att == "group-from":
                res.iResult["group-from"] = self.parseGRegex(val)
            elif att == "group-to":
                res.iResult["group-to"] = self.parseGRegex(val)
            elif att == "parameter":
                res.iResult[att] = self.parsePRegex(val)
            elif att == "compartment" or att == "compartment-from":
                res.iResult["compartment-from"] = self.parseCRegex(val)
            elif att == "compartment-to":
                res.iResult["compartment-to"] = self.parseCRegex(val)
            elif att == "facility":
                res.iResult["facility"] = self.parseFRegex(val)
            elif att == "intervention":
                res.iResult["intervention"] = self.parseIRegex(val)
            else:
                res.sResult[str(att)] = val
        return res

    # def parseRegex(self, regex):
    #     res = Regsult()
    #     keywords = regex.split(' ')
    #     for key in keywords:
    #         att, val = key.split(":")
    #         if att == "locale":
    #             if "from" not in res.lResult:
    #                 res.lResult["from"] = self.parseLRegex(val)
    #             else:
    #                 res.lResult["to"] = self.parseLRegex(val)
    #         elif att == "group":
    #             if "fromGroup" not in res.iResult:
    #                 res.iResult["fromGroup"] = self.parseGRegex(val)
    #             else:
    #                 res.iResult["toGroup"] = self.parseGRegex(val)
    #         elif att == "parameter":
    #             res.iResult[att] = self.parsePRegex(val)
    #         elif att == "compartment":
    #             if "fromCompartment" not in res.iResult:
    #                 res.iResult["fromCompartment"] = self.parseCRegex(val)
    #             else:
    #                 res.iResult["toCompartment"] = self.parseCRegex(val)
    #         elif att == "facility":
    #             res.iResult[att] = self.parseFRegex(val)
    #         else:
    #             res.sResult[att] = val
    #     return res



regexParserType = RegexParser.class_type.instance_type
