import numpy as np
import numba as nb
from numba.types import DictType, string
from numba.typed import Dict, List
from numba.experimental import jitclass
from numba import njit

from ..utility.utils import get_normalized_string
from ..utility.singleton import ALL, NOT, OR, AND
from ..obj.static import StaticType
from .regsult import Regsult

regex_parser_spec = [
    ("static", StaticType)
]

@njit
def get_att_val(s, atts):
    pos = s.find(':')
    atts[s[1:pos-1]] = s[pos+2:-1]

@njit
def update_groupers(lvs, c):
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
def parse_mask_regex_recurse(static, mask_regex, mask_property, mat):
    mask_map = static.property_name_to_index[mask_property]
    res = np.zeros(len(mat), dtype=nb.boolean)
    st = 0
    end = st
    lvs = np.zeros(4, dtype=np.int32)
    previous_op = OR
    while end < len(mask_regex):
        update_groupers(lvs, mask_regex[end])
        if (mask_regex[end] == OR or mask_regex[end] == AND) and np.sum(lvs) == 0:
            if previous_op == OR:
                res = np.logical_or(res, parse_mask_regex_recurse(static, mask_regex[st:end], mask_property, mat))
            elif previous_op == AND:
                res = np.logical_and(res, parse_mask_regex_recurse(static, mask_regex[st:end], mask_property, mat))
            previous_op = mask_regex[end]
            st = end+1
        end += 1
    if st > 0:
        if previous_op == OR:
            res = np.logical_or(res, parse_mask_regex_recurse(static, mask_regex[st:end], mask_property, mat))
        elif previous_op == AND:
            res = np.logical_and(res, parse_mask_regex_recurse(static, mask_regex[st:end], mask_property, mat))
    else:
        if mask_regex[st] == ALL:
            res = np.sum(mat, axis=1) > 0
        elif mask_regex[st] == NOT:
            return np.logical_not(parse_mask_regex_recurse(static, mask_regex[1:], mask_property, mat))
        elif mask_regex[st] == '(':
            return parse_mask_regex_recurse(static, mask_regex[1:-1], mask_property, mat)
        else:
            res = mat[:, mask_map[mask_regex]] > 0
    return res

@njit
def parse_subdict_regex(static, subdict_regex, property):
    map = static.property_name_to_index[property]
    st = 0
    end = st
    lvs = np.zeros(4, dtype=np.int32)
    atts = Dict.empty(string, string)
    while end < len(subdict_regex):
        update_groupers(lvs, subdict_regex[end])
        if subdict_regex[end] == ',' and np.sum(lvs) == 0:
            get_att_val(subdict_regex[st:end], atts)
            st = end+1
        end += 1
    get_att_val(subdict_regex[st:end], atts)
    for att, val in atts.items():
        if property == "compartment":
            if att == "tag":
                return parse_mask_regex_recurse(static, val, "compartment_tag", static.compartment_tags)
        elif property == "parameter":
            if att == "tag":
                return parse_mask_regex_recurse(static, val, "parameter_tag", static.parameter_tags)
    return np.zeros(len(map), dtype=nb.boolean)

@njit
def parse_regex_recurse(static, regex, property):
    map = static.property_name_to_index[property]
    res = np.zeros(len(map), dtype=nb.boolean)
    st = 0
    end = st
    lvs = np.zeros(4, dtype=np.int32)
    previous_op = OR
    while end < len(regex):
        update_groupers(lvs, regex[end])
        if (regex[end] == OR or regex[end] == AND) and np.sum(lvs) == 0:
            if previous_op == OR:
                res = np.logical_or(res, parse_regex_recurse(static, regex[st:end], property))
            elif previous_op == AND:
                res = np.logical_and(res, parse_regex_recurse(static, regex[st:end], property))
            previous_op = regex[end]
            st = end+1
        end += 1
    if st > 0:
        if previous_op == OR:
            res = np.logical_or(res, parse_regex_recurse(static, regex[st:end], property))
        elif previous_op == AND:
            res = np.logical_and(res, parse_regex_recurse(static, regex[st:end], property))
    else:
        if regex[st] == ALL:
            return np.ones(len(map), dtype=nb.boolean)
        elif regex[st] == NOT:
            return np.logical_not(parse_regex_recurse(static, regex[1:], property))
        elif regex[st] == '(':
            return parse_regex_recurse(static, regex[1:-1], property)
        elif regex[st] == '{':
            return parse_subdict_regex(static, regex[1:-1], property)
        else:
            res[map[regex]] = True
    return res

@jitclass(regex_parser_spec)
class RegexParser:

    def __init__(self, static):
        self.static = static

    def get_locale_hierarchy(self, name):
        if len(name) == 0:
            res = Dict.empty(nb.int32, nb.boolean)
            for i in range(self.static.locale_count):
                res[i] = True
            return res
        else:
            return self.static.locale_hierarchy[name]

    def parse_locale_regex(self, locale_regex):
        is_found = locale_regex.find(ALL)
        name = locale_regex
        if is_found >= 0:
            name = locale_regex[:is_found]
            if len(name) > 0:
                if name[-1] == ".":
                    name = name[:-1]
            else:
                return self.get_locale_hierarchy(name)
        name = get_normalized_string(name)
        isNot = False
        if name[0] == NOT:
            isNot = True
            if len(name) > 1:
                name = name[1:]
            else:
                name = ""
        res = self.get_locale_hierarchy(name)
        if isNot:
            notRes = Dict.empty(nb.int32, nb.boolean)
            for i in range(self.static.locale_count):
                if i not in res:
                    notRes[i] = True
            return notRes
        return res

    def parse_regex(self, regex, property):
        res = parse_regex_recurse(self.static, regex.replace(" ", ""), property)
        dict_res = Dict.empty(nb.int32, nb.int32)
        for index, is_matched in enumerate(res):
            if is_matched:
                dict_res[index] = len(dict_res)
        list_res = np.zeros(len(dict_res), dtype=nb.int32)
        for index, i in dict_res.items():
            list_res[i] = index
        return list_res

    def parse_group_regex(self, group_regex):
        return self.parse_regex(group_regex, "group")

    def parse_facility_regex(self, facility_regex):
        return self.parse_regex(facility_regex, "facility")

    def parse_compartment_regex(self, compartment_regex):
        return self.parse_regex(compartment_regex, "compartment")

    def parse_parameter_regex(self, parameter_regex):
        return self.parse_regex(parameter_regex, "parameter")

    def parse_intervention_regex(self, intervention_regex):
        return self.parse_regex(intervention_regex, "intervention")

    def parse_dict_regex(self, keywords):
        res = Regsult()
        for att, val in keywords.items():
            if att == "locale" or att == "locale-from":
                res.locale_result["locale-from"] = self.parse_locale_regex(val)
            elif att == "locale-to":
                res.locale_result["locale-to"] = self.parse_locale_regex(val)
            elif att == "group" or att == "group-from":
                res.index_result["group-from"] = self.parse_group_regex(val)
            elif att == "group-to":
                res.index_result["group-to"] = self.parse_group_regex(val)
            elif att == "parameter":
                res.index_result["parameter"] = self.parse_parameter_regex(val)
            elif att == "compartment" or att == "compartment-from":
                res.index_result["compartment-from"] = self.parse_compartment_regex(val)
            elif att == "compartment-to":
                res.index_result["compartment-to"] = self.parse_compartment_regex(val)
            elif att == "facility":
                res.index_result["facility"] = self.parse_facility_regex(val)
            elif att == "intervention":
                res.index_result["intervention"] = self.parse_intervention_regex(val)
            else:
                res.string_result[att] = val
        return res

RegexParserType = RegexParser.class_type.instance_type
