import numpy, pandas
import numba as nb
from numba.types import string
from numba.typed import Dict, List
from numba import njit

from ..obj.delta_parameter import get_initial_delta_parameter
from ..utility.singleton import DIE_TAG
from ..utility.utils import get_influence

DEFAULT_MODE = "border"
DEFAULT_VALUE_MODE = "current"

def find_matching_bracket(s, st):
    cnt = 1
    for i in range(st, len(s)):
        if s[i] == '(':
            cnt += 1
        elif s[i] == ')':
            cnt -= 1
            if cnt == 0:
                return i
    return -1

def sub_inject(func, search, insert, add_params):
    start = 0
    while True:
        pos = func.find(search, start)
        if pos < 0:
            break
        next_pos = find_matching_bracket(func, pos + len(search))
        dot_pos = func.find('.', pos)
        func = func[:pos] + insert + func[dot_pos:next_pos] + add_params + func[next_pos:]
        start = pos + 1 + len(insert)
    return func

def inject(func_index, func_str, func_name):
    sim_func_names = ['select', 'apply', 'move', 'add']
    sim_name = "self"
    add_params = ", state, delta_parameter, itv_index"
    f1 = "def {}(cp, locales)".format(func_name)
    f2 = "def {}(cp)".format(func_name)
    pos = max(func_str.find(f1), func_str.find(f2))
    if pos >= 0:
        next_pos = func_str.find(")", pos + 1)
        func_str = func_str[:pos+4+len(func_name)] + str(func_index) + "({}, ".format(sim_name) + func_str[pos+5+len(func_name):next_pos] + add_params + func_str[next_pos:]
        for sim_func_name in sim_func_names:
            func_str = sub_inject(func_str, "sim.{}(".format(sim_func_name), sim_name, add_params)
        content = func_str[func_str.find(":")+1:].strip()
        if len(content) == 0:
            # Empty function, we just need to add return function to make it valid
            func_str = func_str.strip() + "\n\treturn"
        return "global {}{}\n{}".format(func_name, func_index, func_str)

@njit
def stricten_locale(static, func, res):
    if "locale-from" not in res.locale_result:
        locale_from = Dict.empty(nb.int32, nb.boolean)
        for l in range(static.locale_count):
            locale_from[l] = True
        res.locale_result["locale-from"] = locale_from

@njit
def stricten_group(static, func, res):
    if "group-from" not in res.index_result:
        res.index_result["group-from"] = numpy.arange(static.group_count, dtype=numpy.int32)

@njit
def stricten_facility(static, func, res):
    if "facility" not in res.index_result:
        res.index_result["facility"] = numpy.arange(static.facility_count, dtype=numpy.int32)

@njit
def stricten_compartment(static, func, res):
    if "compartment-from" not in res.index_result:
        if func == "move":
            res.index_result["compartment-from"] = static.alive_comp_list

@njit
def stricten_intervention(static, itv_index, func, res):
    if "intervention" not in res.index_result:
        res.index_result["intervention"] = numpy.array([itv_index], dtype=numpy.int32)

@njit
def stricten_result(static, itv_index, func, res):
    if func == "select" or func == "apply":
        if "parameter" in res.index_result:
            stricten_locale(static, func, res)
            stricten_facility(static, func, res)
            stricten_group(static, func, res)
        elif "locale-from" in res.locale_result and "locale-to" in res.locale_result:
            stricten_group(static, func, res)
        elif "group-from" in res.index_result and "group-to" in res.index_result:
            stricten_locale(static, func, res)
            stricten_facility(static, func, res)
        elif "compartment-from" in res.index_result:
            stricten_locale(static, func, res)
            stricten_group(static, func, res)
        elif "facility" in res.index_result:
            stricten_locale(static, func, res)
            stricten_group(static, func, res)
    elif func == "add":
        stricten_locale(static, func, res)
        stricten_intervention(static, itv_index, func, res)
    elif func == "move":
            stricten_compartment(static, func, res)
            stricten_locale(static, func, res)
            stricten_group(static, func, res)
    return res

@njit
def get_regsult(static, parser, itv_index, func, string_dict_regex):
    res = parser.parse_dict_regex(string_dict_regex)
    return stricten_result(static, itv_index, func, res)

@njit
def nb_apply(static, res, multiplier, state, delta_parameter, itv_index):
    influence_score = get_influence(multiplier)
    influence = numpy.zeros((static.locale_count, static.group_count), dtype=numpy.float32)
    default_parameter = static.default_state.unobs
    if "locale-from" in res.locale_result and "locale-to" in res.locale_result:
        mode_index = static.get_property_index("mode", DEFAULT_MODE)
        if "mode" in res.string_result:
            mode_index = static.get_property_index("mode", res.string_result["mode"])
        group_coo = static.mode_group_coo[mode_index]
        for g in res.index_result["group-from"]:
            delta_parameter.mode_group_coo[mode_index][g].pairwise_multiply(group_coo[g], multiplier, res.locale_result["locale-from"], res.locale_result["locale-to"])

        for g in res.index_result["group-from"]:
            for l1 in res.locale_result["locale-from"]:
                if l1 in group_coo[g].mat:
                    for l2, val in group_coo[g].mat[l1].items():
                        if l2 in res.locale_result["locale-to"]:
                            influence[l1, g] += static.locale_group_pop_proportion[l1, g]*val*influence_score

    elif "parameter" in res.index_result:
        for p in res.index_result["parameter"]:
            for l in res.locale_result["locale-from"]:
                for f in res.index_result["facility"]:
                    for g in res.index_result["group-from"]:
                        delta_parameter.model_parameters[p, l, f, g] *= multiplier

        for l in res.locale_result["locale-from"]:
            for f in res.index_result["facility"]:
                for g in res.index_result["group-from"]:
                    influence[l, g] += default_parameter.facility_timespent[l, f, g]*influence_score

    elif "group-from" in res.index_result and "group-to" in res.index_result:
        for l in res.locale_result["locale-from"]:
            for f in res.index_result["facility"]:
                for g1 in res.index_result["group-from"]:
                    for g2 in res.index_result["group-to"]:
                        delta_parameter.facility_interaction[l, f, g1, g2] *= multiplier
                        influence[l, g1] += default_parameter.facility_timespent[l, f, g1]*default_parameter.facility_interaction[l, f, g1, g2]*influence_score
                        influence[l, g2] += default_parameter.facility_timespent[l, f, g2]*default_parameter.facility_interaction[l, f, g2, g1]*influence_score

    elif "facility" in res.index_result:
        for l in res.locale_result["locale-from"]:
            for f in res.index_result["facility"]:
                for g in res.index_result["group-from"]:
                    delta_parameter.facility_timespent[l, f, g] *= multiplier
                    influence[l, g] += default_parameter.facility_timespent[l, f, g]*influence_score

    delta_parameter.influence[itv_index] = 1-(1-delta_parameter.influence[itv_index])*(1-numpy.minimum(influence, 1))

@njit
def nb_move(static, res1, res2, value, state, delta_parameter, itv_index):
    influence = numpy.zeros((static.locale_count, static.group_count), dtype=numpy.float32)
    depart_sum = 0
    current_comp = state.obs.current_comp
    same_groups = Dict.empty(nb.int32, nb.boolean)
    for g in res1.index_result["group-from"]:
        same_groups[g] = False
    for g in res2.index_result["group-from"]:
        if g in same_groups:
            same_groups[g] = True
    for g in same_groups:
        if same_groups[g]:
            for c1 in res1.index_result["compartment-from"]:
                for l1 in res1.locale_result["locale-from"]:
                    depart_sum += current_comp[c1, l1, g]
    if depart_sum > 0:
        c2_set = Dict.empty(nb.int32, nb.boolean)
        for c2 in res2.index_result["compartment-from"]:
            c2_set[c2] = True
        for g in same_groups:
            if same_groups[g]:
                for c1 in res1.index_result["compartment-from"]:
                    for l1 in res1.locale_result["locale-from"]:
                        depart_amount = current_comp[c1, l1, g]/depart_sum*value
                        if c1 in c2_set:
                            if l1 not in res2.locale_result["locale-from"]:
                                arrive_sum = 0
                                for l2 in res2.locale_result["locale-from"]:
                                    arrive_sum += current_comp[c1, l2, g]
                                if arrive_sum > 0:
                                    for l2 in res2.locale_result["locale-from"]:
                                        move_amount = current_comp[c1, l2, g]/arrive_sum*depart_amount
                                        delta_parameter.add_move(static, influence, g, c1, c1, l1, l2, move_amount)
                                else:
                                    for l2 in res2.locale_result["locale-from"]:
                                        move_amount = 1.0/len(res2.locale_result["locale-from"])*depart_amount
                                        delta_parameter.add_move(static, influence, g, c1, c1, l1, l2, move_amount)
                        else:
                            if l1 in res2.locale_result["locale-from"]:
                                arrive_sum = 0
                                for c2 in res2.index_result["compartment-from"]:
                                    arrive_sum += current_comp[c2, l1, g]
                                if arrive_sum > 0:
                                    for c2 in res2.index_result["compartment-from"]:
                                        move_amount = current_comp[c2, l1, g]/arrive_sum*depart_amount
                                        delta_parameter.add_move(static, influence, g, c1, c2, l1, l1, move_amount)
                                else:
                                    for c2 in res2.index_result["compartment-from"]:
                                        move_amount = 1.0/len(res2.index_result["compartment-from"])*depart_amount
                                        delta_parameter.add_move(static, influence, g, c1, c2, l1, l1, move_amount)
                            else:
                                arrive_sum = 0
                                for c2 in res2.index_result["compartment-from"]:
                                    for l2 in res2.locale_result["locale-from"]:
                                        arrive_sum += current_comp[c2, l2, g]
                                if arrive_sum > 0:
                                    for c2 in res2.index_result["compartment-from"]:
                                        for l2 in res2.locale_result["locale-from"]:
                                            move_amount = current_comp[c2, l2, g]/arrive_sum*depart_amount
                                            delta_parameter.add_move(static, influence, g, c1, c2, l1, l2, move_amount)
                                else:
                                    partition_count = len(res2.index_result["compartment-from"])*len(res2.locale_result["locale-from"])
                                    for c2 in res2.index_result["compartment-from"]:
                                        for l2 in res2.locale_result["locale-from"]:
                                            move_amount = 1.0/partition_count*depart_amount
                                            delta_parameter.add_move(static, influence, g, c1, c2, l1, l2, move_amount)

    delta_parameter.influence[itv_index] = 1-(1-delta_parameter.influence[itv_index])*(1-numpy.minimum(influence, 1))

@njit
def nb_add(static, res, value, state, delta_parameter, itv_index):
    if "intervention" in res.index_result:
        partition_count = len(res.index_result["intervention"])*len(res.locale_result["locale-from"])
        if partition_count > 0:
            for i in res.index_result["intervention"]:
                for l in res.locale_result["locale-from"]:
                    delta_parameter.cost[i, l] += value/partition_count

class Executor:
    def __init__(self, epi, leaf_locales):
        self.epi = epi
        self.leaf_locales = leaf_locales
        self.set_intervention_functions()

    def get_locale_name(self, locale_index):
        return self.leaf_locales[locale_index]["name"]

    def register_function(self, func_index, func_str, func_name):
        func = inject(func_index, func_str, func_name)
        exec(func)
        exec("self.{}{} = {}{}.__get__(self)".format(func_name, func_index, func_name, func_index))

    def set_intervention_functions(self):
        interventions = self.epi.session["interventions"]
        costs = self.epi.session["costs"]

        for intervention in interventions:
            func_index = self.epi.static.get_property_index("intervention", intervention["name"])
            self.register_function(func_index, intervention["effect"], "effect")
            self.register_function(func_index, intervention["cost"], "cost")

        for cost in costs:
            func_index = self.epi.static.get_property_index("intervention", cost["name"])
            self.register_function(func_index, cost["func"], "cost")

    def execute(self, state, action):
        delta_parameter = get_initial_delta_parameter(self.epi.static)
        for act in action:
            cp_dict = {}
            cp_list = self.epi.static.interventions[act.index].cp_list
            for cp_index, cpv in enumerate(act.cpv_list):
                cp_dict[cp_list[cp_index].name] = cpv
            if not self.epi.static.interventions[act.index].is_cost:
                exec("self.effect{}(cp_dict, act.locale_regex, state, delta_parameter, act.index)".format(act.index))
                exec("self.cost{}(cp_dict, act.locale_regex, state, delta_parameter, act.index)".format(act.index))
            else:
                exec("self.cost{}(cp_dict, state, delta_parameter, act.index)".format(act.index))
        return delta_parameter

    def to_string_dict_regex(self, regex):
        string_dict_regex = Dict.empty(string, string)
        for k, v in regex.items():
            string_dict_regex[str(k)] = str(v)
        return string_dict_regex

    def select(self, regex, state, delta_parameter, itv_index):
        res = get_regsult(self.epi.static, self.epi.parser, itv_index, "select", self.to_string_dict_regex(regex))
        rows = []
        value_mode = DEFAULT_VALUE_MODE
        current_comp = state.obs.current_comp
        current_parameter = state.unobs
        default_comp = self.epi.static.default_state.obs.current_comp
        default_parameter = self.epi.static.default_state.unobs
        if "value-mode" in res.string_result:
            value_mode = res.string_result["value-mode"]
        if "locale-from" in res.locale_result and "locale-to" in res.locale_result:
            mode_index = self.epi.static.get_property_index("mode", DEFAULT_MODE)
            if "mode" in res.string_result:
                mode_index = self.epi.static.get_property_index("mode", res.string_result["mode"])
            group_coo = self.epi.static.mode_group_coo[mode_index]
            if value_mode == "current":
                indices = self.epi.static.mode_csr[mode_index].indices
                indptr = self.epi.static.mode_csr[mode_index].indptr
                data = state.unobs.mode_reduced_csr[mode_index]
                for g in res.index_result["group-from"]:
                    for l1 in res.locale_result["locale-from"]:
                        if l1 in group_coo[g].mat:
                            for col_index in range(indptr[l1], indptr[l1+1]):
                                l2 = indices[col_index]
                                if l2 in res.locale_result["locale-to"]:
                                    rows.append([self.epi.static.get_property_name("group", g), self.get_locale_name(l1), self.get_locale_name(l2), data[g, col_index]])
            elif value_mode == "default":
                for g in res.index_result["group-from"]:
                    for l1 in res.locale_result["locale-from"]:
                        if l1 in group_coo[g].mat:
                            for l2 in group_coo[g].mat[l1]:
                                if l2 in res.locale_result["locale-to"]:
                                    rows.append([self.epi.static.get_property_name("group", g), self.get_locale_name(l1), self.get_locale_name(l2), group_coo[g].mat[l1][l2]])
            ret = pandas.DataFrame(rows, columns=['Group', 'Locale From', 'Locale To', 'Value'])
        elif "parameter" in res.index_result:
            if value_mode == "current":
                model_parameters = current_parameter.model_parameters
            elif value_mode == "default":
                model_parameters = default_parameter.model_parameters
            for p in res.index_result["parameter"]:
                for l in res.locale_result["locale-from"]:
                    for f in res.index_result["facility"]:
                        for g in res.index_result["group-from"]:
                            rows.append([self.epi.static.get_property_name("parameter", p), self.get_locale_name(l), self.epi.static.get_property_name("facility", f), self.epi.static.get_property_name("group", g), model_parameters[p, l, f, g]])
            ret = pandas.DataFrame(rows, columns=['Parameter', 'Locale', 'Facility', 'Group', 'Value'])
        elif "group-from" in res.index_result and "group-to" in res.index_result:
            if value_mode == "current":
                facility_interaction = current_parameter.facility_interaction
            elif value_mode == "default":
                facility_interaction = default_parameter.facility_interaction
            for l in res.locale_result["locale-from"]:
                for f in res.index_result["facility"]:
                    for g1 in res.index_result["group-from"]:
                        for g2 in res.index_result["group-to"]:
                            rows.append([self.get_locale_name(l), self.epi.static.get_property_name("facility", f), self.epi.static.get_property_name("group", g1), self.epi.static.get_property_name("group", g2), facility_interaction[l, f, g1, g2]])
            ret = pandas.DataFrame(rows, columns=['Locale', 'Facility', 'Group From', 'Group To', 'Value'])
        elif "compartment-from" in res.index_result:
            comp_matrix = current_comp
            if value_mode == "default":
                comp_matrix = default_comp
            elif value_mode == "change":
                # This is a violation of forgetful property of MDP
                comp_matrix = current_comp - self.epi.get_previous_state().obs.current_comp
            for c in res.index_result["compartment-from"]:
                for l in res.locale_result["locale-from"]:
                    for g in res.index_result["group-from"]:
                        rows.append([self.epi.static.get_property_name("compartment", c), self.get_locale_name(l), self.epi.static.get_property_name("group", g), comp_matrix[c, l, g]])
            ret = pandas.DataFrame(rows, columns=['Compartment', 'Locale', 'Group', 'Value'])
        elif "facility" in res.index_result:
            if value_mode == "current":
                facility_timespent = current_parameter.facility_timespent
            elif value_mode == "default":
                facility_timespent = default_parameter.facility_timespent
            for l in res.locale_result["locale-from"]:
                for f in res.index_result["facility"]:
                    for g in res.index_result["group-from"]:
                        rows.append([self.get_locale_name(l), self.epi.static.get_property_name("facility", f), self.epi.static.get_property_name("group", g), facility_timespent])
            ret = pandas.DataFrame(rows, columns=['Locale', 'Facility', 'Group', 'Value'])
        return ret

    def apply(self, regex, multiplier, state, delta_parameter, itv_index):
        res = get_regsult(self.epi.static, self.epi.parser, itv_index, "apply", self.to_string_dict_regex(regex))
        nb_apply(self.epi.static, res, multiplier, state, delta_parameter, itv_index)

    def move(self, regex1, regex2, value, state, delta_parameter, itv_index):
        res1 = get_regsult(self.epi.static, self.epi.parser, itv_index, "move", self.to_string_dict_regex(regex1))
        res2 = get_regsult(self.epi.static, self.epi.parser, itv_index, "move", self.to_string_dict_regex(regex2))
        nb_move(self.epi.static, res1, res2, value, state, delta_parameter, itv_index)

    def add(self, regex, value, state, delta_parameter, itv_index):
        res = get_regsult(self.epi.static, self.epi.parser, itv_index, "add", self.to_string_dict_regex(regex))
        nb_add(self.epi.static, res, value, state, delta_parameter, itv_index)
