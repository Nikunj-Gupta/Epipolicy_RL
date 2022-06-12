import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba.types import ListType, DictType, string
from numba.typed import List, Dict
from numba import njit

from ..matrix.coo import CooMatrixListType, get_coo_matrix_llist
from ..matrix.sparse import SparseMatrixType
from ..obj.state import StateType, get_initial_state
from ..obj.schedule import ScheduleType
from ..obj.edge import EdgeType, InfectiousEdgeType
from ..obj.intervention import InterventionType
from ..obj.act import ActType, get_default_act
from ..utility.numba_type import SetType, StringIntDictType, StringListType, Int1DArrayType

static_spec = [
    ("locale_hierarchy", DictType(string, SetType)),
    ("interventions", ListType(InterventionType)),
    ("properties", ListType(string)),
    ("property_name_to_index", DictType(string, StringIntDictType)),
    ("property_index_to_name", DictType(string, StringListType)),
    ("mode_group_coo", ListType(CooMatrixListType)),
    ("mode_csr", ListType(SparseMatrixType)),
    ("sum_mode_csr", SparseMatrixType), # Uninitialized
    ("sum_mode_csc", SparseMatrixType), # Uninitialized
    ("mode_bias", nb.float32[:]),
    ("compartment_tags", nb.int32[:,:]),
    ("parameter_tags", nb.int32[:,:]),
    ("default_state", StateType),
    ("schedule", ScheduleType), # Unitialized
    ("locale_group_pop", nb.float64[:,:]),
    ("locale_group_pop_proportion", nb.float64[:,:]),
    ("locale_count", nb.int32),
    ("compartment_count", nb.int32),
    ("parameter_count", nb.int32),
    ("group_count", nb.int32),
    ("facility_count", nb.int32),
    ("intervention_count", nb.int32),
    ("mode_count", nb.int32),
    ("compartment_tag_count", nb.int32),
    ("parameter_tag_count", nb.int32),
    ("edges", DictType(string, EdgeType)),
    ("infectious_edges", DictType(string, InfectiousEdgeType)),
    ("hashed_incidence_edges", DictType(nb.int32, nb.boolean)),
    ("alive_comp_list", Int1DArrayType) # Uninitialized
]

@jitclass(static_spec)
class Static:
    def __init__(self, locale_count, compartment_count, parameter_count, group_count, facility_count, intervention_count, mode_count, compartment_tag_count, parameter_tag_count):
        self.locale_hierarchy = Dict.empty(string, SetType)
        self.interventions = List.empty_list(InterventionType)

        self.properties = List.empty_list(string)
        self.properties.append("parameter")
        self.properties.append("compartment")
        self.properties.append("locale")
        self.properties.append("mode")
        self.properties.append("group")
        self.properties.append("facility")
        self.properties.append("intervention")
        self.properties.append("compartment_tag")
        self.properties.append("parameter_tag")

        self.locale_count = locale_count
        self.compartment_count = compartment_count
        self.parameter_count = parameter_count
        self.group_count = group_count
        self.facility_count = facility_count
        self.intervention_count = intervention_count
        self.mode_count = mode_count
        self.compartment_tag_count = compartment_tag_count
        self.parameter_tag_count = parameter_tag_count

        self.property_name_to_index = Dict.empty(string, StringIntDictType)
        self.property_index_to_name = Dict.empty(string, StringListType)
        for property in self.properties:
            self.property_name_to_index[property] = Dict.empty(string, nb.int32)
            self.property_index_to_name[property] = List.empty_list(string)

        self.mode_group_coo = get_coo_matrix_llist(mode_count, group_count, (locale_count, locale_count))
        self.mode_csr = List.empty_list(SparseMatrixType)
        self.mode_bias = np.zeros(mode_count, dtype=np.float32)
        self.compartment_tags = np.zeros((compartment_count, compartment_tag_count), dtype=np.int32)
        self.parameter_tags = np.zeros((parameter_count, parameter_tag_count), dtype=np.int32)

        self.default_state = get_initial_state(self)

        self.locale_group_pop = np.zeros((locale_count, group_count), dtype=np.float64)
        self.locale_group_pop_proportion = np.zeros((locale_count, group_count), dtype=np.float64)

        self.edges = Dict.empty(string, EdgeType)
        self.infectious_edges = Dict.empty(string, InfectiousEdgeType)
        self.hashed_incidence_edges = Dict.empty(nb.int32, nb.boolean)

    def add_property_name(self, property, name):
        if name not in self.property_name_to_index[property]:
            self.property_name_to_index[property][name] = len(self.property_index_to_name[property])
            self.property_index_to_name[property].append(name)
        return self.property_name_to_index[property][name]

    def get_property_name(self, property, index):
        return self.property_index_to_name[property][index]

    def get_property_index(self, property, name):
        return self.property_name_to_index[property][name]

    def has_property_name(self, property, name):
        return name in self.property_name_to_index[property]

    def compartment_has_tag(self, comp_index, tag_name):
        if self.has_property_name("compartment_tag", tag_name):
            return self.compartment_tags[comp_index, self.get_property_index("compartment_tag", tag_name)] > 0
        return False

    def parameter_has_tag(self, parameter_index, tag_name):
        if self.has_property_name("parameter_tag", tag_name):
            return self.parameter_tags[parameter_index, self.get_property_index("parameter_tag", tag_name)] > 0
        return False

    def generate_default_action(self):
        action = List.empty_list(ActType)
        for itv_index in range(len(self.interventions)):
            action.append(get_default_act(self, itv_index))
        return action

    def generate_empty_action(self):
        action = List.empty_list(ActType)
        for itv_index, itv in enumerate(self.interventions):
            if itv.is_cost:
                action.append(get_default_act(self, itv_index))
        return action

StaticType = Static.class_type.instance_type

def get_total_incidence_matrix(static, obs):
    res = np.zeros((static.locale_count, static.group_count), dtype=np.float64)
    for h in static.hashed_incidence_edges:
        c1 = h // static.compartment_count
        c2 = h % static.compartment_count
        res += obs.cumulative_move[c1, c2]
        res -= obs.cumulative_move[c2, c1]
    return res # l x g

def get_total_new_matrix(static, obs):
    res = obs.cumulative_gain
    for c1 in range(static.compartment_count):
        for c2 in range(static.compartment_count):
            if c1 != c2:
                add = obs.cumulative_move[c2, c1] - obs.cumulative_move[c1, c2]
                res[c1] += np.maximum(add, 0)
    return res # c x l x g

def get_total_delta_matrix(static, obs):
    res = obs.cumulative_gain - obs.cumulative_lost
    for c1 in range(static.compartment_count):
        for c2 in range(static.compartment_count):
            if c1 != c2:
                res[c1] -= obs.cumulative_move[c1, c2]
                res[c2] += obs.cumulative_move[c1, c2]
    return res # c x l x g
