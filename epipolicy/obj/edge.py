import numba as nb
from numba.experimental import jitclass
from numba.types import DictType
from numba.typed import Dict

from ..parser.fraction import FractionType
from ..utility.numba_type import IntFloatDictType
from ..utility.singleton import FLOAT_EPSILON

edge_spec = [
    ("fraction", FractionType),
    ("compartment_map", DictType(nb.int32, nb.float32)),
    ("has_transmission_rate", nb.boolean),
    ("gain_map", DictType(nb.int32, nb.float32)),
    ("lost_map", DictType(nb.int32, nb.float32)),
    ("move_map", DictType(nb.int32, IntFloatDictType))
]

@jitclass(edge_spec)
class Edge:

    def __init__(self, fraction, has_transmission_rate):
        self.fraction = fraction
        self.compartment_map = Dict.empty(nb.int32, nb.float32)
        self.has_transmission_rate = has_transmission_rate
        self.gain_map = Dict.empty(nb.int32, nb.float32)
        self.lost_map = Dict.empty(nb.int32, nb.float32)
        self.move_map = Dict.empty(nb.int32, IntFloatDictType)

    def add_compartment(self, comp_index, coef):
        if comp_index not in self.compartment_map:
            self.compartment_map[comp_index] = 0.0
        self.compartment_map[comp_index] += coef

EdgeType = Edge.class_type.instance_type

infectious_edge_spec = [
    ("fraction", FractionType),
    ("compartment_map", DictType(nb.int32, nb.float32)),
    ("transmission_rate_index", nb.int32),
    ("susceptible_comp_index", nb.int32),
    ("infectious_comp_index", nb.int32),
    ("numer_param_index_list", nb.int32[:]), # Unitialized
    ("denom_param_index_list", nb.int32[:]), # Unitialized
    ("gain_map", DictType(nb.int32, nb.float32)),
    ("lost_map", DictType(nb.int32, nb.float32)),
    ("move_map", DictType(nb.int32, IntFloatDictType))
]

@jitclass(infectious_edge_spec)
class InfectiousEdge:
    def __init__(self, fraction, transmission_rate_index, susceptible_comp_index, infectious_comp_index, numer_param_index_list, denom_param_index_list):
        self.fraction = fraction
        self.compartment_map = Dict.empty(nb.int32, nb.float32)
        self.transmission_rate_index = transmission_rate_index
        self.susceptible_comp_index = susceptible_comp_index
        self.infectious_comp_index = infectious_comp_index
        self.numer_param_index_list = numer_param_index_list
        self.denom_param_index_list = denom_param_index_list
        self.gain_map = Dict.empty(nb.int32, nb.float32)
        self.lost_map = Dict.empty(nb.int32, nb.float32)
        self.move_map = Dict.empty(nb.int32, IntFloatDictType)

    def add_compartment(self, comp_index, coef):
        if comp_index not in self.compartment_map:
            self.compartment_map[comp_index] = 0.0
        self.compartment_map[comp_index] += coef

InfectiousEdgeType = InfectiousEdge.class_type.instance_type

def set_move(move_map, c1, c2, v):
    if c1 not in move_map:
        move_map[c1] = Dict.empty(nb.int32, nb.float32)
    move_map[c1][c2] = v

def compute_transition(edge):
    pos = {}
    neg = {}
    sum_pos = 0
    sum_neg = 0
    for comp_index, coef in edge.compartment_map.items():
        if coef > 0:
            pos[comp_index] = coef
            sum_pos += coef
        else:
            neg[comp_index] = -coef
            sum_neg -= coef
    if abs(sum_pos-sum_neg) < FLOAT_EPSILON:
        for neg_index, neg_coef in neg.items():
            for pos_index, pos_coef in pos.items():
                set_move(edge.move_map, neg_index, pos_index, neg_coef*pos_coef/sum_pos)
    elif sum_pos > sum_neg:
        dif = sum_pos - sum_neg
        for pos_index, pos_coef in pos.items():
            v = dif*pos_coef/sum_pos
            edge.gain_map[pos_index] = v
            pos[pos_index] -= v
        if sum_neg > 0:
            for neg_index, neg_coef in neg.items():
                for pos_index, pos_coef in pos.items():
                    set_move(edge.move_map, neg_index, pos_index, neg_coef*pos_coef/sum_neg)
    else:
        dif = sum_neg - sum_pos
        for neg_index, neg_coef in neg.items():
            v = dif*neg_coef/sum_neg
            edge.lost_map[neg_index] = v
            neg[neg_index] -= v
        if sum_pos > 0:
            for neg_index, neg_coef in neg.items():
                for pos_index, pos_coef in pos.items():
                    set_move(edge.move_map, neg_index, pos_index, neg_coef*pos_coef/sum_pos)
