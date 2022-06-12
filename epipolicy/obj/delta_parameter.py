import numpy as np
import numba as nb
from numba.types import DictType, ListType
from numba.typed import Dict, List
from numba.experimental import jitclass
from numba import njit

from ..matrix.coo import CooMatrix, CooMatrixType, CooMatrixListType, D2CooType, D1CooType, add_d3, get_coo_matrix_llist

delta_parameter_spec = [
    ("mode_group_coo", ListType(CooMatrixListType)),
    ("model_parameters", nb.float32[:,:,:,:]), # pxlxfxg
    ("facility_timespent", nb.float32[:,:,:]), # lxfxg
    ("facility_interaction", nb.float32[:,:,:,:]), # lxfxgxg
    ("comp_to_comp", DictType(nb.int32, D2CooType)), # gxcxcxlxl
    ("cost", nb.float64[:,:]), #ixl
    ("influence", nb.float32[:,:,:]) # ixlxg
]

@jitclass(delta_parameter_spec)
class DeltaParameter:

    def __init__(self, mode_group_coo, model_parameters, facility_timespent, facility_interaction, comp_to_comp, cost, influence):
        self.mode_group_coo = mode_group_coo
        self.model_parameters = model_parameters
        self.facility_timespent = facility_timespent
        self.facility_interaction = facility_interaction
        self.comp_to_comp = comp_to_comp
        self.cost = cost
        self.influence = influence

    def add_move(self, static, influence, g, c1, c2, l1, l2, amount):
        if g not in self.comp_to_comp:
            self.comp_to_comp[g] = Dict.empty(nb.int32, D1CooType)
        if c1 not in self.comp_to_comp[g]:
            self.comp_to_comp[g][c1] = Dict.empty(nb.int32, CooMatrixType)
        if c2 not in self.comp_to_comp[g][c1]:
            self.comp_to_comp[g][c1][c2] = CooMatrix((static.locale_count, static.locale_count))
        self.comp_to_comp[g][c1][c2].element_add(l1, l2, amount)
        if static.locale_group_pop[l1, g] > 0:
            influence[l1, g] += amount/static.locale_group_pop[l1, g]

    def combine(self, other):
        for m in range(len(self.mode_group_coo)):
            for g in range(len(self.mode_group_coo[0])):
                self.mode_group_coo[m][g].multiply(other.mode_group_coo[m][g])

        self.model_parameters *= other.model_parameters
        self.facility_timespent *= other.facility_timespent
        self.facility_interaction *= other.facility_interaction

        add_d3(self.comp_to_comp, other.comp_to_comp)
        self.cost += other.cost
        self.influence = 1-(1-self.influence)*(1-other.influence)

DeltaParameterType = DeltaParameter.class_type.instance_type

@njit
def get_initial_delta_parameter(static):
    mode_group_coo = get_coo_matrix_llist(static.mode_count, static.group_count, (static.locale_count, static.locale_count))
    model_parameters = np.ones((static.parameter_count, static.locale_count, static.facility_count, static.group_count), dtype=np.float32)
    facility_timespent = np.ones((static.locale_count, static.facility_count, static.group_count), dtype=np.float32)
    facility_interaction = np.ones((static.locale_count, static.facility_count, static.group_count, static.group_count), dtype=np.float32)
    comp_to_comp = Dict.empty(nb.int32, D2CooType)
    cost = np.zeros((static.intervention_count, static.locale_count), dtype=np.float64)
    influence = np.zeros((static.intervention_count, static.locale_count, static.group_count), dtype=np.float32)
    return DeltaParameter(mode_group_coo, model_parameters, facility_timespent, facility_interaction, comp_to_comp, cost, influence)
