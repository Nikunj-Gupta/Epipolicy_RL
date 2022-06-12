import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba.types import ListType, DictType, Array
from numba.typed import List, Dict
from numba import njit

from ..matrix.coo import D2CooType, copy_coo_matrix, get_normalized_matrix, copy_d3_coo, scale_d3, add_d3
from ..matrix.sparse import get_csr_matrix_from_coo
from ..matrix.dense import get_normalized_facility_timespent, get_normalized_facility_interaction
from ..utility.utils import quadratic_step_function
from ..utility.numba_type import Float2DArrayType

parameter_spec = [
    ("mode_reduced_csr", ListType(Float2DArrayType)),
    ("model_parameters", nb.float32[:,:,:,:]), #pxlxfxg
    ("facility_timespent", nb.float32[:,:,:]), #lxfxg
    ("facility_interaction", nb.float32[:,:,:,:]), #lxfxgxg
    ("delta_comp_to_comp", DictType(nb.int32, D2CooType)), #gxcxcxlxl
    ("delta_cost", nb.float64[:,:]) #ixl
]

@jitclass(parameter_spec)
class Parameter:

    def __init__(self, mode_reduced_csr, model_parameters, facility_timespent, facility_interaction, delta_comp_to_comp, delta_cost):
        self.mode_reduced_csr = mode_reduced_csr
        self.model_parameters = model_parameters
        self.facility_timespent = facility_timespent
        self.facility_interaction = facility_interaction
        self.delta_comp_to_comp = delta_comp_to_comp
        self.delta_cost = delta_cost

ParameterType = Parameter.class_type.instance_type

@njit
def copy_parameter(parameter):
    copied_mode_reduced_csr = List.empty_list(Float2DArrayType)
    for reduced_csr in parameter.mode_reduced_csr:
        copied_mode_reduced_csr.append(reduced_csr.copy())
    return Parameter(copied_mode_reduced_csr, parameter.model_parameters.copy(), parameter.facility_timespent.copy(), parameter.facility_interaction.copy(), copy_d3_coo(parameter.delta_comp_to_comp), parameter.delta_cost.copy())

@njit
def get_initial_parameter(static):
    mode_reduced_csr = List.empty_list(Float2DArrayType)
    model_parameters = np.zeros((static.parameter_count, static.locale_count, static.facility_count, static.group_count), dtype=np.float32)
    facility_timespent = np.ones((static.locale_count, static.facility_count, static.group_count), dtype=np.float32) / static.facility_count
    facility_interaction = np.ones((static.locale_count, static.facility_count, static.group_count, static.group_count), dtype=np.float32) / static.group_count
    delta_comp_to_comp = Dict.empty(nb.int32, D2CooType)
    delta_cost = np.zeros((static.intervention_count, static.locale_count), dtype=np.float64)
    return Parameter(mode_reduced_csr, model_parameters, facility_timespent, facility_interaction, delta_comp_to_comp, delta_cost)

@njit
def get_parameter_from_delta(static, delta_parameter):
    mode_reduced_csr = List.empty_list(Float2DArrayType)
    default_parameter = static.default_state.unobs
    for i in range(static.mode_count):
        reduced_csr = np.zeros((static.group_count, len(static.mode_csr[i].data)), dtype=np.float32)
        for g in range(static.group_count):
            coo = copy_coo_matrix(static.mode_group_coo[i][g]).multiply(delta_parameter.mode_group_coo[i][g])
            reduced_csr[g] = get_csr_matrix_from_coo(get_normalized_matrix(static.mode_group_coo[i][g], coo)).data
        mode_reduced_csr.append(reduced_csr)
    model_parameters = default_parameter.model_parameters * delta_parameter.model_parameters
    facility_timespent = get_normalized_facility_timespent(default_parameter.facility_timespent*delta_parameter.facility_timespent)
    facility_interaction = get_normalized_facility_interaction(default_parameter.facility_interaction*delta_parameter.facility_interaction)
    return Parameter(mode_reduced_csr, model_parameters, facility_timespent, facility_interaction, delta_parameter.comp_to_comp, delta_parameter.cost)

@njit
def get_gap_parameter(previous_param, current_param):
    gap_mode_reduced_csr = List.empty_list(Float2DArrayType)
    for i in range(len(current_param.mode_reduced_csr)):
        gap_mode_reduced_csr.append(current_param.mode_reduced_csr[i]-previous_param.mode_reduced_csr[i])
    gap_model_parameters = current_param.model_parameters - previous_param.model_parameters
    gap_facility_timespent = current_param.facility_timespent - previous_param.facility_timespent
    gap_facility_interaction = current_param.facility_interaction - previous_param.facility_interaction
    gap_delta_comp_to_comp = add_d3(scale_d3(copy_d3_coo(previous_param.delta_comp_to_comp), -1), current_param.delta_comp_to_comp)
    gap_delta_cost = current_param.delta_cost - previous_param.delta_cost
    return Parameter(gap_mode_reduced_csr, gap_model_parameters, gap_facility_timespent, gap_facility_interaction, gap_delta_comp_to_comp, gap_delta_cost)

@njit
def get_interpolated_parameter(current_param, gap_param, step):
    float_step = nb.float32(step)
    quad_step = nb.float32(quadratic_step_function(step))
    mode_reduced_csr = List.empty_list(Float2DArrayType)
    for i in range(len(current_param.mode_reduced_csr)):
        mode_reduced_csr.append(current_param.mode_reduced_csr[i] + gap_param.mode_reduced_csr[i]*float_step)
    model_parameters = current_param.model_parameters + gap_param.model_parameters*float_step
    facility_timespent = current_param.facility_timespent + gap_param.facility_timespent*float_step
    facility_interaction = current_param.facility_interaction + gap_param.facility_interaction*float_step
    delta_comp_to_comp = add_d3(scale_d3(copy_d3_coo(gap_param.delta_comp_to_comp), quad_step), current_param.delta_comp_to_comp)
    delta_cost = current_param.delta_cost + gap_param.delta_cost*quad_step
    return Parameter(mode_reduced_csr, model_parameters, facility_timespent, facility_interaction, delta_comp_to_comp, delta_cost)
