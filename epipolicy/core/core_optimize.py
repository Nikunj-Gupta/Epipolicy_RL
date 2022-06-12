import numpy as np
import numba as nb
from numba import njit

from ..obj.parameter import get_interpolated_parameter
from ..obj.observable import get_initial_observable, get_observable_from_flat
from ..utility.singleton import ALIVE_COMPARTMENT, DIE_TAG, FLOAT_EPSILON, HOS_TAG

@njit
def get_matrix_with_tag(static, current_comp, tag):
    res = np.zeros((current_comp.shape[1], current_comp.shape[2]), dtype=np.float64)
    if static.has_property_name("compartment_tag", tag):
        tag_index = static.get_property_index("compartment_tag", tag)
        for comp_index in range(static.compartment_count):
            res += current_comp[comp_index]*static.compartment_tags[comp_index, tag_index]
    return res

@njit
def get_matrix_without_tag(static, current_comp, tag):
    if not static.has_property_name("compartment_tag", tag):
        return np.sum(current_comp, axis=0)
    tag_index = static.get_property_index("compartment_tag", tag)
    res = np.zeros((current_comp.shape[1], current_comp.shape[2]), dtype=np.float64)
    for comp_index in range(static.compartment_count):
        res += current_comp[comp_index]*(1 - static.compartment_tags[comp_index, tag_index])
    return res

@njit
def get_matrix_without_tags(static, current_comp, tags):
    tags_index = -np.ones(len(tags))
    for i, tag in enumerate(tags):
        if static.has_property_name("compartment_tag", tag):
            tags_index[i] = static.get_property_index("compartment_tag", tag)
    res = np.zeros((current_comp.shape[1], current_comp.shape[2]), dtype=np.float64)
    for comp_index in range(static.compartment_count):
        coef = 1
        for tag_index in tags_index:
            if tag_index >= 0:
                coef *= (1 - static.compartment_tags[comp_index, int(tag_index)])
        res += current_comp[comp_index]*coef
    return res

@njit
def compute_summant(static, current_comp, model_parameters, alive_matrix, summant):
    res = np.ones((static.locale_count, static.group_count), dtype=np.float64)
    for var in summant.muls:
        if static.has_property_name("parameter", var):
            res *= model_parameters[static.get_property_index("parameter", var), :, 0]
        elif var == ALIVE_COMPARTMENT:
            res *= alive_matrix
        elif static.has_property_name("compartment", var):
            res *= current_comp[static.get_property_index("compartment", var)]
    return res

@njit
def compute_fraction(static, current_comp, model_parameters, alive_matrix, fraction):
    numer, denom = np.zeros((2, static.locale_count, static.group_count), dtype=np.float64)
    numer += compute_summant(static, current_comp, model_parameters, alive_matrix, fraction.numer)
    for summant in fraction.denom:
        denom += summant.coef * compute_summant(static, current_comp, model_parameters, alive_matrix, summant)
    for i in range(static.locale_count):
        for j in range(static.group_count):
            if denom[i,j] > 0:
                numer[i,j] /= denom[i,j]
    return numer

# Ref: https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h
# csr to csc
@njit
def compute_combined_mobility(static, mode_reduced_csr):
    nnz = static.sum_mode_csr.nnz
    combined_reduced_csr, combined_reduced_csc = np.zeros((2, static.group_count, nnz), dtype=np.float32)
    combined_indptr = static.sum_mode_csc.indptr.copy()
    for r in range(static.locale_count):
        moving_index = np.zeros(static.mode_count, dtype=np.int32)
        for i in range(static.mode_count):
            moving_index[i] = static.mode_csr[i].indptr[r]
        for col_index in range(static.sum_mode_csr.indptr[r], static.sum_mode_csr.indptr[r+1]):
            c = static.sum_mode_csr.indices[col_index]
            v = np.zeros(static.group_count, dtype=np.float32)
            for i in range(static.mode_count):
                if moving_index[i] < static.mode_csr[i].indptr[r+1] and c == static.mode_csr[i].indices[moving_index[i]]:
                    for g in range(static.group_count):
                        v[g] += static.mode_bias[i]*mode_reduced_csr[i][g, moving_index[i]]
                    moving_index[i] += 1

            dest = combined_indptr[c]
            for g in range(static.group_count):
                combined_reduced_csr[g, col_index] = v[g]
                combined_reduced_csc[g, dest] = v[g]
            combined_indptr[c] += 1
    return combined_reduced_csr, combined_reduced_csc

# c2 = infectious_comp_index
# u0 = group_index
# j = locale_index
# v = facility_index
@njit
def compute_foi_entry(static, current_comp, facility_timespent, facility_interaction, alive_matrix, c2, u0, j, v, combined_reduced_csc):
    numer = 0
    denom = 0
    indices = static.sum_mode_csc.indices
    indptr = static.sum_mode_csc.indptr
    default_facility_interaction = static.default_state.unobs.facility_interaction
    for u in range(static.group_count):
        numer_summant = 0
        denom_summant = 0
        for row_index in range(indptr[j], indptr[j+1]):
            i = indices[row_index]
            val = combined_reduced_csc[u, row_index]
            numer_summant += current_comp[c2,i,u]*val
            denom_summant += alive_matrix[i,u]*val
        #print("HERE", u0, u, numer_summant, denom_summant, facility_timespent[j,v,u], facility_interaction[j,v,u0,u], facility_interaction[j,v,u,u0])
        factor = facility_timespent[j,v,u]*facility_interaction[j,v,u0,u]*facility_interaction[j,v,u,u0]
        numer += numer_summant*factor
        # Previously
        # denom += denom_summant*factor
        # Previously KU
        # denom += denom_summant*facility_timespent[j,v,u]
        denom += denom_summant*facility_timespent[j,v,u]*default_facility_interaction[j,v,u0,u]*default_facility_interaction[j,v,u,u0]
    #print("G1", u0, "G2", u, "J", j, "v", v, numer, denom)
    if denom <= FLOAT_EPSILON:
        denom = 1.0
    return numer / denom

# c2 = infectious_comp_index
@njit
def compute_foi_matrix(static, current_comp, facility_timespent, facility_interaction, alive_matrix, c2, combined_reduced_csc):
    foi_matrix = np.zeros((static.group_count, static.locale_count, static.facility_count), dtype=np.float64)
    for u0 in range(static.group_count):
        for j in range(static.locale_count):
            for v in range(static.facility_count):
                foi_matrix[u0,j,v] = compute_foi_entry(static, current_comp, facility_timespent, facility_interaction, alive_matrix, c2, u0, j, v, combined_reduced_csc)
    return foi_matrix

# c0 = susceptible_comp_index
# p0 = transmission_rate_index
@njit
def compute_infectious_edge(static, current_comp, model_parameters, facility_timespent, foi_matrix, c0, p0, combined_reduced_csr, numer_param_index_list, denom_param_index_list):
    res = np.zeros((static.locale_count, static.group_count), dtype=np.float64)
    indices = static.sum_mode_csr.indices
    indptr = static.sum_mode_csr.indptr
    for i0 in range(static.locale_count):
        for u0 in range(static.group_count):
            entry = 0
            for col_index in range(indptr[i0], indptr[i0+1]):
                j = indices[col_index]
                val = combined_reduced_csr[u0][col_index]
                tmp = 0
                for v in range(static.facility_count):
                    coef = 1.0
                    for p1 in numer_param_index_list:
                        coef *= model_parameters[p1,j,v,u0]
                    for p1 in denom_param_index_list:
                        if abs(model_parameters[p1,j,v,u0]) > 0:
                            coef /= model_parameters[p1,j,v,u0]
                    tmp += facility_timespent[j,v,u0]*foi_matrix[u0,j,v]*model_parameters[p0,j,v,u0]*coef
                    # if u0 == 8:
                    #     print(tmp, val, facility_timespent[j,v,u0], foi_matrix[u0,j,v], model_parameters[p0,j,v,u0], coef)
                entry += tmp*val
            res[i0,u0] = entry*current_comp[c0,i0,u0]
            #print("I", i0, "U", u0, res[i0,u0], entry, current_comp[c0,i0,u0])
    return res

@njit
def compute_delta_observable(static, obs, parameter):
    delta_obs = get_initial_observable(static)
    alive_matrix = get_matrix_without_tags(static, obs.current_comp, [DIE_TAG, HOS_TAG])
    #alive_matrix = get_matrix_without_tag(static, obs.current_comp, DIE_TAG)
    combined_reduced_csr, combined_reduced_csc = compute_combined_mobility(static, parameter.mode_reduced_csr)
    for edge in static.edges.values():
        delta_edge = compute_fraction(static, obs.current_comp, parameter.model_parameters, alive_matrix, edge.fraction)
        for comp_index, coef in edge.compartment_map.items():
            delta_obs.current_comp[comp_index] += delta_edge*coef
        for comp_index, coef in edge.gain_map.items():
            delta_obs.cumulative_gain[comp_index] += delta_edge*coef
        for comp_index, coef in edge.lost_map.items():
            delta_obs.cumulative_lost[comp_index] += delta_edge*coef
        for c1 in edge.move_map:
            for c2, coef in edge.move_map[c1].items():
                delta_obs.cumulative_move[c1, c2] += delta_edge*coef
    for infectious_edge in static.infectious_edges.values():
        #print("SIG", infectious_edge.fraction.signature)
        foi_matrix = compute_foi_matrix(static, obs.current_comp, parameter.facility_timespent, parameter.facility_interaction, alive_matrix, infectious_edge.infectious_comp_index, combined_reduced_csc)
        #print(foi_matrix)
        delta_edge = compute_infectious_edge(static, obs.current_comp, parameter.model_parameters, parameter.facility_timespent, foi_matrix, infectious_edge.susceptible_comp_index, infectious_edge.transmission_rate_index, combined_reduced_csr, infectious_edge.numer_param_index_list, infectious_edge.denom_param_index_list)
        for comp_index, coef in infectious_edge.compartment_map.items():
            delta_obs.current_comp[comp_index] += delta_edge*coef
        for comp_index, coef in infectious_edge.gain_map.items():
            delta_obs.cumulative_gain[comp_index] += delta_edge*coef
        for comp_index, coef in infectious_edge.lost_map.items():
            delta_obs.cumulative_lost[comp_index] += delta_edge*coef
        for c1 in infectious_edge.move_map:
            for c2, coef in infectious_edge.move_map[c1].items():
                delta_obs.cumulative_move[c1, c2] += delta_edge*coef
    next_current_comp = obs.current_comp + delta_obs.current_comp
    for g in parameter.delta_comp_to_comp:
        for c1 in parameter.delta_comp_to_comp[g]:
            for c2 in parameter.delta_comp_to_comp[g][c1]:
                coo = parameter.delta_comp_to_comp[g][c1][c2]
                for l1 in coo.mat:
                    for l2, v in coo.mat[l1].items():
                        new_v = min(v, next_current_comp[c1, l1, g])
                        next_current_comp[c1, l1, g] -= new_v
                        next_current_comp[c2, l2, g] += new_v
                        if l1 == l2:
                            delta_obs.cumulative_move[c1, c2, l1, g] += new_v
    delta_obs.current_comp = next_current_comp - obs.current_comp
    delta_obs.cumulative_cost += parameter.delta_cost
    return delta_obs

@njit
def forward(static, flat, parameter):
    obs = get_observable_from_flat(static, flat)
    delta_obs = compute_delta_observable(static, obs, parameter)
    return delta_obs.flatten()

@njit
def scipy_fun(t, flat, static, src_parameter, gap_parameter):
    # print("INTERPOLATING at", t)
    parameter = get_interpolated_parameter(src_parameter, gap_parameter, t)
    return forward(static, flat, parameter)
