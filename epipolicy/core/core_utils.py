import numba as nb
import numpy as np
from numba.typed import Dict, List
from numba.types import string

from .core_optimize import get_matrix_with_tag
from ..obj.state import copy_state
from ..obj.schedule import Schedule
from ..obj.edge import Edge, InfectiousEdge, compute_transition
from ..obj.intervention import Intervention
from ..obj.control_parameter import ControlParameter
from ..obj.act import Act, get_default_act
from ..matrix.coo import CooMatrixType, CooMatrix
from ..matrix.sparse import get_csr_matrix_from_coo, get_csc_matrix_from_coo
from ..utility.utils import get_normalized_string, parse_date
from ..utility.singleton import MODES, WITHIN_RATIO_FACTILITY, DIE_TAG, SUS_TAG, INF_TAG, TRANS_TAG, ALIVE_COMPARTMENT
from ..parser.equation_parser import EquationParser

def get_tag_count(objs):
    tags = set()
    for obj in objs:
        if "tags" in obj:
            for tag in obj["tags"]:
                tags.add(tag)
    return len(tags)

def get_leaf_locales(locales):
    parents = set()
    for locale in locales:
        parents.add(locale["parent_id"])
    leaf_locales = []
    for locale in locales:
        if locale["id"] not in parents:
            leaf_locales.append(locale)
    return leaf_locales

def set_locales(static, locales):
    locale_id_to_index = {}
    parents = set()
    leaf_locales = []
    for index, locale in enumerate(locales):
        locale_id_to_index[locale["id"]] = index
        static.locale_hierarchy[get_normalized_string(locale["name"])] = Dict.empty(nb.int32, nb.boolean)
        parents.add(locale["parent_id"])
    for locale in locales:
        locale_id = locale["id"]
        if locale_id not in parents:
            # leaf locale
            leaf_locales.append(locale)
            leaf_locale_index = static.add_property_name("locale", locale_id)
            current_locale = locale
            while True:
                static.locale_hierarchy[get_normalized_string(current_locale["name"])][leaf_locale_index] = True
                if len(current_locale["parent_id"]) > 0:
                    current_locale = locales[locale_id_to_index[current_locale["parent_id"]]]
                else:
                    break
    all = Dict.empty(nb.int32, nb.boolean)
    for i in range(static.locale_count):
        all[i] = True
    static.locale_hierarchy[""] = all
    return leaf_locales

def set_groups(static, groups):
    if len(groups) == 0:
        static.add_property_name("group", "All")
    else:
        for group in groups:
            static.add_property_name("group", group["name"])

def set_compartments(static, compartments):
    for comp in compartments:
        comp_index = static.add_property_name("compartment", comp["name"])
        for tag in comp["tags"]:
            tag_index = static.add_property_name("compartment_tag", tag)
            static.compartment_tags[comp_index, tag_index] = 1

    alive_compartments = []
    for comp_index in range(static.compartment_count):
        if not static.compartment_has_tag(comp_index, DIE_TAG):
            alive_compartments.append(comp_index)
    static.alive_comp_list = np.zeros(len(alive_compartments), dtype=np.int32)
    for i, comp_index in enumerate(alive_compartments):
        static.alive_comp_list[i] = comp_index

def set_facilities(static, facilities):
    if len(facilities) == 0:
        static.add_property_name("facility", "All")
    else:
        for facility in facilities:
            static.add_property_name("facility", facility["name"])

# locales = leaf_locales
def populate(static, locales, groups):
    if static.group_count == 1:
        for i, locale in enumerate(locales):
            static.locale_group_pop[i, 0] = float(locale["population"])
    else:
        for i, group in enumerate(groups):
            for locale in group["locales"]:
                static.locale_group_pop[static.get_property_index("locale", locale["id"]), i] = float(locale["population"])
        for j, locale in enumerate(locales):
            for i in range(static.group_count):
                static.locale_group_pop[j, i] *= float(locale["population"])
    static.locale_group_pop_proportion = static.locale_group_pop/np.sum(static.locale_group_pop, axis=1).reshape(-1, 1)

def set_parameters(static, parser, parameters, custom_parameters):
    initial_parameter = static.default_state.unobs
    for param in parameters:
        param_index = static.add_property_name("parameter", param["name"])
        initial_parameter.model_parameters[param_index] = np.ones((static.locale_count, static.facility_count, static.group_count), dtype=np.float32)*float(param["default_value"])

        for tag in param["tags"]:
            tag_index = static.add_property_name("parameter_tag", tag)
            static.parameter_tags[param_index, tag_index] = 1

    for param in custom_parameters:
        param_index = static.get_property_index("parameter", param["param"])
        locale_index_list = parser.parse_locale_regex(param["locale"])
        group_index = static.get_property_index("group", param["group"])
        for locale_index in locale_index_list:
            initial_parameter.model_parameters[param_index, locale_index, :, group_index] = float(param["value"])

def set_modes(static):
    for mode in MODES:
        static.add_property_name("mode", mode)

def set_bias(static):
    static.mode_bias[static.get_property_index("mode", "airport")] = 0
    static.mode_bias[static.get_property_index("mode", "border")] = 1

def set_locale_mobility(static, parser, session, mode):
    group_coo = static.mode_group_coo[static.get_property_index("mode", mode)]
    if mode in session:
        mobilities = session[mode]["data"]
        for mob in mobilities:
            groups = parser.parse_group_regex(mob["group"])
            src = static.get_property_index("locale", mob["src_locale_id"])
            dst = static.get_property_index("locale", mob["dst_locale_id"])
            for g in groups:
                group_coo[g].set(src, dst, float(mob["value"]))

    default_within = WITHIN_RATIO_FACTILITY if static.locale_count > 1 else 1.0
    specifications = session[mode]["specifications"]
    if len(specifications) > 0:
        if 'impedance' in specifications[0]:
            default_within = float(specifications[0]['impedance'])/100

    for g in range(static.group_count):
        for l in range(static.locale_count):
            within = default_within
            if group_coo[g].has(l, l):
                within = group_coo[g].get(l, l)
            group_coo[g].set(l, l, within)
            for dest, v in group_coo[g].mat[l].items():
                if dest != l:
                    group_coo[g].set(l, dest, v*(1-within))

def set_locale_matrix(static, parser, session):
    mode_coo = List.empty_list(CooMatrixType)
    for i, mode in enumerate(MODES):
        set_locale_mobility(static, parser, session, mode)
        coo = CooMatrix((static.locale_count, static.locale_count)).add_signature(static.mode_group_coo[i])
        mode_coo.append(coo)
        static.mode_csr.append(get_csr_matrix_from_coo(coo))

    sum_mode_coo = CooMatrix((static.locale_count, static.locale_count)).add_signature(mode_coo)
    static.sum_mode_csr = get_csr_matrix_from_coo(sum_mode_coo)
    static.sum_mode_csc = get_csc_matrix_from_coo(sum_mode_coo)

    mode_reduced_csr = static.default_state.unobs.mode_reduced_csr
    for i in range(len(MODES)):
        reduced_csr = np.array([get_csr_matrix_from_coo(static.mode_group_coo[i][j].add_signature(mode_coo[i:i+1])).data for j in range(static.group_count)], dtype=np.float32)
        mode_reduced_csr.append(reduced_csr)

def set_facility_timespent_matrix(static, parser, facilities_timespent):
    facility_timespent = static.default_state.unobs.facility_timespent
    for timespent in facilities_timespent:
        locales = parser.parse_locale_regex(timespent["locales"])
        for l in locales:
            facility_timespent[l] = np.array(timespent["matrix"], dtype=np.float32)[:static.facility_count,:static.group_count]

def set_facility_interaction_matrix(static, parser, facilities_interactions):
    facility_interaction = static.default_state.unobs.facility_interaction
    for interactions in facilities_interactions:
        locales = parser.parse_locale_regex(interactions["locales"])
        for l in locales:
            facility_interaction[l] = np.array(interactions["facilities"], dtype=np.float32)[:static.facility_count, :static.group_count, :static.group_count]

def check_infectious_edge(static, fraction):
    transmission_rate_index, susceptible_comp_index, infectious_comp_index = None, None, None
    numer_param_index_list = []
    denom_param_index_list = []
    is_infectious_edge = True
    for mul in fraction.numer.muls:
        if static.has_property_name("parameter", mul):
            param_index = static.get_property_index("parameter", mul)
            if static.parameter_has_tag(param_index, TRANS_TAG):
                if transmission_rate_index is None:
                    transmission_rate_index = param_index
                else:
                    is_infectious_edge = False
            else:
                numer_param_index_list.append(param_index)
        elif static.has_property_name("compartment", mul):
            comp_index = static.get_property_index("compartment", mul)
            if static.compartment_has_tag(comp_index, INF_TAG):
                if infectious_comp_index is None:
                    infectious_comp_index = comp_index
                else:
                    is_infectious_edge = False
            elif static.compartment_has_tag(comp_index, SUS_TAG):
                if susceptible_comp_index is None:
                    susceptible_comp_index = comp_index
                else:
                    is_infectious_edge = False
    if len(fraction.denom) != 1:
        is_infectious_edge = False
    else:
        denom = fraction.denom[0].muls
        has_alive_compartment = False
        for mul in denom:
            if mul == ALIVE_COMPARTMENT:
                if has_alive_compartment == False:
                    has_alive_compartment = True
                else:
                    is_infectious_edge = False
            elif static.has_property_name("parameter", mul):
                denom_param_index_list.append(static.get_property_index("parameter", mul))
        if not has_alive_compartment:
            is_infectious_edge = False
    if transmission_rate_index is None or susceptible_comp_index is None or infectious_comp_index is None:
        is_infectious_edge = False
    return {
        "is_infectious_edge": is_infectious_edge,
        "transmission_rate_index": transmission_rate_index,
        "susceptible_comp_index": susceptible_comp_index,
        "infectious_comp_index": infectious_comp_index,
        "numer_param_index_list": np.array(numer_param_index_list, dtype=np.int32),
        "denom_param_index_list": np.array(denom_param_index_list, dtype=np.int32)
    }

def set_incidence_edge(static, edge):
    in_comp_list = []
    out_comp_list = []
    for comp, coef in edge.compartment_map.items():
        if coef > 0:
            in_comp_list.append(comp)
        else:
            out_comp_list.append(comp)
    for c1 in out_comp_list:
        if static.compartment_has_tag(c1, SUS_TAG):
            for c2 in in_comp_list:
                static.hashed_incidence_edges[c1*static.compartment_count+c2] = True

def set_edges(static, compartments):
    for comp_index, comp in enumerate(compartments):
        eq_parser = EquationParser(comp["equation"].replace('\n',''))
        fractions = eq_parser.get_all_fractions()
        for fraction in fractions:
            checks = check_infectious_edge(static, fraction)
            key = fraction.signature
            if not checks["is_infectious_edge"]:
                if key not in static.edges:
                    static.edges[key] = Edge(fraction, checks["transmission_rate_index"] is not None)
                static.edges[key].add_compartment(comp_index, fraction.numer.coef)
            else:
                if key not in static.infectious_edges:
                    static.infectious_edges[key] = InfectiousEdge(fraction, checks["transmission_rate_index"], checks["susceptible_comp_index"], checks["infectious_comp_index"], checks["numer_param_index_list"], checks["denom_param_index_list"])
                static.infectious_edges[key].add_compartment(comp_index, fraction.numer.coef)
    for edge in static.edges.values():
        if edge.has_transmission_rate:
            set_incidence_edge(static, edge)
        compute_transition(edge)
    for edge in static.infectious_edges.values():
        set_incidence_edge(static, edge)
        compute_transition(edge)

def seed(static, parser, initializers):
    locale_pop = np.sum(static.locale_group_pop, axis=1)
    initial_comp = static.default_state.obs.current_comp
    for initializer in initializers:
        locale_indices = parser.parse_locale_regex(initializer['locale_regex'])
        total_pop = 0
        for l in locale_indices:
            total_pop += locale_pop[l]
        group_indices = parser.parse_group_regex(initializer['group'])
        comp_indices = parser.parse_compartment_regex(initializer['compartment'])
        partition_count = len(group_indices) * len(comp_indices)
        if partition_count*total_pop > 0:
            val = float(initializer['value'])
            if val < 0:
                val += total_pop
            for c in comp_indices:
                for l in locale_indices:
                    for g in group_indices:
                        initial_comp[c, l, g] += val * locale_pop[l] / (total_pop * partition_count)

    susceptible_comp_matrix = get_matrix_with_tag(static, initial_comp, SUS_TAG)
    susceptible_comp_count = 0
    for comp_index in range(static.compartment_count):
        if static.compartment_has_tag(comp_index, SUS_TAG):
            susceptible_comp_count += 1
    for l in range(static.locale_count):
        for g in range(static.group_count):
            total_pop = np.sum(initial_comp[:, l, g])
            if susceptible_comp_matrix[l, g] > 0 and total_pop > 0:
                initial_comp[:, l, g] *= static.locale_group_pop[l, g]/total_pop
            elif susceptible_comp_count > 0:
                left_over = max(static.locale_group_pop[l, g] - total_pop, 0)
                for comp_index in range(static.compartment_count):
                    if static.compartment_has_tag(comp_index, SUS_TAG):
                        # First susceptible compartment will be initialize with left_over pop
                        initial_comp[comp_index, l, g] += left_over
                        break

def set_interventions(static, interventions, costs):
    for itv in interventions:
        static.add_property_name("intervention", itv["name"])
        intervention = Intervention(itv["name"], False)
        for cp in itv["control_params"]:
            intervention.cp_list.append(ControlParameter(cp["name"], float(cp["default_value"])))
        static.interventions.append(intervention)

    for cost in costs:
        static.add_property_name("intervention", cost["name"])
        cost_intervention = Intervention(cost["name"], True)
        for cp in cost["control_params"]:
            cost_intervention.cp_list.append(ControlParameter(cp["name"], float(cp["default_value"])))
        static.interventions.append(cost_intervention)

def set_schedule(static, features, schedules):
    start_date = parse_date(features["start_date"])
    end_date = parse_date(features["end_date"])
    horizon = int((end_date - start_date).days + 1)
    static.schedule = Schedule(horizon, copy_state(static.default_state))
    for schedule in schedules:
        itv_index = static.get_property_index("intervention", schedule['name'])
        for detail in schedule['detail']:
            itv_start = (parse_date(detail['start_date']) - start_date).days
            itv_end = (parse_date(detail['end_date']) - start_date).days
            cpv_list = np.zeros(len(static.interventions[itv_index].cp_list), dtype=np.float32)
            for cp_index, cp in enumerate(detail['control_params']):
                cpv_list[cp_index] = float(cp['value'])
            for t in range(itv_start, itv_end+1):
                static.schedule.add_act(t, Act(itv_index, detail['locales'], cpv_list))
    for t in range(static.schedule.horizon):
        for itv_index, itv in enumerate(static.interventions):
            if itv.is_cost:
                static.schedule.add_act(t, get_default_act(static, itv_index))

def set_optimize(static, session):
    if "optimize" in session:
        for itv in session["optimize"]["interventions"]:
            itv_id = static.get_property_index("intervention", itv["name"])
            for cp in itv["control_params"]:
                for static_cp in static.interventions[itv_id].cp_list:
                    if static_cp.name == cp["name"]:
                        static_cp.set_interval(float(cp["min_value"]), float(cp["max_value"]))
                        break

def make_static(static, parser, session):
    features = session['features']
    model = session["model"]
    locales = session["locales"]
    groups = session["groups"]
    facilities = session["facilities"]
    compartments = model["compartments"]
    parameters = model["parameters"]
    custom_parameters = session["groups_locales_parameters"]
    facilities_timespent = session["facilities_timespent"]
    facilities_interactions = session["facilities_interactions"]
    interventions = session["interventions"]
    schedules = session["schedules"]
    costs = session["costs"]
    initializers = session["initial_info"]["initializers"]

    leaf_locales = set_locales(static, locales)
    set_groups(static, groups)
    set_facilities(static, facilities)
    set_compartments(static, compartments)
    populate(static, leaf_locales, groups)
    set_parameters(static, parser, parameters, custom_parameters)
    set_modes(static)
    set_bias(static)

    set_locale_matrix(static, parser, session)
    set_facility_timespent_matrix(static, parser, facilities_timespent)
    set_facility_interaction_matrix(static, parser, facilities_interactions)

    set_edges(static, compartments)
    seed(static, parser, initializers)
    set_interventions(static, interventions, costs)
    set_schedule(static, features, schedules)

    set_optimize(static, session)
