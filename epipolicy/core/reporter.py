import math
import numpy as np

from .core_optimize import get_matrix_with_tag, get_matrix_without_tag
from ..obj.delta_parameter import get_initial_delta_parameter
from ..obj.static import get_total_incidence_matrix, get_total_new_matrix
from ..utility.singleton import INF_TAG, TRANS_TAG, SUS_TAG, DIE_TAG, R_WINDOW
from ..utility.utils import get_consecutive_difference, get_normalized_string, true_divide, get_infectivity

class Reporter:

    def __init__(self, epi, simulation_id, connection, locations):
        self.epi = epi
        self.simulation_id = simulation_id
        self.connection = connection
        self.locations = locations

    def start(self):
        if self.connection is not None:
            self.connection.update(self.simulation_id, "running")

    def end(self):
        if self.connection is not None:
            self.connection.update(self.simulation_id, "finished")

    def report(self):
        self.make_responses()
        if self.connection is not None:
            threads = [self.connection.post(response) for response in self.responses]
            for thread in threads:
                if thread is not None:
                    thread.join()

    def get_report_length(self):
        return len(self.epi.history.states)+1

    def compute_history(self):
        self.states = [self.epi.static.schedule.initial_state] + self.epi.history.states
        self.delta_parameters = [get_initial_delta_parameter(self.epi.static)] + self.epi.history.delta_parameters
        initial_action = self.epi.static.generate_empty_action()
        if self.epi.config["detached_initial_state"]:
            initial_action = self.epi.static.schedule.action_list[0]
        self.actions = [initial_action] + self.epi.history.actions

    def compute_incidences(self):
        total_initial_infection = get_matrix_with_tag(self.epi.static, self.states[0].obs.current_comp, INF_TAG)
        total_incidences = [get_total_incidence_matrix(self.epi.static, state.obs)+total_initial_infection for state in self.states]
        self.incidences = np.array(get_consecutive_difference(total_incidences))

    def compute_comp_news(self):
        total_comp_news = [get_total_new_matrix(self.epi.static, state.obs) for state in self.states]
        self.comp_news = np.array(get_consecutive_difference(total_comp_news))

    def compute_average_infectious_duration(self):
        T = self.get_report_length()
        total_infectious_time, total_incidence = 0, 0
        self.average_infectious_duration = np.zeros(T, dtype=np.float64)
        if self.epi.config["average_infectious_duration"] == "instantaneous_based":
            for t in range(T):
                total_infectious_time += np.sum(get_matrix_with_tag(self.epi.static, self.states[t].obs.current_comp, INF_TAG))
                total_incidence += np.sum(self.incidences[t])
                if total_incidence > 0:
                    self.average_infectious_duration[t] = total_infectious_time / total_incidence
        elif self.epi.config["average_infectious_duration"] == "beta_window_based":
            max_beta = 0
            for param_index in range(self.epi.static.parameter_count):
                if self.epi.static.parameter_has_tag(param_index, TRANS_TAG) > 0:
                    max_beta = max(max_beta, np.amax(self.epi.static.default_state.unobs.model_parameters[param_index]))
            beta_window = int(math.ceil(1.0/max_beta)) if max_beta > 0 else self.get_report_length()

            for t in range(T):
                if t >= beta_window:
                    total_infectious_time += np.sum(get_matrix_with_tag(self.epi.static, self.states[t].obs.current_comp-self.states[t-betaWindow].obs.current_comp, INF_TAG))
                    total_incidence += np.sum(self.incidences[t] - self.incidences[t-beta_window])
                else:
                    total_infectious_time += np.sum(get_matrix_with_tag(self.epi.static, self.states[t].obs.current_comp, INF_TAG))
                    total_incidence += np.sum(self.incidences[t])
                if total_incidence > 0:
                    self.average_infectious_duration[t] = total_infectious_time / total_incidence

    def compute_location_statistics(self):
        T = self.get_report_length()
        self.location_incidences = np.zeros((T, len(self.locations), self.epi.static.group_count), dtype=np.float64)
        self.location_comp_news = np.zeros((T, self.epi.static.compartment_count, len(self.locations), self.epi.static.group_count), dtype=np.float64)
        self.location_counts = np.zeros((T, self.epi.static.compartment_count, len(self.locations), self.epi.static.group_count), dtype=np.float64)
        self.location_itv_costs = np.zeros((T, self.epi.static.intervention_count, len(self.locations)), dtype=np.float64)
        self.location_influences = np.zeros((T, self.epi.static.intervention_count, len(self.locations), self.epi.static.group_count), dtype=np.float64)
        self.location_avg_cpvs = [[[[0 for k in range(len(self.epi.static.interventions[i].cp_list))] for i in range(self.epi.static.intervention_count)] for j in range(len(self.locations))] for t in range(T)]
        self.location_avg_model_parameters = np.zeros((T, self.epi.static.parameter_count, len(self.locations)), dtype=np.float64)
        self.location_cumulative_moves = np.zeros((T, self.epi.static.compartment_count, self.epi.static.compartment_count, len(self.locations)), dtype=np.float64)

        for t in range(T):
            current_comp = self.states[t].obs.current_comp
            cumulative_move = self.states[t].obs.cumulative_move
            model_parameters = self.states[t].unobs.model_parameters
            facility_timespent = self.states[t].unobs.facility_timespent
            location_cpvs_count = [[[0 for k in range(len(self.epi.static.interventions[i].cp_list))] for i in range(self.epi.static.intervention_count)] for j in range(len(self.locations))]
            location_model_parameters_count = np.zeros((self.epi.static.parameter_count, len(self.locations)), dtype=np.float64)
            susceptible_matrix = get_matrix_with_tag(self.epi.static, current_comp, SUS_TAG)
            alive_matrix = get_matrix_without_tag(self.epi.static, current_comp, DIE_TAG)

            for location_index, location in enumerate(self.locations):
                locales = self.epi.static.locale_hierarchy[get_normalized_string(location["name"])]
                for l in locales:
                    self.location_incidences[t,location_index] += self.incidences[t,l]
                    self.location_comp_news[t,:,location_index] += self.comp_news[t,:,l]
                    self.location_counts[t,:,location_index] += current_comp[:,l]
                    self.location_itv_costs[t,:,location_index] += self.states[t].obs.cumulative_cost[:,l]
                    self.location_influences[t,:,location_index] += self.delta_parameters[t].influence[:,l]*alive_matrix[l]
                    alive_timespent = facility_timespent[l] * alive_matrix[l]
                    self.location_avg_model_parameters[t,:,location_index] += np.sum(model_parameters[:,l]*alive_timespent, axis=(1,2))
                    location_model_parameters_count[:,location_index] += np.sum(alive_timespent)
                    self.location_cumulative_moves[t,:,:,location_index] = np.sum(cumulative_move[:,:,l], axis=2)

                for act in self.actions[t]:
                    act_locales = self.epi.parser.parse_locale_regex(act.locale_regex)
                    for l in act_locales:
                        if l in locales:
                            total_alive = np.sum(alive_matrix[l])
                            for cp_index, cpv in enumerate(act.cpv_list):
                                self.location_avg_cpvs[t][location_index][act.index][cp_index] += cpv*total_alive
                                location_cpvs_count[location_index][act.index][cp_index] += total_alive
                for itv_index in range(self.epi.static.intervention_count):
                    for cp_index in range(len(self.epi.static.interventions[itv_index].cp_list)):
                        if location_cpvs_count[location_index][itv_index][cp_index] > 0:
                            self.location_avg_cpvs[t][location_index][itv_index][cp_index] /= location_cpvs_count[location_index][itv_index][cp_index]

            location_alive_matrix = get_matrix_without_tag(self.epi.static, self.location_counts[t], DIE_TAG)
            self.location_influences[t] = true_divide(self.location_influences[t], location_alive_matrix)
            self.location_avg_model_parameters[t] = true_divide(self.location_avg_model_parameters[t], location_model_parameters_count)

    def compute_window_statistics(self):
        T = self.get_report_length()
        self.location_window_incidence = np.zeros((T, len(self.locations)), dtype=np.float64)
        self.location_window_infectious_count = np.zeros((T, len(self.locations)), dtype=np.float64)
        self.window_infectious_duration = np.zeros(T, dtype=np.float64)

        for t in range(T):
            if t >= R_WINDOW:
                self.location_window_incidence[t] += np.sum(self.location_incidences[t]-self.location_incidences[t-R_WINDOW], axis=1)
                self.location_window_infectious_count[t] += np.sum(get_matrix_with_tag(self.epi.static, self.location_counts[t] - self.location_counts[t-R_WINDOW], INF_TAG), axis=1)
                self.window_infectious_duration[t] += self.average_infectious_duration[t] - self.average_infectious_duration[t-R_WINDOW]
            else:
                self.location_window_incidence[t] += np.sum(self.location_incidences[t], axis=1)
                self.location_window_infectious_count[t] += np.sum(get_matrix_with_tag(self.epi.static, self.location_counts[t], INF_TAG), axis=1)
                self.window_infectious_duration[t] += self.average_infectious_duration[t]
            if t > 0:
                self.location_window_incidence[t] += self.location_window_incidence[t-1]
                self.location_window_infectious_count[t] += self.location_window_infectious_count[t-1]
                self.window_infectious_duration[t] += self.window_infectious_duration[t-1]

    def compute_instantaneous_reproductive_number(self):
        T = self.get_report_length()
        self.location_instant_R = np.zeros((T, len(self.locations)), dtype=np.float64)
        if self.epi.config["instantaneous_reproductive_number"] == "infectious_duration_based":
            for t in range(T):
                coef = self.window_infectious_duration[t]/min(t+1, R_WINDOW)
                self.location_instant_R[t] = np.maximum(true_divide(self.location_window_incidence[t]*coef, self.location_window_infectious_count[t], out=np.zeros_like(self.location_instant_R[t])), 0)
        elif self.epi.config["instantaneous_reproductive_number"] == "infectivity_based":
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325187/
            for t in range(T):
                for s in range(1, t+1):
                    self.location_instant_R[t] += get_infectivity(s)*np.sum(self.location_incidences[t-s], axis=1)
                It = np.sum(self.location_incidences[t], axis=1)
                self.location_instant_R[t] = true_divide(It, self.location_instant_R[t], out=self.location_instant_R[t])

    def compute_basic_reproductive_number(self):
        T = self.get_report_length()
        self.location_basic_R = np.zeros((T, len(self.locations)), dtype=np.float64)
        if self.epi.config["basic_reproductive_number"] == "beta_based":
            final_beta = np.zeros((T, len(self.locations)), dtype=np.float64)
            for t in range(T):
                sum_weights = np.zeros(len(self.locations), dtype=np.float64)
                for h in self.epi.static.hashed_incidence_edges:
                    c1 = h // self.epi.static.compartment_count
                    c2 = h % self.epi.static.compartment_count
                    weight = 1 + self.location_cumulative_moves[t, c1, c2]
                    beta = np.zeros(len(self.locations), dtype=np.float64)
                    for infectious_edges in self.epi.static.infectious_edges.values():
                        for comp_index, coef in infectious_edges.compartment_map.items():
                            if infectious_edges.susceptible_comp_index == c1 and comp_index == c2:
                                add = self.location_avg_model_parameters[t, infectious_edges.transmission_rate_index]
                                for p in infectious_edges.numer_param_index_list:
                                    add *= self.location_avg_model_parameters[t, p]
                                for p in infectious_edges.denom_param_index_list:
                                    add = true_divide(add, self.location_avg_model_parameters[t, p])
                                beta += add*coef
                    final_beta[t] += beta*weight
                    sum_weights += weight
                final_beta[t] = true_divide(final_beta[t], sum_weights)
                self.location_basic_R[t] = final_beta[t] * self.window_infectious_duration[t] / min(t+1, R_WINDOW)
        elif self.epi.config["basic_reproductive_number"] == "instantaneous_R_based":
            for t in range(T):
                location_alive = np.sum(get_matrix_without_tag(self.epi.static, self.location_counts[t], DIE_TAG), axis=1)
                location_susceptible = np.sum(get_matrix_with_tag(self.epi.static, self.location_counts[t], SUS_TAG), axis=1)
                self.location_basic_R[t] = true_divide(self.location_instant_R[t]*location_alive, location_susceptible, out=self.location_instant_R[t])

    def compute_case_reproductive_number(self):
        T = self.get_report_length()
        self.location_case_R = np.zeros((T, len(self.locations)), dtype=np.float64)
        if self.epi.config["case_reproductive_number"] == "infectivity_based":
            for t in range(T):
                for u in range(t, T):
                    self.location_case_R[t] += self.location_instant_R[u]*get_infectivity(u-t)

    def compute_responses(self):
        T = self.get_report_length()
        self.responses = []
        for t in range(T):
            location_alive_matrix = get_matrix_without_tag(self.epi.static, self.location_counts[t], DIE_TAG)
            result = []
            for l, location in enumerate(self.locations):
                location_json = {
                    "id":location["id"],
                    "name":location["name"],
                    "parent_id":location["parent_id"],
                    "incidence":np.sum(self.location_incidences[t, l]),
                    "r_instant":self.location_instant_R[t, l],
                    "r_0":self.location_basic_R[t, l],
                    "r_case":self.location_case_R[t, l],
                    "compartments":[],
                    "interventions":[]
                }

                for c in range(self.epi.static.compartment_count):
                    com = {
                        "compartment_name":self.epi.static.get_property_name("compartment", c),
                        "groups":[]
                    }
                    for g in range(self.epi.static.group_count):
                        group = {
                            "group_name": self.epi.static.get_property_name("group", g),
                            "count": self.location_counts[t, c, l, g],
                            "new": self.location_comp_news[t, c, l, g]
                        }
                        com["groups"].append(group)
                    location_json["compartments"].append(com)

                for i in range(self.epi.static.intervention_count):
                    intervention = {
                        "intervention_name":self.epi.static.get_property_name("intervention", i),
                        "cost": self.location_itv_costs[t, i, l] if t == 0 else self.location_itv_costs[t, i, l] - self.location_itv_costs[t-1, i, l],
                        "groups":[],
                        "parameter":[]
                    }
                    for g in range(self.epi.static.group_count):
                        group_influence = self.location_influences[t,i,l,g]
                        group = {
                            "group_name": self.epi.static.get_property_name("group", g),
                            "inf_count": group_influence*location_alive_matrix[l, g],
                            "inf": group_influence
                        }
                        intervention["groups"].append(group)

                    for k, cp in enumerate(self.epi.static.interventions[i].cp_list):
                        parameter = {
                            "name": cp.name,
                            "value": self.location_avg_cpvs[t][l][i][k]
                        }
                        intervention["parameter"].append(parameter)

                    location_json["interventions"].append(intervention)

                result.append(location_json)

            response = {
                "day":t,
                "simulation_id":self.simulation_id,
                "result":result,
            }
            self.responses.append(response)

    def make_responses(self):
        self.compute_history()
        self.compute_incidences()
        self.compute_comp_news()
        self.compute_average_infectious_duration()
        self.compute_location_statistics()
        self.compute_window_statistics()
        self.compute_instantaneous_reproductive_number()
        self.compute_basic_reproductive_number()
        self.compute_case_reproductive_number()
        self.compute_responses()
