import numpy as np
from ..obj.act import construct_act
from ..utility.utils import true_divide

def normalize(x, vmin, vmax, l=0, r=1):
    zeros = np.zeros_like(x)
    return true_divide(x-vmin, vmax-vmin, zeros)*(r-l)+l

def denormalize(x, vmin, vmax, l=0, r=1):
    return (x-l)/(r-l)*(vmax-vmin)+vmin

class Vectorizer:
    def __init__(self, obs_domain, act_domain):
        self.obs_domain = obs_domain
        self.act_domain = act_domain
        self.mask = np.ones(len(act_domain), dtype=bool)
        for i in range(len(act_domain)):
            if act_domain[i][0] == act_domain[i][1]:
                self.mask[i] = False
        self.reduced_act_domain = act_domain[self.mask,...]

    def featurize_state(self, state):
        raise NotImplementedError

    def defeaturize_action(self, featurized_action):
        raise NotImplementedError

    def vectorize_state(self, state):
        return normalize(self.featurize_state(state), self.obs_domain[:,0], self.obs_domain[:,1])

    def defeaturize_reduced_action(self, reduced_action):
        reduced_action = denormalize(reduced_action, self.reduced_act_domain[:,0], self.reduced_act_domain[:,1])
        vectorized_action = np.zeros(len(self.act_domain), dtype=np.float64)
        index = 0
        for i in range(len(self.act_domain)):
            if self.mask[i]:
                vectorized_action[i] = reduced_action[index]
                index += 1
            else:
                vectorized_action[i] = self.act_domain[i][0]
        return self.defeaturize_action(vectorized_action)

class EpiVectorizer(Vectorizer):
    def __init__(self, epi):
        self.epi = epi
        total_population = np.sum(epi.static.default_state.obs.current_comp)
        comp_count = epi.static.compartment_count * epi.static.locale_count * epi.static.group_count
        obs_count = comp_count + epi.static.intervention_count * epi.static.locale_count + 1
        obs_domain = np.zeros((obs_count, 2), dtype=np.float64)
        for i in range(obs_count):
            if i < comp_count:
                obs_domain[i,1] = total_population
            elif i < obs_count - 1:
                obs_domain[i,1] = epi.config["max_cost"]
            else:
                obs_domain[i,1] = epi.static.schedule.horizon
        action_count = 0
        for itv in epi.static.interventions:
            if not itv.is_cost:
                action_count += len(itv.cp_list)
        act_domain = np.zeros((action_count, 2), dtype=np.float64)
        index = 0
        for itv in epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    act_domain[index, 0] = cp.min_value
                    act_domain[index, 1] = cp.max_value
                    index += 1
        super().__init__(obs_domain, act_domain)

    def featurize_state(self, state):
        return np.concatenate((state.obs.current_comp.reshape(-1), state.obs.cumulative_cost.reshape(-1), [state.t]))

    def defeaturize_action(self, featurized_action):
        action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                action.append(construct_act(itv_id, featurized_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)
        return action

class EpiVectorizerS1(Vectorizer):
    def __init__(self, epi):
        self.epi = epi
        total_population = np.sum(epi.static.default_state.obs.current_comp)
        comp_count = epi.static.compartment_count * epi.static.locale_count * epi.static.group_count
        obs_count = comp_count
        obs_domain = np.zeros((obs_count, 2), dtype=np.float64)
        for i in range(obs_count):
            obs_domain[i,1] = total_population
        action_count = 0
        for itv in epi.static.interventions:
            if not itv.is_cost:
                action_count += len(itv.cp_list)
        act_domain = np.zeros((action_count, 2), dtype=np.float64)
        index = 0
        for itv in epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    act_domain[index, 0] = cp.min_value
                    act_domain[index, 1] = cp.max_value
                    index += 1
        super().__init__(obs_domain, act_domain)

    def featurize_state(self, state):
        return state.obs.current_comp.reshape(-1)

    def defeaturize_action(self, featurized_action):
        action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                action.append(construct_act(itv_id, featurized_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)
        return action

class EpiVectorizerS2(Vectorizer):
    def __init__(self, epi, env):
        self.epi = epi
        self.env = env
        total_population = np.sum(epi.static.default_state.obs.current_comp)
        comp_count = epi.static.compartment_count * epi.static.locale_count * epi.static.group_count
        obs_count = comp_count * 2
        obs_domain = np.zeros((obs_count, 2), dtype=np.float64)
        for i in range(obs_count):
            obs_domain[i,1] = total_population
        action_count = 0
        for itv in epi.static.interventions:
            if not itv.is_cost:
                action_count += len(itv.cp_list)
        act_domain = np.zeros((action_count, 2), dtype=np.float64)
        index = 0
        for itv in epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    act_domain[index, 0] = cp.min_value
                    act_domain[index, 1] = cp.max_value
                    index += 1
        super().__init__(obs_domain, act_domain)

    def featurize_state(self, state):
        if self.env.prev_state is None:
            delta = state.obs.current_comp
        else:
            delta = state.obs.current_comp - self.env.prev_state.obs.current_comp
        return np.concatenate((state.obs.current_comp.reshape(-1), delta.reshape(-1)))

    def defeaturize_action(self, featurized_action):
        action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                action.append(construct_act(itv_id, featurized_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)
        return action
