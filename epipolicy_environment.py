PERIOD = 7 

# Env
import gym, numpy as np 
from gym import spaces
from epipolicy.core.epidemic import construct_epidemic
from epipolicy.obj.act import construct_act

class EpiEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, session):
        super(EpiEnv, self).__init__()
        self.epi = construct_epidemic(session)
        total_population = np.sum(self.epi.static.default_state.obs.current_comp)
        obs_count = self.epi.static.compartment_count * self.epi.static.locale_count * self.epi.static.group_count
        action_count = 0
        action_param_count =  0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                action_count += 1
                action_param_count += len(itv.cp_list)
        self.act_domain = np.zeros((action_param_count, 2), dtype=np.float64)
        index = 0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    self.act_domain[index, 0] = cp.min_value
                    self.act_domain[index, 1] = cp.max_value
                    index += 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1, shape=(action_count,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=total_population, shape=(obs_count,), dtype=np.float64)

    def step(self, action):
        expanded_action = np.zeros(len(self.act_domain), dtype=np.float64)
        index = 0
        for i in range(len(self.act_domain)):
            if self.act_domain[i, 0] == self.act_domain[i, 1]:
                expanded_action[i] = self.act_domain[i, 0]
            else:
                expanded_action[i] = action[index]
                index += 1

        epi_action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                epi_action.append(construct_act(itv_id, expanded_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)

        total_r = 0
        for i in range(PERIOD):
            state, r, done = self.epi.step(epi_action)
            total_r += r
            if done:
                break
        return state.obs.current_comp.flatten(), total_r, done, dict()

    def reset(self):
        state = self.epi.reset()
        return state.obs.current_comp.flatten()  # reward, done, info can't be included
    def render(self, mode='human'):
        print("Rendering")
        pass
    def close(self):
        pass
