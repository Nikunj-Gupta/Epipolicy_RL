# import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

from scipy.integrate import solve_ivp
import numpy as np

from .reporter import Reporter
from .executor import Executor
from .core_optimize import scipy_fun
from .core_utils import get_leaf_locales, get_tag_count, make_static
from ..obj.static import Static
from ..obj.history import History
from ..obj.observable import get_observable_from_flat
from ..obj.parameter import get_parameter_from_delta, get_gap_parameter
from ..obj.state import State
from ..obj.act import get_normalized_action
from ..parser.regex_parser import RegexParser
from ..utility.singleton import MODES, DEFAULT_CONFIG

def debug(epi):
    if epi.config["debug"]:
        current_comp = epi.get_current_state().obs.current_comp
        print(epi.get_current_time_step(), [(epi.static.get_property_name('compartment', i), np.sum(current_comp[i])) for i in range(epi.static.compartment_count)], flush=True)
        # if epi.get_current_time_step() == epi.T-1:
            # print(epi.get_current_time_step(), [(epi.static.get_property_name('compartment', i), np.sum(current_comp[i])) for i in range(epi.static.compartment_count)], flush=True)

def construct_epidemic(json_input, connection=None, config={}):
    return Epidemic(json_input, connection=connection, config=config)

class Epidemic:

    def __init__(self, json_input, connection=None, config={}):
        if "session" not in json_input:
            json_input = {"session":json_input, "id":0}
        self.session = json_input["session"]
        optimize = None if "optimize" not in self.session else self.session["optimize"]
        self.config = {}
        for key in DEFAULT_CONFIG:
            if key in config:
                self.config[key] = config[key]
            elif optimize is not None and key in optimize["configs"]:
                self.config[key] = optimize["configs"][key]
            else:
                self.config[key] = DEFAULT_CONFIG[key]
        if "optimize" in self.session:
            self.config["possible_optimize"] = True
        if "sensitivity_analysis" in self.session:
            self.config["possible_sensitivity_analysis"] = True

        locations = self.session["locales"]
        compartments = self.session["model"]["compartments"]
        parameters = self.session["model"]["parameters"]
        leaf_locales = get_leaf_locales(locations)

        self.static = Static(
            locale_count=len(leaf_locales),
            compartment_count=len(compartments),
            parameter_count=len(parameters),
            group_count=max(len(self.session["groups"]), 1),
            facility_count=max(len(self.session["facilities"]), 1),
            intervention_count=len(self.session["interventions"])+len(self.session["costs"]),
            mode_count=len(MODES),
            compartment_tag_count=get_tag_count(compartments),
            parameter_tag_count=get_tag_count(parameters)
        )
        self.parser = RegexParser(self.static)
        make_static(self.static, self.parser, self.session)
        self.executor = Executor(self, leaf_locales)
        self.history = History()
        self.reporter = Reporter(self, json_input["id"], connection, locations)
        self.make_setting()

    def make_setting(self):
        if self.config["detached_initial_state"]:
            initial_state, _ = self.get_next_state(self.static.default_state, self.static.schedule.action_list[0])
            self.static.schedule.initial_state = State(0, self.static.default_state.obs, initial_state.unobs)

    def get_next_state(self, state, action):
        delta_parameter = self.executor.execute(state, action)
        dest_parameter = get_parameter_from_delta(self.static, delta_parameter)
        gap_parameter = get_gap_parameter(state.unobs, dest_parameter)

        res = solve_ivp(scipy_fun, (0.0, 1.0), state.obs.flatten(), method=self.config["solver"], args=(self.static, state.unobs, gap_parameter))
        next_obs = get_observable_from_flat(self.static, res.y[:, -1])
        return State(state.t+1, next_obs, dest_parameter), delta_parameter

    def get_current_state(self):
        if len(self.history.states) > 0:
            return self.history.states[-1]
        return self.static.schedule.initial_state

    def get_previous_state(self):
        if len(self.history.states) <= 1:
            return self.static.schedule.initial_state
        return self.history.states[-2]

    def get_current_time_step(self):
        return len(self.history.states)

    def reset(self):
        self.history = History()
        return self.get_current_state()

    def step(self, action):
        current_state = self.get_current_state()
        normalized_action = get_normalized_action(self.static, action)
        next_state, delta_parameter = self.get_next_state(current_state, normalized_action)
        self.history.states.append(next_state)
        self.history.delta_parameters.append(delta_parameter)
        self.history.actions.append(normalized_action)
        reward = -np.sum(next_state.obs.cumulative_cost - current_state.obs.cumulative_cost)
        return next_state, reward, self.get_current_time_step() >= self.static.schedule.horizon

    def run(self, T=None):
        self.reporter.start()
        if T is None:
            self.T = self.static.schedule.horizon
        else:
            self.T = T
        self.reset()
        for t in range(self.T):
            # print("DAYYYYYYYYYY", t)
            action = self.static.schedule.action_list[t]
            self.step(action)
            # if t == 2:
            #     break
            debug(self)
        self.reporter.report()
        self.reporter.end()
        return self.reporter.responses
