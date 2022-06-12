import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba import njit

observable_spec = [
    ("current_comp", nb.float64[:,:,:]), #cxlxg
    ("cumulative_gain", nb.float64[:,:,:]), #cxlxg
    ("cumulative_lost", nb.float64[:,:,:]), #cxlxg
    ("cumulative_move", nb.float64[:,:,:,:]), #cxcxlxg
    ("cumulative_cost", nb.float64[:,:]) #ixl
]

@jitclass(observable_spec)
class Observable:

    def __init__(self, current_comp, cumulative_gain, cumulative_lost, cumulative_move, cumulative_cost):
        self.current_comp = current_comp
        self.cumulative_gain = cumulative_gain
        self.cumulative_lost = cumulative_lost
        self.cumulative_move = cumulative_move
        self.cumulative_cost = cumulative_cost

    def flatten(self):
        return np.concatenate((self.current_comp.copy().reshape(-1), self.cumulative_gain.copy().reshape(-1), self.cumulative_lost.copy().reshape(-1), self.cumulative_move.copy().reshape(-1), self.cumulative_cost.copy().reshape(-1)))

ObservableType = Observable.class_type.instance_type

@njit
def get_observable_from_flat(static, flat):
    clg_size = static.compartment_count * static.locale_count * static.group_count
    cclg_size = (3 + static.compartment_count) * clg_size
    current_comp = flat[:clg_size].reshape((static.compartment_count, static.locale_count, static.group_count))
    cumulative_gain = flat[clg_size:2*clg_size].reshape((static.compartment_count, static.locale_count, static.group_count))
    cumulative_lost = flat[2*clg_size:3*clg_size].reshape((static.compartment_count, static.locale_count, static.group_count))
    cumulative_move = flat[3*clg_size:cclg_size].reshape((static.compartment_count, static.compartment_count, static.locale_count, static.group_count))
    cumulative_cost = flat[cclg_size:].reshape((static.intervention_count, static.locale_count))
    return Observable(current_comp, cumulative_gain, cumulative_lost, cumulative_move, cumulative_cost)

@njit
def get_initial_observable(static):
    initial_comp = np.zeros((static.compartment_count, static.locale_count, static.group_count), dtype=np.float64)
    cumulative_gain = np.zeros((static.compartment_count, static.locale_count, static.group_count), dtype=np.float64)
    cumulative_lost = np.zeros((static.compartment_count, static.locale_count, static.group_count), dtype=np.float64)
    cumulative_move = np.zeros((static.compartment_count, static.compartment_count, static.locale_count, static.group_count), dtype=np.float64)
    cumulative_cost = np.zeros((static.intervention_count, static.locale_count), dtype=np.float64)
    return Observable(initial_comp, cumulative_gain, cumulative_lost, cumulative_move, cumulative_cost)

@njit
def copy_observable(obs):
    return Observable(obs.current_comp.copy(), obs.cumulative_gain.copy(), obs.cumulative_lost.copy(), obs.cumulative_move.copy(), obs.cumulative_cost.copy())
