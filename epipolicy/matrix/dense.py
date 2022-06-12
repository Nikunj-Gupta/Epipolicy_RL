import numpy as np
from numba import njit

from ..utility.singleton import FLOAT_EPSILON

@njit
def get_normalized_facility_timespent(facility_timespent):
    locale_count, facility_count, group_count = facility_timespent.shape
    normalized_facility_timespent = np.zeros((locale_count, facility_count, group_count), dtype=np.float32)
    sum_on_facility = np.sum(facility_timespent, axis=1)
    for l in range(locale_count):
        for g in range(group_count):
            if sum_on_facility[l, g] > FLOAT_EPSILON:
                for f in range(facility_count):
                    normalized_facility_timespent[l, f, g] = facility_timespent[l, f, g] / sum_on_facility[l, g]
            else:
                # By default if sum is 0 or negative, the first facility gets all timespent
                normalized_facility_timespent[l, 0, g] = 1.0
    return normalized_facility_timespent

@njit
def get_normalized_facility_interaction(facility_interaction):
    locale_count, facility_count, group_count = facility_interaction.shape[:3]
    normalized_facility_interaction = facility_interaction.copy()
    sum_on_group = np.sum(facility_interaction, axis=3)
    for l in range(locale_count):
        for f in range(facility_count):
            for g1 in range(group_count):
                if sum_on_group[l, f, g1] > 1.0 or sum_on_group[l, f, g1] < 0:
                    for g2 in range(group_count):
                        normalized_facility_interaction[l, f, g1, g2] = facility_interaction[l, f, g1, g2] / sum_on_group[l, f, g1]
    return normalized_facility_interaction
