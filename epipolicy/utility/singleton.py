FLOAT_EPSILON = 1e-15

WITHIN_RATIO_FACTILITY = 0.7
ALIVE_COMPARTMENT = 'N'
ALL = '*'
NOT = '~'
OR = ','
AND = '^'
AGE_GROUP = 5
MODES = ["airport", "border"]
R_WINDOW = 7

SUS_TAG = "susceptible"
INF_TAG = "infectious"
DIE_TAG = "death"
HOS_TAG = "hospitalized"
TRANS_TAG = "transmission"

DEFAULT_CONFIG = {
    # Epi's config
    "detached_initial_state": True,
    "solver": "RK45",
    "debug": False,
    "possible_optimize": False,
    "possible_sensitivity_analysis": False,

    # Optimizer's config
    "max_cost": 1000000000000,

    # Reporter's config
    "average_infectious_duration": "instantaneous_based", # or "beta_window_based"
    "instantaneous_reproductive_number": "infectious_duration_based", # or "infectivity_based"
    "basic_reproductive_number": "beta_based", # or "instantaneous_R_based"
    "case_reproductive_number": "infectivity_based",
}
