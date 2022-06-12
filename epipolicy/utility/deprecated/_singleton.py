from numba import int64, int32, float64, boolean
from numba.types import DictType, string
import numpy as np

nbFloat = float64
nbInt = int32
nbHashedInt = int64
nbBool = boolean
npFloat = np.float64
npInt = np.int32
npLong = np.int64
npBool = np.bool

#RunMode
RUN_NORMAL = 0
RUN_MCTS = 1
RUN_SA = 2

#DebugMode
DEBUG = 0
WARMUP = 1
NORMAL = 2

EPSILON = 1e-15

WITHIN_RATIO_FACTILITY = 0.7
ALIVE_COMPARTMENT = 'N'
ALL = '*'
NOT = '~'
OR = ','
AND = '^'
AGE_GROUP = 5
MODES = ['airport', 'border']
R_WINDOW = 7

# Temporary customize
# FACILITIES = ['Household', 'School', 'Workplace', 'Community']
# GROUPS = ['Children', 'Adults', 'Seniors']
# PARAMETERS = ['psym', 'psev', 'pcri', 'pdeath']
# PARAMETER_VALUES = np.array([
#     [0.52272, 0.65682, 0.81286],
#     [0.0002, 0.05242, 0.15892],
#     [0.7, 0.05577, 0.28083],
#     [0.57143, 0.69792, 0.8258]
# ])

SUS_TAG = 'susceptible'
INF_TAG = 'infectious'
DIE_TAG = 'death'
# INI_TAG = 'initial-population'
HOS_TAG = 'hospitalized'
TRANS_TAG = 'transmission'

#Numba setting
NO_PYTHON = True
NO_GIL = True
FAST_MATH = True

#MCTS setting
MCTS_C = 0.01
MCTS_INTERVAL = 7
MAX_SIBLING_PAIRS = 4
MAX_CP_PER_ACTION = 10
MAX_NODE = 10000
MAX_RAM_PERCENTAGE = 0.8
I_BUDGET = 1000
T_BUDGET = -1
TREE_TERMINAL = 35
BIAS_PROBABILITY = 0.5
UNBIAS_PROBABILITY = 0.25

UNCHANGE = 0
POSITIVE_CHANGE = 1
NEGATIVE_CHANGE = -1
SMART_DISTANCE = 2
RANDOM_DISTANCE = 1
RANDOM_APPLY_PROBABILITY = 0.25

LONG_PRECISION = 5
ULONG_PRECISION = 1
LONG_SIZE = 8
INT_SIZE = 4

ULONG_MAX = (1<<64) - 1
MAX_UCOST = (1<<64) - 1
MAX_COST = MAX_UCOST / (10 ** ULONG_PRECISION)

DEBUG_INTERVAL = 1
DEBUG_METHOD = "sp_RK23"

PRIME = 1000000007
