"""Constants for e.g. number of observations or samples"""

# constants for DGPs
CONTAMINATION_LEVEL = 0.2  # ratio for contamination dataset
DGP_NORMAL_KNOWN_VARIANCE_STD = 5.0 # standard deviation of 1D normal with known variance
IN_SAMPLE_TIME_WINDOW = 52  # number of weeks in training period for portfolio problem
OUT_OF_SAMPLE_TIME_WINDOW = 12  # number of weeks in out-of-sample period for portfolio

# experiment constants
NUM_OBSERVATIONS = 20  # in-sample 'training' observations
NUM_POSTERIOR_SAMPLES = 100  # theta samples from posterior
NUM_TEST_OBSERVATIONS = 50  # out-of-sample 'test' observations
NUM_LIKELIHOOD_SAMPLES = 100  # xi samples from likelihood
NUM_REPLICATIONS = 200  # num times to repeat for loop
BAS_NUM_REPLICATIONS = 500
NUM_CERTIFY = 200 # num certufying points for discretisation of KDRO problem constraints
MAX_PARAMS_OOM = 1000   # if the number of params of a cvxpy exceeds this number, we might go out-of-memory
BAS_TOTAL_MODEL_SAMPLES = [25, 100, 900]

# experiment constants for RoBAS
ROBAS_NEWSVENDOR_NUM_REPLICATIONS = 100
NPL_ETA = 0.1

BAS_DRO_EPSILON_SET = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.5,
    2,
    2.5,
    3,
]

SMALL_BAS_DRO_EPSILON_SET = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]


PORTFOLIO_EPSILON_SET = [
    0.00001,
    0.00002,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
]

ROBAS_DRO_EPSILON_SET = [
    0.0001,
    # 0.0005,
    0.001,
    # 0.003,
    0.005,
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    # 0.25,
    0.3,
    0.5,
    1.0
]

def upper_triangular_size(dim: int) -> int:
    """Includes the diagonal!"""
    return int(dim * (dim-1) / 2 + dim)
