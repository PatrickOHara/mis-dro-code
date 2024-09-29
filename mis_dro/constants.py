"""Constants for e.g. number of observations or samples"""

CONTAMINATION_LEVEL = 0.2  # ratio for contamination dataset
NUM_OBSERVATIONS = 400  # in-sample 'training' observations
NUM_POSTERIOR_SAMPLES = 100  # theta samples from posterior
NUM_TEST_OBSERVATIONS = 50  # out-of-sample 'test' observations
NUM_LIKELIHOOD_SAMPLES = 100  # xi samples from likelihood
NUM_REPLICATIONS = 200  # num times to repeat for loop
NUM_CERTIFY = 200 # num certufying points for discretisation of KDRO problem constraints
MAX_PARAMS_OOM = 1000   # if the number of params of a cvxpy exceeds this number, we might go out-of-memory

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
