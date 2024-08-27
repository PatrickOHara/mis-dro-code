"""Constants for e.g. number of observations or samples"""

CONTAMINATION_LEVEL = 0.2  # ratio for contamination dataset
NUM_OBSERVATIONS = 20  # in-sample 'training' observations
NUM_POSTERIOR_SAMPLES = 100  # theta samples from posterior
NUM_TEST_OBSERVATIONS = 50  # out-of-sample 'test' observations
NUM_LIKELIHOOD_SAMPLES = 100  # xi samples from likelihood
NUM_REPLICATIONS = 200  # num times to repeat for loop
MAX_PARAMS_OOM = 1000   # if the number of params of a cvxpy exceeds this number, we might go out-of-memory
