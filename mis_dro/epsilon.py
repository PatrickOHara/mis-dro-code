import numpy as np

def get_num_observations_in_train_split(n_splits: int, split_idx: int, n_observations: int):
    ratio = float(n_observations) / float(n_splits)
    num_splits_with_less_than_max_test_size = n_splits * np.ceil(ratio) - n_observations
    if split_idx < n_splits - num_splits_with_less_than_max_test_size:
        return int(n_observations - np.ceil(ratio))
    else:
        return int(n_observations - np.floor(ratio))