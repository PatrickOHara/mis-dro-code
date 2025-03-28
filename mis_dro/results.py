
from pathlib import Path
import numpy as np
import pandas as pd

def get_result_df_list(experiment_dir: Path, uuid_list: list[str]):
    result_list = []
    failed_uuid_list = []
    missing_uuid_list = []
    for uuid in uuid_list:
        if (experiment_dir / f"{uuid}.csv").exists():
            try:
                result_list.append(pd.read_csv(
                    experiment_dir / f"{uuid}.csv", index_col=["uuid", "replication"]
                ))
            except pd.errors.ParserError:
                failed_uuid_list.append(uuid)
        else:
            missing_uuid_list.append(uuid)

    print("The following UUIDs did not have a CSV file:")
    print(missing_uuid_list)
    print()
    print("The following UUIDs failed due to a pandas.errors.ParserError:")
    print(failed_uuid_list)
    return result_list

def preprocess_results_df(results_df: pd.DataFrame, dgp: str, dataset: str = "newsvendor"):
    """Filter results, process columns, and create new columns"""
    assert len(results_df["num_test_observations"].unique()) == 1
    assert len(results_df.loc[(results_df["dgp"] == dgp)]["dim"].unique()) == 1 
    num_test_observations = results_df["num_test_observations"].unique()[0]
    
    processed_df = results_df.copy()
    dim = processed_df.loc[(processed_df["dgp"] == dgp)]["dim"].unique()
    # filter by the DGP and cases where the the log partition function is feasible for epsilon
    processed_df = processed_df.loc[processed_df["dgp"] == dgp]
    if dataset != "portfolio":
        if "use_cv_epsilon" not in processed_df.columns:
            processed_df["use_cv_epsilon"] = False
        processed_df = processed_df.loc[(processed_df["log_partition_constant"] < processed_df["epsilon"]) | (processed_df["use_cv_epsilon"])]

    # get useful stats such as the number of samples and total time spent sampling
    processed_df["num_total_samples"] = processed_df["num_posterior_samples"] * processed_df["num_likelihood_samples"]
    processed_df["sample_time"] = processed_df["likelihood_time"] + processed_df["posterior_time"]

    # convert strings into list of floats where necessary
    processed_df["out_of_sample_cost"] = processed_df["out_of_sample_cost"].map(lambda x: convert_str_to_float_list(x, num_test_observations))
    processed_df["solution"] = processed_df["solution"].map(lambda x: convert_str_to_float_list(x, dim))

    # calculate the in-group mean and in-group variance for each replication
    processed_df["in_group_mean"] = processed_df["out_of_sample_cost"].map(np.mean)
    processed_df["in_group_var"] = processed_df["out_of_sample_cost"].map(lambda x: np.var(x, ddof=1))

    # return preprocessed dataframe
    return processed_df

def get_agg_df(results_df: pd.DataFrame, gb_cols: list[str]):
    """Groupby the given columns then apply summary statistics for each group"""
    assert len(results_df["num_replications"].unique()) == 1
    assert len(results_df["num_test_observations"].unique()) == 1
    num_replications = results_df["num_replications"].unique()[0]
    num_test_observations = results_df["num_test_observations"].unique()[0]
    gb = results_df.groupby(by=gb_cols)
    agg_df = gb.agg(
        out_of_sample_mean = pd.NamedAgg(column="out_of_sample_cost", aggfunc=lambda x: np.mean(np.concatenate(x.values))),
        out_of_sample_var = pd.NamedAgg(column="out_of_sample_cost", aggfunc=lambda x: np.var(np.concatenate(x.values), ddof=1)),
        sum_of_in_group_var = pd.NamedAgg(column="in_group_var", aggfunc=lambda x: float(num_test_observations - 1) / float(num_replications * num_test_observations - 1) * np.sum(x.values)),
        var_of_in_group_mean = pd.NamedAgg(column="in_group_mean", aggfunc=lambda x: float(num_test_observations * (num_replications - 1)) / float(num_replications * num_test_observations - 1) * np.var(x, ddof=1)),
        mean_solve_time = pd.NamedAgg(column="solve_time", aggfunc=np.mean),
        std_solve_time = pd.NamedAgg(column="solve_time", aggfunc=np.std),
        mean_sample_time = pd.NamedAgg(column="sample_time", aggfunc=np.mean),
        std_sample_time = pd.NamedAgg(column="sample_time", aggfunc=np.std),
    )
    return agg_df

def convert_str_to_float_list(str_list: str, list_len: int) -> list[float]:
    if str_list == "[]":
        return [np.nan for _ in range(list_len)]
    else:
        return [float(x) for x in str_list.strip('[]').split(',')]


def is_minimise_pareto_front(out_of_sample_var, out_of_sample_mean):
    """Returns true if the point lies on the Pareto front of a minimisation problem"""
    assert out_of_sample_var.shape == out_of_sample_mean.shape
    pareto = []
    for i in range(out_of_sample_var.shape[0]):
        point_is_pareto = True
        for j in range(out_of_sample_var.shape[0]):
            if out_of_sample_var[j] < out_of_sample_var[i] and out_of_sample_mean[j] < out_of_sample_mean[i]:
                point_is_pareto = False
                break
        pareto.append(point_is_pareto)
    return pareto

def is_maximise_pareto_front(out_of_sample_var, out_of_sample_mean):
    """Returns true if the point lies on the Pareto front of a maximisation problem"""
    assert out_of_sample_var.shape == out_of_sample_mean.shape
    pareto = []
    for i in range(out_of_sample_var.shape[0]):
        point_is_pareto = True
        for j in range(out_of_sample_var.shape[0]):
            if out_of_sample_var[j] < out_of_sample_var[i] and out_of_sample_mean[j] > out_of_sample_mean[i]:
                point_is_pareto = False
                break
        pareto.append(point_is_pareto)
    return pareto
