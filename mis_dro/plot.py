
from enum import StrEnum
from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AlgorithmLineStyle(StrEnum):
    """Consistent algorithm line styles"""

    bdro_grid_search = "dashed"
    kl_bdro = "dotted"
    kl_dro_bas = "dotted"
    kl_empirical = "dotted"
    dro_bas_mmd = "dotted"
    empirical_mmd = "dotted"
    kl_pp = "dotted"

# Color blind palette from https://gist.github.com/thriveth/8560036
CB_color_cycle = [
    '#377eb8', # blue
    '#ff7f00', # orange
    '#4daf4a', # green
    '#f781bf', # pink
    '#a65628', # brown
    '#984ea3', # purple
    '#999999', # gray
    '#e41a1c', # red
    '#dede00', # yellow
]


class AlgorithmColor(StrEnum):
    """Colors of algorithm lines"""
    dro_bas_mmd = "#377eb8"   #"#f781bf"
    kl_bdro = "#ff7f00"
    kl_dro_bas = "#984ea3"
    kl_empirical = "#999999"
    empirical_mmd = "#999999"
    kl_pp = "#4daf4a"



class AlgorithmName(StrEnum):
    """Consistent algorithm line styles"""

    bdro_grid_search = "BDRO grid search"
    dro_bas_mmd = "RoBAS"
    empirical_mmd = "Empirical MMD"
    kl_bdro = "BDRO"
    kl_dro_bas = "DRO-BAS$_{PE}$"
    kl_pp = "DRO-BAS$_{PP}$"
    kl_empirical = "Empirical KL"




class NiceNameDGP(StrEnum):
    contaminated_exp = "Contaminated Exp DGP"
    contaminated_exp_large_outliers = "Large Outliers Exp"
    contaminated_exp_small_outliers = "Small Outliers Exp"
    exponential = "Exponential DGP"
    normal = "Normal DGP"
    multivariate_normal = "5D Normal DGP"
    truncated_normal = "Truncated Normal DGP"
    DowJones = "DowJones"
    bimodal_univariate_gaussian = "Bimodal Normal DGP"
    bimodal_multivariate_gaussian = "Bimodal 5D Normal DGP"
    contaminated_normal = "Contaminated Normal DGP"
    portfolio_contaminated_multivariate_normal = "Contaminated Normal DGP"



class PosteriorPrettyName(StrEnum):
    """Nice looking names for posteriors"""

    gamma = "gamma"
    normal_gamma = "normal-gamma"


class InferencePrettyName(StrEnum):
    """Nice looking names for inference"""

    bayes = "Bayes"
    empirical = ""
    npl_wlb = "NPL-WLB"
    npl_mmd = "NPL-MMD"


ALGORITHM_INFERENCE_MARKERS = {
    ("kl_dro_bas", "bayes"): 'o',
    ("kl_pp", "bayes"): '*',
    ("kl_bdro", "bayes"): '^',
    ("kl_bdro", "npl_mmd"): "x",
    ("dro_bas_mmd", "npl_mmd"): "*",
    ("empirical_mmd", "empirical"): "+",
    ("kl_empirical", "empirical"): "x",
}


def algorithm_inference_style(algorithm: str, inference: str, label_inference: bool = True) -> dict[str, str]:
    """Get matplotlib style for an algorithm"""
    # FIXME need to include inference for linestyle and color
    label = AlgorithmName[algorithm]
    if label_inference:
        label += " " + InferencePrettyName[inference]
    return {
        "marker": ALGORITHM_INFERENCE_MARKERS[(algorithm, inference)],
        "linestyle": AlgorithmLineStyle[algorithm],
        "label": label,
        "color": AlgorithmColor[algorithm],
    }





def mean_variance_plot(
    axis: mpl.axis.Axis,
    df: pd.DataFrame,
    alpha: float = 1.0,
    is_labelled: bool = True,
    offset: float = 1.0,
    special_epsilons: list[float] = [0.1, 0.5],
    var_col: float = "out_of_sample_var",
    add_log_partition_function: float = 0.0,
    **kwargs,
) -> None:
    """Plot mean-variance trade-off."""
    out_of_sample_var = df[var_col].values
    out_of_sample_mean = df["out_of_sample_mean"].values
    epsilon_list = df.index.get_level_values("epsilon").values
    if "is_pareto_front" in df.columns:
        fillstyles = ["full" if is_pareto else "none" for is_pareto in df["is_pareto_front"].values]
    else:
        fillstyles = len(df) * ["full"]
    label=kwargs.pop("label")
    # plot the markers
    for i, var in enumerate(out_of_sample_var):
        if i == 0 and is_labelled:
            local_label = label
        else:
            local_label='_nolegend_'
        axis.plot(var, out_of_sample_mean[i], markersize=4, fillstyle=fillstyles[i], **kwargs, label=local_label, alpha=alpha)

    # plot the lines
    axis.plot(out_of_sample_var, out_of_sample_mean, linestyle=kwargs["linestyle"], markersize=0, color=kwargs["color"], label='_nolegend_', alpha=alpha)

    # label the points with epsilon values
    if is_labelled:
        for i, epsilon in enumerate(epsilon_list):
            if i == 0 or i == len(epsilon_list) - 1 or epsilon in special_epsilons:
                # if line is blue then put text on bottom left
                if kwargs["color"] == AlgorithmColor.kl_dro_bas:
                    ha = "right"
                    va = "top"
                    local_offset = -offset

                # else if line is black then put text on top right
                elif kwargs["color"] == AlgorithmColor.kl_bdro:
                    ha = "left"
                    va = "bottom"
                    local_offset = offset
                else:
                    ha = "left"
                    va = "bottom"
                    local_offset = offset

                axis.text(
                    out_of_sample_var[i] + local_offset,
                    out_of_sample_mean[i] + local_offset,
                    np.round(epsilon + add_log_partition_function, 5),
                    ha=ha,
                    va=va,
                    color=kwargs["color"],
                )

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
        processed_df = processed_df.loc[processed_df["log_partition_constant"] < processed_df["epsilon"]]

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
