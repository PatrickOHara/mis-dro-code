
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
    dro_bas_mmd = "dotted"
    empirical_mmd = "dotted"


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

class AlgorithmColor(StrEnum):
    """Colors of algorithm lines"""
    dro_bas_mmd = "#f781bf"
    kl_bdro = "#FF800E"
    kl_dro_bas = "#006BA4"
    empirical_mmd = "#999999"



class AlgorithmName(StrEnum):
    """Consistent algorithm line styles"""

    bdro_grid_search = "BDRO grid search"
    dro_bas_mmd = "RoBAS"
    empirical_mmd = "Empirical MMD"
    kl_bdro = "BDRO"
    kl_dro_bas = "DRO-BAS"




class NiceNameDGP(StrEnum):
    contaminated_exp = "Contaminated Exponential DGP"
    exponential = "Exponential DGP"
    normal = "Normal DGP"
    multivariate_normal = "5D Normal DGP"
    truncated_normal = "Truncated Normal DGP"
    DowJones = "DowJones"



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
    ("kl_bdro", "bayes"): '^',
    ("kl_bdro", "npl_mmd"): "x",
    ("dro_bas_mmd", "npl_mmd"): "*",
    ("empirical_mmd", "empirical"): "+",
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
    **kwargs,
) -> None:
    """Plot mean-variance trade-off."""
    out_of_sample_var = df["out_of_sample_var"].values
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
                    epsilon,
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

    gb = results_df.groupby(by=gb_cols)
    agg_df = gb.agg(
        out_of_sample_mean = pd.NamedAgg(column="out_of_sample_cost", aggfunc=np.mean),
        out_of_sample_var = pd.NamedAgg(column="out_of_sample_cost", aggfunc=np.var),
        mean_solve_time = pd.NamedAgg(column="solve_time", aggfunc=np.mean),
        std_solve_time = pd.NamedAgg(column="solve_time", aggfunc=np.std),
    )
    return agg_df
