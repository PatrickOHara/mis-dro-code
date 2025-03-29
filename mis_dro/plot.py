
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
    wasserstein_empirical = "dotted"

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
    wasserstein_empirical = "#a65628"
    empirical_mmd = "#999999"
    kl_pp = "#4daf4a"



class AlgorithmName(StrEnum):
    """Consistent algorithm line styles"""

    bdro_grid_search = "BDRO grid search"
    dro_bas_mmd = "DRO-RoBAS"
    empirical_mmd = "Empirical MMD"
    kl_bdro = "BDRO"
    kl_dro_bas = "DRO-BAS$_{PE}$"
    kl_pp = "DRO-BAS$_{PP}$"
    kl_empirical = "Empirical KL"
    wasserstein_empirical = "Empirical Wasserstein"




class NiceNameDGP(StrEnum):
    contaminated_exp = "Contaminated Exp"
    contaminated_exp_large_outliers = "Large Outliers Exp"
    contaminated_exp_small_outliers = "Small Outliers Exp"
    exponential = "Exponential DGP"
    normal = "Normal DGP"
    multivariate_normal = "5D Normal DGP"
    truncated_normal = "Truncated Normal DGP"
    DowJones = "DowJones"
    bimodal_univariate_gaussian = "Bimodal 1D Normal"
    bimodal_multivariate_gaussian = "Bimodal 5D Normal"
    contaminated_normal = "Contaminated Normal"
    portfolio_contaminated_multivariate_normal = "Contaminated 5D Normal"



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
    ("wasserstein_empirical", "empirical"): "+",
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
    add_log_partition_function: bool = False,
    minimise: bool = True,
    add_end_epsilons: bool = True,
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
    axis.plot(out_of_sample_var, out_of_sample_mean, linestyle=kwargs["linestyle"], markersize=0, color=kwargs["color"], label='_nolegend_', alpha=alpha, lw=2)

    # label the points with epsilon values
    if is_labelled:
        if add_end_epsilons:
            special_epsilons += [epsilon_list[0], epsilon_list[-1]]
        for i, epsilon in enumerate(epsilon_list):
            if epsilon in special_epsilons:
                # if line is blue then put text on bottom left
                if kwargs["color"] in (AlgorithmColor.kl_dro_bas, AlgorithmColor.kl_empirical):
                    ha = "right"
                    va = "top" if minimise else "bottom"
                    local_offset = -offset

                # else if line is black then put text on top right
                elif kwargs["color"] == AlgorithmColor.kl_bdro:
                    ha = "left"
                    va = "bottom" if minimise else "top"
                    local_offset = offset
                else:
                    ha = "left"
                    va = "bottom" if minimise else "top"
                    local_offset = offset

                if add_log_partition_function:
                    epsilon_label = "$G +" + str(np.round(epsilon, 5)) + "$"
                else:
                    epsilon_label = str(np.round(epsilon, 5))
                axis.text(
                    out_of_sample_var[i] + local_offset,
                    out_of_sample_mean[i] + local_offset,
                    epsilon_label,
                    ha=ha,
                    va=va,
                    color=kwargs["color"],
                )