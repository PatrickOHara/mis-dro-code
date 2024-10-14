from enum import StrEnum
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


class AlgorithmMarkerStyle(StrEnum):
    """Consistent algorithm marker styles"""

    # bdro_grid_search = "*"
    dro_bas_mmd = "*"
    empirical_mmd = "v"
    kl_bdro = "^"
    kl_dro_bas = "o"

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


class InferenceLineStyle(StrEnum):
    """Consistent line styles"""

    bayes = "solid"
    npl_wlb = "dashed"
    npl_mmd = "dotted"


class NiceNameDGP(StrEnum):
    contaminated_exp = "Contaminated Exponential DGP"
    exponential = "Exponential DGP"
    normal = "Normal DGP"
    multivariate_normal = "5D Normal DGP"
    truncated_normal = "Truncated Normal DGP"
    DowJones = "DowJones"

class InferenceMarker(StrEnum):
    """Consistent marker styles"""

    bayes = "o"
    npl_wlb = "x"
    npl_mmd = "v"


class PosteriorPrettyName(StrEnum):
    """Nice looking names for posteriors"""

    gamma = "gamma"
    normal_gamma = "normal-gamma"


class InferencePrettyName(StrEnum):
    """Nice looking names for inference"""

    bayes = "Bayes"
    npl_wlb = "NPL-WLB"
    npl_mmd = "NPL-MMD"

class InferenceColor(StrEnum):
    bayes = "blue"
    npl_wlb = "orange"
    npl_mmd = "green"


def inference_style(inference: str) -> dict[str, str]:
    """Get matplotlib style for an inference"""
    return {
        "marker": InferenceMarker[inference].value,
        "linestyle": InferenceLineStyle[inference].value,
        "label": InferencePrettyName[inference].value,
        "color": InferenceColor[inference].value,
    }


def algorithm_style(algorithm: str) -> dict[str, str]:
    """Get matplotlib style for an algorithm"""
    return {
        "marker": AlgorithmMarkerStyle[algorithm],
        "linestyle": AlgorithmLineStyle[algorithm],
        "label": AlgorithmName[algorithm],
        "color": AlgorithmColor[algorithm],
    }


def flexible_figure(
    agg_df: pd.DataFrame,
    plot_type: str,
    compare_col: str = "inference",
    gb_col: str = "dgp",
    max_epsilon: float = np.inf,
    ncols: int = 2,
    sharex: bool = False,
    sharey: bool = False,
):
    """A flexible plotting function

    Examples:
        Plot the mean-variance tradeoff for each value of epsilon:
        ```
        fig, axes = mean_variance_figure(agg_df, "mean_variance")

        fig.show()
        ```

        Plot the solve time of algorithms for each value of epsilon:
        ```
        fig, axes = mean_variance_figure(agg_df, "solve_time", gb_col="posterior")

        fig.show()
        ```
    """
    gb = agg_df.groupby(gb_col)
    nrows = int(np.ceil(len(gb) / float(ncols)))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=(ncols * 5, nrows * 5),
    )
    for i, (group, group_df) in enumerate(gb):
        index_set = set(
            zip(
                group_df.index.get_level_values("algorithm"),
                group_df.index.get_level_values("dgp"),
                group_df.index.get_level_values("inference"),
                group_df.index.get_level_values("posterior"),
            )
        )

        # get the column and row indicies to get the axis
        col = i % ncols
        if nrows == 1 and ncols == 1:
            axis = axes
        elif nrows == 1:
            axis = axes[col]
        else:
            row = int(np.floor(i / 2.0))
            axis = axes[row][col]

        # add a new plot on the same axis for each parameter setting
        for algorithm, dgp, inference, posterior in index_set:
            df = group_df.loc[algorithm, dgp, :max_epsilon, inference, posterior]

            # decide how to style the lines
            if gb_col == "posterior":
                assert len(df.index.get_level_values("dgp").unique()) == 1
                style = algorithm_style(algorithm)
            elif gb_col == "dgp":
                if compare_col == "inference":
                    assert len(df.index.get_level_values("algorithm").unique()) == 1
                    style = inference_style(inference)
                elif compare_col == "algorithm":
                    assert len(df.index.get_level_values("inference").unique()) == 1
                    style = algorithm_style(algorithm)
                else:
                    raise NotImplementedError(f"Comparing column '{compare_col}' not supported.")
            else:
                raise NotImplementedError()

            # plot
            if plot_type == "mean_variance":
                mean_variance_plot(axis, df, **style)
            elif plot_type in ("solve_time", "setup_time", "posterior_time"):
                time_taken_plot(axis, df, plot_type, **style)
        # set title and label epsilon
        axis.set_title(f"{plot_type}: {group}")
        axis.legend()
    axis.legend()
    return fig, axes


def time_taken_plot(
    axis: mpl.axis.Axis,
    df: pd.DataFrame,
    time_name: str,
    **kwargs,
) -> None:
    # get a dataframe for each posterior
    axis.plot(df.index.get_level_values("epsilon"), df[time_name]["mean"], **kwargs)
    axis.set_xlabel("$\epsilon$")
    axis.set_ylabel(f"{time_name} (s)")
    axis.set_xscale("log")
    axis.set_yscale("log")


def mean_variance_plot(
    axis: mpl.axis.Axis,
    df: pd.DataFrame,
    is_labelled: bool = True,
    pareto_front_col: str = "is_minimise_pareto_front",
    offset: float = 1.0,
    special_epsilons: list[float] = [0.1, 0.5],
    **kwargs,
) -> None:
    # plot mean-variance trade-off
    epsilon_list = list(df.index.get_level_values("epsilon"))
    fillstyles = ["full" if is_pareto else "none" for is_pareto in df[pareto_front_col].tolist() ] 
    out_of_sample_var = df["out_of_sample_var"].to_list()
    out_of_sample_mean = df["out_of_sample_mean"].to_list()
    label=kwargs.pop("label")
    for i in range(len(out_of_sample_var)):
        if i == 0 and is_labelled:
            local_label = label
        else:
            local_label='_nolegend_'
        axis.plot(out_of_sample_var[i], out_of_sample_mean[i], markersize=4, fillstyle=fillstyles[i], **kwargs, label=local_label)

    axis.plot(df["out_of_sample_var"], df["out_of_sample_mean"], linestyle=kwargs["linestyle"], markersize=0, color=kwargs["color"], label='_nolegend_', alpha=kwargs["alpha"])
    # axis.scatter(posterior_var, df["mean_cost"]["mean"], s=100*np.sqrt(np.array(epsilon_list)), **kwargs)
    if is_labelled:
        
        for i, epsilon in enumerate(epsilon_list):
            # if i % 4 == 0:

            if i == 0 or i == len(epsilon_list) - 1 or epsilon in special_epsilons:
                # if line is blue then put text on bottom left
                if kwargs["color"] == AlgorithmColor.kl_dro_bas:
                    ha = "right"
                    va = "top"
                    # offset = -1 if i==0 or i == len(epsilon_list) else -5
                    local_offset = -offset

                # else if line is black then put text on top right
                elif kwargs["color"] == AlgorithmColor.kl_bdro:
                    ha = "left"
                    va = "bottom"
                    # offset = 1 if i==0 or i == len(epsilon_list) else 5
                    local_offset = offset
                else:
                    ha = "left"
                    va = "bottom"
                    local_offset = offset

                axis.text(
                    # df["out_of_sample_var"][:, :, epsilon, :].iloc[0] + local_offset,
                    # df["out_of_sample_mean"][:, :, epsilon, :].iloc[0],
                    df["out_of_sample_var"].iloc[i] + local_offset,
                    df["out_of_sample_mean"].iloc[i] + local_offset,
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
