from enum import StrEnum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AlgorithmLineStyle(StrEnum):
    """Consistent algorithm line styles"""

    bdro_grid_search = "solid"
    kl_bdro = "dashed"
    our_kl_bdro = "dotted"


class AlgorithmMarkerStyle(StrEnum):
    """Consistent algorithm marker styles"""

    bdro_grid_search = "o"
    kl_bdro = "x"
    our_kl_bdro = "*"

class AlgorithmColor(StrEnum):
    """Colors of algorithm lines"""
    bdro_grid_search = "orange"
    kl_bdro = "black"
    our_kl_bdro = "blue"



class AlgorithmName(StrEnum):
    """Consistent algorithm line styles"""

    bdro_grid_search = "BDRO grid search"
    kl_bdro = "BDRO"
    our_kl_bdro = "BAS-DRO"


class InferenceLineStyle(StrEnum):
    """Consistent line styles"""

    bayes = "solid"
    npl_wlb = "dashed"
    npl_mmd = "dotted"


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
    **kwargs,
) -> None:
    posterior_var = df["var_cost"]["mean"] + df["mean_cost"]["var"]

    # plot mean-variance trade-off
    axis.plot(posterior_var, df["mean_cost"]["mean"], **kwargs)
    axis.set_xlabel("out-of-sample variance")
    axis.set_ylabel("out-of-sample mean")
    if is_labelled:
        epsilon_list = list(df.index.get_level_values("epsilon"))
        for i, epsilon in enumerate(epsilon_list):
            if i % 4 == 0:
                axis.text(
                    posterior_var[:, :, epsilon, :].iloc[0],
                    df["mean_cost"]["mean"][:, :, epsilon, :].iloc[0],
                    epsilon,
                    ha="left",
                    va="bottom",
                )
