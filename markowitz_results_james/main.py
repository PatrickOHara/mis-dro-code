import numpy as np
from scipy.optimize import minimize
import pickle
from datetime import datetime
from tqdm import tqdm
from mis_dro.portfolio import calculate_transaction_cost
from mis_dro.metrics import calculate_sharpe_ratio, calculate_sortino_ratio
from argparse import ArgumentParser

# === DO NOT CHANGE THIS ===
using_ipynb = False
parser = ArgumentParser()
parser.add_argument(
    "--num-folds",
    type=int,
    default=None,
)
parser.add_argument(
    "--ratio-type",
    choices=["sharpe", "sortino"],
    default=None,
)
args = parser.parse_args()
num_folds_for_rolling_validation = args.num_folds
ratio_types = (args.ratio_type,)
# === DO NOT CHANGE THIS ===

# TODO: cite https://github.com/BorisForce/PyPortfolioModels/blob/main/Min_Mean_Variance/Min_Mean_Variance_model.py for code if necessary
def mean_variance_opt(Sigma: np.ndarray, mu: np.ndarray, risk_aversion: float, include_transaction_costs_in_cost_function, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list):
    """
    Compute the mean-variance optimal portfolio weights subject to the constraints of no short-selling (weights >= 0)
    and full investment (sum(weights) = 1).
    """
    n = len(mu)
    
    def objective(w):
        cost = - np.dot(w, mu) + 0.5 * risk_aversion * np.dot(w, Sigma @ w)
        if include_transaction_costs_in_cost_function:
            cost += calculate_transaction_cost(prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, w)
        return cost
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def do_markowitz(training_df, test_df, risk_aversion_hyperparameter, ratio_type, all_out_of_sample_costs_so_far, risk_free_rates, period_for_ratio_in_weeks, include_transaction_costs_in_portfolio_returns, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, include_transaction_costs_in_cost_function):

    # Start a timer for measuring total time up to and including solve for window
    Sigma_mu_calculation_start = datetime.now()

    # For each stock, take the mean of the 13-weekly returns in the training set to get its expected 13-weekly return in the test period
    mu = training_df.mean(axis=0)

    # Create a covariance matrix (Sigma) out of the 13-weekly returns (make sure to do this properly, considering dimensionality etc.)
    Sigma = training_df.cov()

    # Start a timer for measuring solve time for window
    solve_start = datetime.now()

    # Supply Sigma, mu and a chosen risk-aversion (remember, you planned to trial values on the upper side of the 1-50 range) to get the portfolio weighting, x
    portfolio_weighting = mean_variance_opt(Sigma, mu, risk_aversion_hyperparameter, include_transaction_costs_in_cost_function, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list)

    # Stop the timer for measuring solve time for window to finalise it
    solve_time = (datetime.now() - solve_start).total_seconds()

    # Stop the timer for measuring total time for window to finalise it
    time_from_Sigma_mu_calculation_to_solving_inclusive = (datetime.now() - Sigma_mu_calculation_start).total_seconds()

    # Calculate the out-of-sample cumulative returns using the portfolio weighting and the test dataset--see how out_of_sample_cost is calculated in mis_dro/main.py
    out_of_sample_cost = test_df @ portfolio_weighting

    if include_transaction_costs_in_portfolio_returns:

        transaction_cost = calculate_transaction_cost(prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, list(portfolio_weighting))

        out_of_sample_cost.iloc[0] -= transaction_cost

    if ratio_type:

        all_out_of_sample_costs_so_far = all_out_of_sample_costs_so_far + list(out_of_sample_cost)

        ratio = {"sharpe": calculate_sharpe_ratio, "sortino": calculate_sortino_ratio}[ratio_type](all_out_of_sample_costs_so_far, risk_free_rates, period_for_ratio_in_weeks)

    results_object = {
        "portfolio_weighting": portfolio_weighting, # TODO: start saving in list format rather than as a numpy array, then modify james-portfolio.ipynb accordingly
        "solve_time": solve_time,
        "time_from_Sigma_mu_calculation_to_solving_inclusive": time_from_Sigma_mu_calculation_to_solving_inclusive,
        "out_of_sample_cost": out_of_sample_cost,
        "risk_aversion_hyperparameter": risk_aversion_hyperparameter
    }

    if ratio_type:

        results_object[ratio_type] = ratio
        results_object["all_out_of_sample_costs_so_far"] = all_out_of_sample_costs_so_far

    return results_object

def unpickle_data(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

def pickle_results(results, using_ipynb, save_filename):

    with open(f"/dcs/pg24/u5674159/mis-dro-code/markowitz_results_james/{'ipynb_' if using_ipynb else ''}{save_filename}.pkl", "wb") as f:
        pickle.dump(results, f)

def do_markowitz_run_without_validation(djia_windows_filename, lambdas, save, using_ipynb, save_filename):

    windows = unpickle_data(djia_windows_filename)

    results_for_all_lambas = {}

    for risk_aversion_hyperparameter in lambdas:

        results_for_each_window = []

        iterable = windows

        if using_ipynb:

            iterable = tqdm(iterable)

        for window in iterable:
            training_df, test_df = window

            # NOTE: I have not included transaction costs below, in order to reflect (post refactor) how I got the existing non-validation data--I got the out of sample costs without including transaction costs, then included them in results processing in james-portfolio.ipynb
            results = do_markowitz(training_df, test_df, risk_aversion_hyperparameter, None, None, None, None, False, None, None, None, False)

            results_for_each_window.append(results)

        results_for_all_lambas[risk_aversion_hyperparameter] = results_for_each_window

    if save:

        pickle_results(results_for_all_lambas, using_ipynb, save_filename)

def get_single_holdout_dataset_sizes_same_ratio_as_train_test(training_df, test_df):

    single_holdout_num_training_observations = round(len(training_df) / (len(training_df) + len(test_df)) * len(training_df))

    single_holdout_num_validation_observations = len(training_df) - single_holdout_num_training_observations

    return single_holdout_num_training_observations, single_holdout_num_validation_observations

def get_stock_figi_list(training_df):

    # TODO: this whole thing with the use of get_window_of_train_and_test_dataframes is a bit of a fudge--the stock IDs should be saved along with the other results. Moreover, I don't know why I appended _{window index} to each of the FIGIs in the first place, so undo that and get rid of the splitting done below.

    return [col.split("_")[0] for col in training_df.columns]

def choose_risk_free_rates_for_single_holdout_validation(risk_free_rates: list[float], number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation: int, num_test_weeks: int, num_validation_dates_per_window: int, window_index: int):

    t = number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation
    v = num_validation_dates_per_window

    return risk_free_rates[t: t + num_test_weeks * window_index] + risk_free_rates[t + num_test_weeks * window_index - v: t + num_test_weeks * window_index]

# TODO: refactor this with choose_risk_free_rates_for_single_holdout_validation
def choose_risk_free_rates_for_rolling_validation(risk_free_rates: list[float], number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation: int, num_test_weeks: int, num_validation_dates_per_window: int, window_index: int, number_of_folds: int, fold_index: int):

    t = number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation
    v = num_validation_dates_per_window

    # NOTE: offset_of_validation_data_from_rebalance_for_fold
    o = number_of_folds - fold_index - 1

    return risk_free_rates[t: t + num_test_weeks * window_index] + risk_free_rates[t + num_test_weeks * window_index - v - o: t + num_test_weeks * window_index - o]

def choose_risk_free_rates_for_testing(risk_free_rates: list[float], number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation: int, num_test_weeks: int, window_index: int):

    t = number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation

    return risk_free_rates[t: t + num_test_weeks * (window_index + 1)]

def get_data_for_single_holdout_validation(markowitz_training_df, test_df, dro_windows, risk_free_rates, window_index):

    single_holdout_training_df_size, single_holdout_validation_df_size = get_single_holdout_dataset_sizes_same_ratio_as_train_test(markowitz_training_df, test_df)  # 30, 10

    single_holdout_training_df = markowitz_training_df[: single_holdout_training_df_size]
    
    single_holdout_validation_df = dro_windows[window_index][0][- single_holdout_validation_df_size:]  # NOTE: doing this because of the need for weekly returns, not 13-week returns from markowitz_training_df, when it comes to validation data

    # TODO: note that 51 is specific to the risk free returns file in use (because it has 51 weekly risk-free rates up to and including the first rebalance date)
    risk_free_rates_for_validation = choose_risk_free_rates_for_single_holdout_validation(risk_free_rates, 51, 13, len(single_holdout_validation_df), window_index)

    return {
        "training_df": single_holdout_training_df,
        "validation_df": single_holdout_validation_df,
        "validation_risk_free_rates": risk_free_rates_for_validation
    }

def get_data_for_rolling_validation(markowitz_training_df, test_df, num_folds, dro_windows, window_index, risk_free_rates):

    _, rolling_validation_validation_df_size = get_single_holdout_dataset_sizes_same_ratio_as_train_test(markowitz_training_df, test_df)

    rolling_validation_training_df_size = len(markowitz_training_df) - rolling_validation_validation_df_size - num_folds + 1

    rolling_validation_folds = []

    for fold_index in range(num_folds):

        validation_start_index_for_fold = fold_index + rolling_validation_training_df_size

        rolling_validation_training_df = markowitz_training_df[fold_index: validation_start_index_for_fold]

        # TODO: note that 51 is specific to the risk free returns file in use (because it has 51 weekly risk-free rates up to and including the first rebalance date)
        rolling_validation_validation_df = dro_windows[window_index][0][validation_start_index_for_fold: validation_start_index_for_fold + rolling_validation_validation_df_size]

        risk_free_rates_for_validation = choose_risk_free_rates_for_rolling_validation(risk_free_rates, 51, 13, len(rolling_validation_validation_df), window_index, num_folds, fold_index)

        rolling_validation_folds.append({
            "training_df": rolling_validation_training_df,
            "validation_df": rolling_validation_validation_df,
            "validation_risk_free_rates": risk_free_rates_for_validation
        })

    return rolling_validation_folds

def perform_validation(lambdas, num_folds, ratio_type, all_out_of_sample_costs_so_far, period_for_test_ratio_in_weeks, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, include_transaction_costs_in_cost_function, data_for_validation):

    total_solve_time = 0

    best_lambda, highest_validation_ratio = None, -float('inf')

    for risk_aversion_hyperparameter in lambdas:

        if num_folds == 1:

            markowitz_results_object = do_markowitz(data_for_validation["training_df"], data_for_validation["validation_df"], risk_aversion_hyperparameter, ratio_type, all_out_of_sample_costs_so_far, data_for_validation["validation_risk_free_rates"], period_for_test_ratio_in_weeks - 13 + len(data_for_validation["validation_df"]), True, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, include_transaction_costs_in_cost_function)

            validation_ratio = markowitz_results_object[ratio_type]
            total_solve_time += markowitz_results_object["solve_time"]

        else:

            sum_of_validation_ratio_for_each_fold = 0

            for fold in data_for_validation:

                # TODO: check risk free rates are passed around rightly in this file
                markowitz_results_object = do_markowitz(fold["training_df"], fold["validation_df"], risk_aversion_hyperparameter, ratio_type, all_out_of_sample_costs_so_far, fold["validation_risk_free_rates"], period_for_test_ratio_in_weeks - 13 + len(fold["validation_df"]), True, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, include_transaction_costs_in_cost_function)

                sum_of_validation_ratio_for_each_fold += markowitz_results_object[ratio_type]
                total_solve_time += markowitz_results_object["solve_time"]

            validation_ratio = sum_of_validation_ratio_for_each_fold / len(data_for_validation)

        if validation_ratio > highest_validation_ratio:
            best_lambda = risk_aversion_hyperparameter
            highest_validation_ratio = validation_ratio

    return {
        "total_solve_time": total_solve_time,
        "best_lambda": best_lambda,
    }

def do_markowitz_run_with_rolling_validation_using_ratio_as_metric(markowitz_djia_windows_filename, dro_djia_windows_filename, risk_free_rates_file, lambdas, ratio_type, save, using_ipynb, save_filename_excluding_ratio_type, include_transaction_costs_in_cost_function, num_folds):

    markowitz_windows = unpickle_data(markowitz_djia_windows_filename)

    dro_windows = unpickle_data(dro_djia_windows_filename)

    results_for_each_window_with_its_best_lambda = []

    prev_stock_figi_list = []
    prev_portfolio_weighting = []

    risk_free_rates = unpickle_data(risk_free_rates_file)

    all_out_of_sample_costs_so_far = []

    iterable = enumerate(markowitz_windows)

    if using_ipynb:

        iterable = tqdm(iterable)

    for i, window in iterable:

        markowitz_training_df, test_df = window # 40, 13

        stock_figi_list = get_stock_figi_list(markowitz_training_df)

        risk_free_rates_for_testing = choose_risk_free_rates_for_testing(risk_free_rates, 51, 13, i)

        if num_folds == 1:  # Single holdout validation

            data_for_validation = get_data_for_single_holdout_validation(markowitz_training_df, test_df, dro_windows, risk_free_rates, i)

        else:

            data_for_validation = get_data_for_rolling_validation(markowitz_training_df, test_df, num_folds, dro_windows, i, risk_free_rates)

        period_for_test_ratio_in_weeks = 156

        # TODO: maybe rethink where you put this, although at the time of coding, the functions before this in the window should be pretty fast compared to Markowitz optimisation problem solving etc.
        time_just_before_hyperparameter_optimisation = datetime.now()

        validation_results = perform_validation(lambdas, num_folds, ratio_type, all_out_of_sample_costs_so_far, period_for_test_ratio_in_weeks, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, include_transaction_costs_in_cost_function, data_for_validation)

        results_for_best_lambda = do_markowitz(markowitz_training_df, test_df, validation_results["best_lambda"], ratio_type, all_out_of_sample_costs_so_far, risk_free_rates_for_testing, period_for_test_ratio_in_weeks, True, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list, include_transaction_costs_in_cost_function)

        time_for_hyperparameter_optimisation_and_running_with_best_one = (datetime.now() - time_just_before_hyperparameter_optimisation).total_seconds()
        results_for_best_lambda["time_for_hyperparameter_optimisation_and_running_with_best_one"] = time_for_hyperparameter_optimisation_and_running_with_best_one

        # TODO: redo Markowitz with SHV (including transaction costs in the portfolio returns, **not** the cost function) and in james-portfolio.ipynb, get "total_solve_time_hyperparameter_optimisation_and_final_run" rather than "solve_time" for Markowitz SHV
        results_for_best_lambda["total_solve_time_hyperparameter_optimisation_and_final_run"] = validation_results["total_solve_time"] + results_for_best_lambda["solve_time"]

        results_for_each_window_with_its_best_lambda.append(results_for_best_lambda)

        prev_stock_figi_list = stock_figi_list
        prev_portfolio_weighting = list(results_for_best_lambda["portfolio_weighting"])

        all_out_of_sample_costs_so_far = results_for_best_lambda["all_out_of_sample_costs_so_far"]

    if save:

        extended_save_file_name = f"{save_filename_excluding_ratio_type}_with_{ratio_type}"
        if include_transaction_costs_in_cost_function:
            extended_save_file_name += "_with_transaction_costs_in_cost_function"

        pickle_results(results_for_each_window_with_its_best_lambda, using_ipynb, extended_save_file_name)

markowitz_djia_windows_filename = "/dcs/pg24/u5674159/mis-dro-code/james-data/windows_rebalance_dates_20080220_to_20250430_inclusive_every_13_weeks_markowitz.pkl"
dro_djia_windows_filename = "/dcs/pg24/u5674159/mis-dro-code/james-data/windows_rebalance_dates_20080220_to_20250430_inclusive_every_13_weeks.pkl"
risk_free_returns_filename = "/dcs/pg24/u5674159/mis-dro-code/james-data/risk_free_returns_weekly_2007-03-07_to_2025-07-30_inclusive.pkl"

lambdas = (
    0.5,
    5,
    50,
    500
)

# NOTE: these won't make a difference when it comes to do_markowitz_run_without_validation, as you can see
save = True
include_transaction_costs_in_cost_function = False

if num_folds_for_rolling_validation:
    for ratio_type in ratio_types:
        if num_folds_for_rolling_validation == 1:
            do_markowitz_run_with_rolling_validation_using_ratio_as_metric(markowitz_djia_windows_filename, dro_djia_windows_filename, risk_free_returns_filename, lambdas, ratio_type, save, using_ipynb, "results_for_best_lambdas_from_single_holdout_validation", include_transaction_costs_in_cost_function, num_folds=1)
        else:
            do_markowitz_run_with_rolling_validation_using_ratio_as_metric(markowitz_djia_windows_filename, dro_djia_windows_filename, risk_free_returns_filename, lambdas, ratio_type, save, using_ipynb, f"results_for_best_lambdas_from_rolling_validation_{num_folds_for_rolling_validation}_folds", include_transaction_costs_in_cost_function, num_folds_for_rolling_validation)
else:
    do_markowitz_run_without_validation(markowitz_djia_windows_filename, lambdas, False, using_ipynb, "results_for_multiple_lambdas")