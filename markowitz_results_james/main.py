import numpy as np
from scipy.optimize import minimize
import pickle
from datetime import datetime

# TODO: cite https://github.com/BorisForce/PyPortfolioModels/blob/main/Min_Mean_Variance/Min_Mean_Variance_model.py for code if necessary
def mean_variance_opt(Sigma: np.ndarray, mu: np.ndarray, risk_aversion: float):
    """
    Compute the mean-variance optimal portfolio weights subject to the constraints of no short-selling (weights >= 0)
    and full investment (sum(weights) = 1).
    """
    n = len(mu)
    
    def objective(w):
        return - (np.dot(w, mu) - 0.5 * risk_aversion * np.dot(w, Sigma @ w))
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

djia_windows_filename_markowitz = "/dcs/pg24/u5674159/mis-dro-code/james-data/windows_rebalance_dates_20080220_to_20250430_inclusive_every_13_weeks.pkl"
with open(djia_windows_filename_markowitz, 'rb') as f:
    windows = pickle.load(f)

lambdas = [
    1,
    5,
    10,
    50
]

results_for_all_lambas = {}

for risk_aversion_hyperparameter in lambdas:

    results_for_each_window = []

    for window in windows:
        training_df, test_df = window

        # Start a timer for measuring total time up to and including solve for window
        Sigma_mu_calculation_start = datetime.now()

        # Take the mean for each stock to get mu (make sure to do this properly, considering dimensionality etc.)
        mu = training_df.mean(axis=0) ** len(test_df)

        # Create a covariance matrix (Sigma) out of the 13-weekly returns (make sure to do this properly, considering dimensionality etc.)
        Sigma = training_df.cov()

        # Start a timer for measuring solve time for window
        solve_start = datetime.now()

        # Supply Sigma, mu and a chosen risk-aversion (remember, you planned to trial values on the upper side of the 1-50 range) to get the portfolio weighting, x
        # TODO: selecting 50 as the lambda for now, may do sensitivity analysis
        portfolio_weighting = mean_variance_opt(Sigma, mu, risk_aversion=risk_aversion_hyperparameter)

        # Stop the timer for measuring solve time for window to finalise it
        solve_time = (datetime.now() - solve_start).total_seconds()

        # Stop the timer for measuring total time for window to finalise it
        time_from_Sigma_mu_calculation_to_solving_inclusive = (datetime.now() - Sigma_mu_calculation_start).total_seconds()

        # Calculate the out-of-sample cumulative returns using the portfolio weighting and the test dataset--see how out_of_sample_cost is calculated in mis_dro/main.py
        out_of_sample_cost = test_df @ portfolio_weighting

        # Save the portfolio weighting, the solve time, the total time up to and including solve and the out-of-sample cumulative returns for this window in a dictionary
        results = {
            "portfolio_weighting": portfolio_weighting,
            "solve_time": solve_time,
            "time_from_Sigma_mu_calculation_to_solving_inclusive": time_from_Sigma_mu_calculation_to_solving_inclusive,
            "out_of_sample_cost": out_of_sample_cost
        }

        # Append this dictionary to the results_for_each_window
        results_for_each_window.append(results)

    results_for_all_lambas[risk_aversion_hyperparameter] = results_for_each_window

# 4. Pickle results_for_each_window
with open("/dcs/pg24/u5674159/mis-dro-code/markowitz_results_james/results_for_multiple_lambdas.pkl", "wb") as f:
    pickle.dump(results_for_all_lambas, f)