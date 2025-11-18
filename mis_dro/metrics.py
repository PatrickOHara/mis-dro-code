import numpy as np

def get_weekly_values_in_period(weekly_values: list[float], period_in_weeks: int):
    return np.array(weekly_values[-period_in_weeks:])

def calculate_ratio(numerator, denominator):
    assert denominator != 0
    return numerator / denominator

def calculate_sharpe_ratio(weekly_portfolio_returns: list[float], weekly_risk_free_rates: list[float], period_in_weeks: int):
    weekly_portfolio_returns_in_period = get_weekly_values_in_period(weekly_portfolio_returns, period_in_weeks)
    weekly_risk_free_rates_in_period = get_weekly_values_in_period(weekly_risk_free_rates, period_in_weeks)
    weekly_excess_portfolio_returns_in_period = weekly_portfolio_returns_in_period - weekly_risk_free_rates_in_period
    numerator = np.mean(weekly_excess_portfolio_returns_in_period)
    denominator = np.std(weekly_excess_portfolio_returns_in_period) # TODO: note that population std is in use
    return calculate_ratio(numerator, denominator)

def calculate_sortino_ratio(weekly_portfolio_returns: list[float], weekly_targets: list[float], period_in_weeks: int):
    weekly_portfolio_returns_in_period = get_weekly_values_in_period(weekly_portfolio_returns, period_in_weeks)
    weekly_targets_in_period = get_weekly_values_in_period(weekly_targets, period_in_weeks)
    weekly_excess_portfolio_returns_in_period = weekly_portfolio_returns_in_period - weekly_targets_in_period
    numerator = np.mean(weekly_excess_portfolio_returns_in_period)
    weekly_downside_excess_portfolio_returns_in_period = np.minimum(0, weekly_excess_portfolio_returns_in_period)
    denominator = np.sqrt(np.mean(weekly_downside_excess_portfolio_returns_in_period ** 2))
    return calculate_ratio(numerator, denominator)