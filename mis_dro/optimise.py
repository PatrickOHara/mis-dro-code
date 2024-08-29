"""Optimisation code"""

from typing import Callable
import cvxpy as cp

from bayesian_dro.Bayesian_DRO_continuous import LARGEST_X, SMALLEST_X


def get_kl_bdro_problem(
    decision_objective: Callable[[cp.Variable, cp.Parameter], cp.Expression],
    num_posterior_samples: int,
    num_likelihood_samples: int,
) -> cp.Problem:
    """Bayesian DRO as a cvxpy optimisaton problem.

    Args:
        decision_objective: A callable objective function implemented using cvxpy.
            The first argument should be a cvxpy variable.
            The second argument should be a cvxpy parameter representing the data samples.
            The return should be a cvxpy expression.
        num_posterior_samples: Number of posterior samples.
        num_likelihood_samples: Number of likelihood samples for each posterior sample.

    Returns:
        problem: A cvxpy Problem object

    Notes:
        We use an epigraph variable t to upper bound the objective function G(x, xi).
        That is, we add constraints G(x, xi[i]) <= t[i] for all i = 1,...,num_posterior_samples.

        The main optimisation trick is then to use the perspective of the log-sum-exp (LSE) function.
        Specifically, the perspective is l * LSE(t[i] / l), where l is the Lagrangian variable.
        As l -> 0, then l * LSE(t[i] / l) tends to max(t[i]).
    """
    # declare variables
    x = cp.Variable(1, name="x")
    lam = [
        cp.Variable(1, name=f"lam_{i}", nonneg=True)
        for i in range(num_posterior_samples)
    ]
    t = cp.Variable((num_posterior_samples, num_likelihood_samples), name="t")

    # declare parameters
    epsilon_minus_constant = cp.Parameter(1, name="epsilon_minus_constant", nonneg=True)
    xi = cp.Parameter((num_posterior_samples, num_likelihood_samples), name="xi")

    # create the objective function for the Bayesian DRO problem
    # NOTE we pass the max function to f_recession because,
    # as lam -> 0, then lam * LSE(t[i] / lam) tends to max(t[i]).
    bdro_obj = cp.Minimize(
        (1.0 / num_posterior_samples)
        * cp.sum(
            [
                lam[i] @ (epsilon_minus_constant)
                + lam[i] * cp.log(1.0 / num_likelihood_samples)
                + cp.perspective(cp.log_sum_exp(t[i]), lam[i], f_recession=cp.max(t[i]))
                for i in range(num_posterior_samples)
            ]
        )
    )
    # add the decision objective as an epigraph constraint
    # examples of decision objectives are the newsvendor objective
    constraints = [
        x >= SMALLEST_X,
        x <= LARGEST_X,
    ] + [decision_objective(x, xi[i]) <= t[i] for i in range(num_posterior_samples)]

    return cp.Problem(bdro_obj, constraints)
