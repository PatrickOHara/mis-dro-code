from datetime import datetime
from pathlib import Path
from uuid import uuid4
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import expon
import cvxpy as cp
from joblib import Parallel, delayed


from bayesian_dro.Bayesian_DRO_continuous import (
    main_Bayesian_DRO,
    data_generation,
    xi_generation,
    theta_generation,
)
from mis_dro.constants import *
from mis_dro.dataset import data_generation_outliers, data_generation_gamma
from mis_dro.npl import Npl
from mis_dro.newsvendor import newsvendor_cost, newsvendor_cost_cvxpy
from mis_dro.models import ExponentialModel
from mis_dro.gaussian_kernel import *
from bayesian_dro.Bayesian_DRO_continuous import EPSILON_SET


def matDecomp(K):
    # decompose matrix
    try:
        L = np.linalg.cholesky(K)
    except:
        # print('warning, Gram matrix K is singular')
        d, v = np.linalg.eigh(K) #L == U*diag(d)*U'. the scipy function forces real eigs
        d[np.where(d < 0)] = 0 # get rid of small eigs
        L = v @ np.diag(np.sqrt(d))
    return L

class KdroJointEpsBall_Cvxpy():
    '''
    Robustify the joint distribution. 
    '''
    def __init__(self, dim_theta, loss_call, Xobs, Xcert, K):
        '''
        #####
        Adjusted from https://github.com/jj-zhu/kdro/blob/main/kdro/kdro.py
        #####
        dim_theta: the dimension of theta (parameter to be optimized)

        loss_call: a callable (function) 
            (theta, xcert) |-> loss value

        K: a Gram matrix computed such that
            K_ij = k( x_i, x_j) where x_i is an observation,
            in the set Xobs (set of observations) and Xcert,
            in that order.

        Xobs: N x d numpy array containing N observations.

        Xcert: n x d numpy array containing n input locations to certify
            the robustness . Can be None (empty) in which case the
            optimization is less robust. obs is part of the set of certifying
            points by default.

        '''
        assert dim_theta > 0 
        #assert K.shape[1] >= Xcert.shape[0]

        self.dim_theta = dim_theta
        self.loss_call = loss_call
        self.Xobs = Xobs
        self.Xcert = Xcert
        self.K = K

    def get_problem(self, n_sample, num_certify_samples):
        '''
        '''
        #results = []
        # n_sample = num_posterior_samples*num_likelihood_samples
        # sample size for the set of certification points
        n_certify = num_certify_samples
        # K = cp.Parameter((n_sample+n_certify, n_sample+n_certify), name="K")
        # K_decomposed = cp.Parameter((n_sample+n_certify, n_sample+n_certify), name="K_decomposed")
        epsilon = cp.Parameter(nonneg=True, name="epsilon")
        # Xobs = cp.Parameter((n_sample,1), name="Xobs")
        # Xcert = cp.Parameter((n_certify,1), name="Xcert")
        
        # All variables to be optimized
        theta = cp.Variable(self.dim_theta, name="theta")

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(n_sample+n_certify)

        # function values at the kernel_points
        fvals = self.K @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call
        # always certify the observations
        for i in range(n_sample):
            constraints += [loss_call(theta, self.Xobs[i]) 
            <= f0 + fvals[i] ]

        # certify the certifying points
        for i in range(n_certify):
            # wi = self.cert_locs[i]
            xcert_i = self.Xcert[i]
            constraints += [loss_call(theta, xcert_i) <= f0 +
            fvals[i+n_sample]]
        
        emp = f0 + cp.sum(fvals[:n_sample]) / n_sample
        # regularization term
        # rkhs_norm = cp.sqrt(cp.quad_form(beta, K + 1e+1*np.eye(K.shape[0])))
        K_decomposed = np.asarray(mat_decomp_jax(self.K))
        rkhs_norm = cp.norm(beta.T @ K_decomposed) # pass matdecomp directly
        reg_term = epsilon * rkhs_norm

        # objective function
        obj = emp + reg_term
        opt = cp.Problem(cp.Minimize(obj), constraints)
        
        # for eps in EPSILON_SET:
        #     epsilon.value = eps
        #     opt.solve(cp.MOSEK, verbose=False)  #solver=cp.MOSEK cp.ECOS_BB cp.GUROBI
        #     results.append(theta.value)
        # def main_loop(eps):
        #     epsilon.value = eps
        #     opt.solve(cp.MOSEK, verbose=False)
        #     return theta.value
            
        # results = Parallel(n_jobs=-1)(delayed(main_loop)(eps) for eps in [0.5])
        
        return opt
    
# def optimise(loss_call, xi, n_certify, l, dim_theta):
#     '''
#     Main function to create gram matrix and optimise objective
#     '''
#     # Sample certification points
#     # Compute Gram matrix 
#     # Optimize
#     # _, dim_x = xi.shape
#     # Xcert = np.random.uniform(np.min(xi), np.max(xi), size=[n_certify,dim_x])
#     # zetai = np.concatenate([xi, Xcert])
#     # if l == -1: # median heuristic
#     #     l = np.sqrt((1/2)*np.median(distance.cdist(zetai, zetai, 'sqeuclidean')))
#     # K = k_jax(zetai, zetai, l)
#     kdro_class = KdroJointEpsBall_Cvxpy(dim_theta, loss_call, xi, Xcert)
#     results = kdro_class.robust_opt()
    
#     return results

def main(num_replications, 
         num_observations, 
         num_test_observations, 
         num_likelihood_samples, 
         contamination, 
         num_posterior_samples, 
         n_certify, 
         lengthscale=-1, 
         posterior='mmd',  # bayes
         algorithm='kdro_exp_mmd',     # bayesian_dro
         dgp="contaminated_exp",
         experiment_dir="./misdro/results_kdro/"):
    
    p = 1  # numbers of unknown parameters
    dim_theta = 1 # dimension of unknown parameter
    cost = np.zeros((num_replications, num_test_observations, len(EPSILON_SET)))  # init costs for each run
    solutions = np.zeros((num_replications, len(EPSILON_SET)))
    times = {
        "dgp_time": [],
        "posterior_time": [],
        "likelihood_time": [],
        "solve_time": [],
    }
    #
    # If the number of parameters is small enough, then use Disciplined Parametrized Programming (DPP)
    # to reduce the compilation time in each replication.
    # However, a large number of parameters uses an enormous amout of RAM in the current cvxpy implementation.
    # ignore_dpp = False
    # n_parameters = np.sum(np.prod(param.shape) for param in problem.parameters())
    # if n_parameters >= cp.settings.PARAM_THRESHOLD:
    #     ignore_dpp = True
        
    for j in range(num_replications):
        # generate dataset
        dgp_start = datetime.now()
        generator = np.random.default_rng(seed=j)
        if dgp == "truncated_normal":
            data = data_generation(num_observations, random_state=generator)  # generate new observations
            data_eval = data_generation(num_test_observations, random_state=generator)  # test data
        elif dgp == "contaminated_exp":
            # specify contamination level for train data
            data = data_generation_outliers(num_observations, contamination, random_state=generator)
            # but DO NOT specify any contamination for test data!
            data_eval = data_generation_outliers(num_test_observations, 0.0, random_state=generator)
        elif dgp == "exponential":
            data = expon.rvs(scale=20.0, size=num_observations, random_state=generator)
            data_eval = expon.rvs(scale=20.0, size=num_test_observations, random_state=generator)
        elif dgp == "gamma":
            data = data_generation_gamma(num_observations, a=10, random_state=generator)
            data_eval = data_generation_gamma(num_test_observations, a=10, random_state=generator)
        else:
            raise ValueError(
                f"The data-generating process specified is not supported: {dgp}"
            )
        times["dgp_time"].append((datetime.now() - dgp_start).total_seconds())

        # sample from the posterior
        posterior_start = datetime.now()
        theta_sample = np.zeros((num_posterior_samples, 1))
        if posterior in ("wll", "mmd"):
            # NPL posterior sample for theta
            m = num_observations
            model = ExponentialModel(m)
            npl_toy = Npl(
                data.reshape((num_observations, 1)),
                num_posterior_samples,
                p,
                m,
                model,
                seed=j,
                l=lengthscale,
                loss_fn=posterior,
            )
            npl_toy.draw_samples(random_state=generator)
            theta_sample = npl_toy.sample
        elif posterior == "bayes":
            # standard Bayesian posterior sample for theta
            theta_sample = theta_generation(data, num_posterior_samples, random_state=generator)
        times["posterior_time"].append(
            (datetime.now() - posterior_start).total_seconds()
        )

        # sample from the likelihood
        likelihood_start = datetime.now()
        xi = np.zeros([num_posterior_samples, num_likelihood_samples])  # init the xi's
        for i in range(num_posterior_samples):
            xi[i] = xi_generation(theta_sample[i], num_likelihood_samples, random_state=generator)
        times["likelihood_time"].append(
            (datetime.now() - likelihood_start).total_seconds()
        )

        # run the chosen DRO algorithm
        solve_start = datetime.now()
        if algorithm == "bayesian_dro":
            for i, eps in enumerate(EPSILON_SET):
                solutions[j,i] = main_Bayesian_DRO(xi, eps)
        # # TODO put in other algorithms here, e.g. main_Bayesian_DRO_epsilon1!
        elif algorithm == "kdro_exp_mmd":
            xi = xi.reshape((num_likelihood_samples*num_posterior_samples,1))
            _, dim_x = xi.shape
            Xcert = np.random.uniform(np.min(xi), np.max(xi), size=[n_certify,dim_x])
            zetai = np.concatenate([xi, Xcert])
            # if l == -1: # median heuristic
            l = np.sqrt((1/2)*np.median(distance.cdist(zetai, zetai, 'sqeuclidean')))
            K = k_jax_sym(zetai, zetai, l)
            print("here")
            # K_decomp = mat_decomp_jax(K)
            kdro_class = KdroJointEpsBall_Cvxpy(dim_theta, newsvendor_cost_cvxpy, xi, Xcert, K)
            n_samples = num_posterior_samples*num_likelihood_samples
            problem = kdro_class.get_problem(n_samples, n_certify)
            # problem.param_dict["Xobs"].value = xi
            # problem.param_dict["Xcert"].value = Xcert
            # problem.param_dict["K"].value = np.asarray(K)
            # problem.param_dict["K_decomposed"].value = np.asarray(K_decomp)
            thetas_list = []
            for eps in EPSILON_SET:
                problem.param_dict["epsilon"].value = eps
                problem.solve(cp.MOSEK, verbose=True)  #solver=cp.MOSEK cp.ECOS_BB cp.GUROBI, , ignore_dpp=ignore_dpp
                thetas_list.append(problem.var_dict["theta"].value)
            # thetas_list = optimise(newsvendor_cost_cvxpy, 
            #                         xi.reshape((num_posterior_samples*num_likelihood_samples,1)), 
            #                         n_certify, lengthscale, dim_theta)
            solutions[j,:] = np.asarray(thetas_list).flatten()
        else:
            solutions[j,:] = 0
            raise ValueError("Please choose a valid algorithm")
        times["solve_time"].append((datetime.now() - solve_start).total_seconds())

        # evaluate the cost
        for i,eps in enumerate(EPSILON_SET):
            cost[j,:,i] = newsvendor_cost(solutions[j,i], data_eval)

    # Calculate out-of-sample mean and variance
    mean_cost = np.mean(cost, axis=1)
    var_cost = np.var(cost, axis=1)
    mean_of_means = np.mean(mean_cost)
    mean_of_variances = np.mean(var_cost)
    var_of_means = np.var(mean_cost)

    print(f"Finished running {algorithm} with posterior {posterior} and DGP {dgp}.")
    print(f"Out-of-sample mean: {mean_of_means}. Out-of-sample variances: {mean_of_variances + var_of_means}.")
    print("Total DGP time:", sum(times["dgp_time"]))
    print("Total posterior time:", sum(times["posterior_time"]))
    print("Total likelihood time:", sum(times["likelihood_time"]))
    print("Total solve time:", sum(times["solve_time"]))

    for i, eps in enumerate(EPSILON_SET):
        df = pd.DataFrame({
        "replication": np.arange(num_replications),
        "mean_cost": mean_cost[:,i],
        "var_cost": var_cost[:,i],
        "solution": solutions[:,i],
        "dgp_time": times["dgp_time"],
        "likelihood_time": times["likelihood_time"],
        "posterior_time": times["posterior_time"],
        "solve_time": times["solve_time"],
        "epsilon": eps
        })
        df.to_csv(experiment_dir + f"test_results_eps_{eps}.csv", index=False)
    
if __name__ == "__main__":
    num_replications = 10
    num_observations = 20
    num_test_observations = 20
    num_likelihood_samples = 100
    contamination = 0.1
    num_posterior_samples = 10
    n_certify = 20
            
    main(num_replications, 
        num_observations, 
        num_test_observations, 
        num_likelihood_samples, 
        contamination, 
        num_posterior_samples, 
        n_certify)
    