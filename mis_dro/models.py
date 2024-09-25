"""Model classes compatible for use within the npl_mmd class"""

import jax
import jax.numpy as jnp
from bayesian_dro.Bayesian_DRO_continuous import DGP_STD_TRUNCATED_NORMAL


class ExponentialModel:
    def __init__(self, m):
        self.m = m  # number of points sampled from the model at each approximation of the MMD

    def sample(self, theta, key):
        lamb = jnp.exp(theta)  # Re-parametrisation to ensure lambda > 0!
        x = (
            jax.random.exponential(key, shape=(self.m, 1)) / lamb
        )  # Exponential with parameter lambda
        return x
    
    def init_params(self, data):
        # This function return the initialisation parameters for the minimisation of the mmd
        return jnp.log((1/jnp.mean(data)))*jnp.ones(1) # Initialisation of unknown parameter, here I inistialise at MLE
    
    def parametrise(self, theta):
        return jnp.exp(theta)  # rate parameter of expoenential model is re-parametrised to ensure positivity
        
        
    
class univariate_GaussianModel:
    def __init__(self, m):
        self.m = m
    
    def sample(self, theta, key):
        mu = theta[0]
        std = jnp.exp(theta[1]) # make sure standard deviation is positive!
        x = (
            mu + std*jax.random.normal(key, shape=(self.m,1))
        )
        
        return x
    
    def init_params(self, data):
        return jnp.array([jnp.mean(data), jnp.log(jnp.std(data))]).reshape((2,))
    
    def parametrise(self, theta): 
        theta = theta.at[1].set(jnp.exp(theta[1]))  # scale parameter is reparametrised to ensure postivity - now parametrise back
        return theta 
    
class univariate_GaussianModel_known_variance:
    def __init__(self, m):
        self.m = m
    
    def sample(self, theta, key):
        mu = theta[0]
        std = DGP_STD_TRUNCATED_NORMAL
        x = (
            mu + std*jax.random.normal(key, shape=(self.m,1))
        )
        
        return x
    
    def init_params(self, data):
        return jnp.mean(data).reshape((1,))
    
    def parametrise(self, theta):
        return theta

class multivariate_GaussianModel:
    def __init__(self, m, d):
        self.m = m
        self.d = d
    
    def sample(self, theta, key):
        mu = theta
        sigma = DGP_STD_TRUNCATED_NORMAL
        x = (
            jax.random.multivariate_normal(key, mean = mu, cov = sigma*jnp.eye(self.d), shape=(self.m,self.d))
        )
        
        return x
    
    def init_params(self, data):
        return  jnp.mean(data, axis=0).reshape((self.d,))
    
    def parametrise(self, theta):
        return theta
    
class regression_GaussianModel:
    def __init__(self, price):
        self.price = price
        self.m = len(price)
        
    def sample(self, theta, key):
        a = theta[0]
        b = theta[1]
        var = theta[2] # make sure standard deviation is positive!
        mu = a - b*self.price
        demand = (
            jax.random.multivariate_normal(key, mean=mu, cov=var*jnp.eye(self.m), shape=(self.m,1))
        )
        return demand
    
    def init_params(self, data, covariates):
        # For regression parameters initialise with the OLS estimator 
        # For the std initialise with the sample std
        init_theta = jnp.zeros(3)
        # regression terms
        X = jnp.column_stack((jnp.ones_like(covariates), -covariates))
        coeffs, residuals, _, _ = jnp.linalg.lstsq(X, data, rcond=None)
        SSR = residuals[0]
        var_init = SSR / (len(data) - 2)
        a_init, b_init = coeffs
        init_theta = init_theta.at[0].set(a_init[0])
        init_theta = init_theta.at[1].set(b_init[0])
        init_theta = init_theta.at[2].set(var_init)
        return init_theta
    
    def parametrise(self, theta):
        theta = theta.at[2].set(jnp.sqrt(theta[2])) # return the std
        return theta
        
    
    

        
    
