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
        # std = DGP_STD_TRUNCATED_NORMAL
        std = 5
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
        sigma = 5
        x = (
            jax.random.multivariate_normal(key, mean = mu, cov = (sigma**2)*jnp.eye(self.d), shape=(self.m,self.d))
        )
        
        return x
    
    def init_params(self, data):
        return  jnp.mean(data, axis=0).reshape((self.d,))
    
    def parametrise(self, theta):
        return theta
    

        
    