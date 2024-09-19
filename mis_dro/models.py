"""Model classes compatible for use within the npl_mmd class"""

import jax
import jax.numpy as jnp


class ExponentialModel:
    def __init__(self, m):
        self.m = m  # number of points sampled from the model at each approximation of the MMD

    def sample(self, theta, key):
        lamb = jnp.exp(theta)  # Re-parametrisation to ensure lambda > 0!
        x = (
            jax.random.exponential(key, shape=(self.m, 1)) / lamb
        )  # Exponential with parameter lambda

        return x
    
class univariate_GaussianModel:
    def __init__(self, m):
        self.m = m
    
    def sample(self, theta, key):
        mu = theta[0]
        std = theta[1]
        x = (
            mu + std*jax.random.normal(key, shape=(self.m,1))
        )
        
        return x

class multivariate_GaussianModel:
    def __init__(self, m, d):
        self.m = m
        self.d = d
    
    def sample(self, theta, key):
        mu, sigma = theta
        x = (
            jax.random.multivariate_normal(key, mean = mu, cov = sigma, shape=(self.m,self.d))
        )
        
        return x
    