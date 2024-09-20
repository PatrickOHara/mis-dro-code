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
    
class regression_GaussianModel:
    def __init__(self, price):
        self.price = price
        self.m = len(price)
        
    def sample(self, theta, key):
        a = theta[0]
        b = theta[1]
        std = theta[2]
        mu = a - b*self.price
        demand = (
            jax.random.multivariate_normal(key, mean=mu, cov=(std**2)*jnp.eye(self.m), shape=(self.m,1))
        )
        return demand
        