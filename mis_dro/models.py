"""Model classes compatible for use within the npl_mmd class"""

import jax
import jax.numpy as jnp

class exponential_model():

    def __init__(self, m):
        self.m = m  # number of points sampled from the model at each approximation of the MMD

    def sample(self,theta, key):
        lamb = jnp.exp(theta) # Re-parametrisation to ensure lambda > 0! 
        x = jax.random.exponential(key, shape=(self.m,1))/lamb # Exponential with parameter lambda 

        return x
    
