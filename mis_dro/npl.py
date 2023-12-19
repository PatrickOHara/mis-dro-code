"""NPL"""

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import dirichlet


class Npl():
    """This class contains functions to perform NPL inference (for alpha = 0 in the DP prior) for the Exponential distribution model."""

    def __init__(self, X, B, p, loss_fn='wll'):
        """
        Args:
            X: Data set
            B: number of bootstrap iterations
            p: number of unknown parameters
            loss_fn : string set to 'wll' or 'mmd' to specify either the negative log-lkh or mmd-based loss function 
        """
        self.B = B
        self.X = X
        self.p = p
        self.loss_fn = loss_fn
        self.n, self.d = self.X.shape

    def draw_samples(self):
        """Draws B samples in parallel from the nonparametric posterior"""

        weights = dirichlet.rvs(np.ones(self.n), size = self.B, random_state = 13)
        samples = np.zeros((self.B,self.p))
        
        if self.loss_fn == 'wll':
            # FIXME n_jobs > 1
            temp = Parallel(n_jobs=1, backend='multiprocessing', max_nbytes=None,batch_size="auto")(delayed(self.WLL)(self.X,weights[i,:]) for i in range(self.B))

            for i in range(self.B):
                  samples[i,:] = temp[i]
                  self.sample = np.array(samples)
        elif self.loss_fn == 'mmd':
            # FIXME implement NPL-MMD here / 1 is just a dummy value for now! 
            self.sample = 1 #np.loadtxt('mmd_samples.txt')

    def WLL(self, data, weights):
        """Get weighted negative log likelihood minimizer, for Exponential distribution model"""

        theta = np.zeros(self.d)
        for i in range(self.n):
            theta += weights[i]*data[i,:]
        return 1/theta