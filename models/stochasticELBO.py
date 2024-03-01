""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import math
import numpy as np
import torch
from torch.nn import Module, Parameter
from .common import *



class StochasticSC(LinearLVModel):
    """ Sparse coding model.
        Computes stochastic ELBO estimate via reparameterization.
    """

    def __init__(self, N, D, H, variationalparams=None, nsamples=1) -> None:
        super().__init__(N, D, H, variationalparams)
        self.nsamples = nsamples  # number of MC samples 

        self.sigma_param = Parameter(torch.Tensor([1.0]))
        

    def make_W(self):
        return self.W


    def make_sigmasqr(self):
        return self.sigma_param**2


    def forward(self, X, indexes):
        # Draw nsamples from the posterior for every data point
        z_samples, L, Sigma = self.sample_posterior(X, indexes, self.nsamples)
        
        # Generate X
        xstar = self.sample_data(z_samples, only_means=True)

        # Compute final ELBO
        sigmasqr = self.make_sigmasqr()
        elbo = unit_Laplace_logprob(z_samples).sum(-1) + zeromean_Gaussian_logprob(X-xstar, sigmasqr).sum(-1) + multivar_normal_entropy(L)
        average_elbo = elbo.sum() / (X.shape[0]* self.nsamples)
        return average_elbo
        
    
class LinearGaussianModel(LinearLVModel):
    """ Sparse coding model.
        Computes stochastic ELBO estimate via reparameterization.
    """

    def __init__(self, N, D, H, variationalparams=None, nsamples=1) -> None:
        super().__init__(N, D, H, variationalparams)
        self.nsamples = nsamples  # number of MC samples 

        self.sigma_param = Parameter(torch.Tensor([1.0]))
        

    def make_W(self):
        return self.W


    def make_sigmasqr(self):
        return self.sigma_param**2


    def forward(self, X, indexes):
        # Draw nsamples from the posterior for every data point
        z_samples, L, Sigma = self.sample_posterior(X, indexes, self.nsamples)
        
        # Generate X
        xstar = self.sample_data(z_samples, only_means=True)

        # Compute final ELBO
        sigmasqr = self.make_sigmasqr()
        elbo = unit_Gaussian_logprob(z_samples).sum(-1) + zeromean_Gaussian_logprob(X-xstar, sigmasqr).sum(-1) + multivar_normal_entropy(L)
        average_elbo = elbo.sum() / (X.shape[0]* self.nsamples)
        return average_elbo
        
    
