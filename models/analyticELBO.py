""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import math
import torch
from torch.nn import Module, Parameter
from .common import *



class AnalyticELBOSC(LinearLVModel):
    def __init__(self, N, D, H, variationalparams=None) -> None:
        super().__init__(N, D, H, variationalparams)

        self.sigmasqr_param = Parameter(torch.Tensor([1.0]))


    def make_W(self):
        return self.W
        #return self.W / torch.sqrt((self.W**2).sum(0))
        
    
    def make_sigmasqr(self):
        return torch.square(self.sigmasqr_param)

    
    def marginal_loglik(self, W, X, mu, Sigma, sigmasqr):
        N = X.shape[0]
        D, H = W.shape
            
        if isbatchofsquarematrices(Sigma):
            assert Sigma.shape == (N, H, H)
            assert mu.shape == (N, H)
        
            loglik = -0.5 * self.D * torch.log(2*math.pi*sigmasqr) \
                -(1/(2*sigmasqr)) * (btrace(torch.matmul(Sigma, torch.mm(W.T, W))) + \
                torch.square(torch.mm(W, mu.T).T - X).sum(-1))
        else:
            assert Sigma.shape == (N, H)
            assert mu.shape == (N, H)

            loglik = -0.5 * self.D * torch.log(2*math.pi*sigmasqr) \
                -(1/(2*sigmasqr)) * ((Sigma * (W**2).sum(0)).sum(-1) + \
                torch.square(torch.mm(W, mu.T).T - X).sum(-1))
        return loglik
    

    def gauss_laplace_cross_entropy(self, mu, Sigma):
        Sigma_hh = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        Sigma_hh_sqrt = torch.sqrt(Sigma_hh)
        crossentropy = -self.H*math.log(0.5) + \
            ((2.0 / math.sqrt(2*math.pi)) * Sigma_hh_sqrt  * torch.exp(-0.5 * mu**2 / Sigma_hh) + \
            mu * torch.erf(math.sqrt(2.0) / 2.0 * mu / Sigma_hh_sqrt)).sum(-1)
        return crossentropy
        

    def make_elbo(self, X, indexes):
        mu, L, Sigma = self.variationalparams(X, indexes)
        sigmasqr = self.make_sigmasqr()
        W = self.make_W()
        elbo = (self.marginal_loglik(W, X, mu, Sigma, sigmasqr) \
                -self.gauss_laplace_cross_entropy(mu, Sigma) \
                + multivar_normal_entropy(L)).sum() / self.N
        return elbo


    def forward(self, X, indexes):
        return self.make_elbo(X, indexes)

