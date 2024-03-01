""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from .common import *
from utils.numerical import ErrorFunction

if True:
    erf = torch.erf
else:
    erf = ErrorFunction.apply

class OptimalManifoldSC(LinearLVModel):
    """ Sparse coding model.
        Computes ELBO on the optimal manifold accorging to the 3-entropies result.
    """
    
    def __init__(self, N, D, H, variationalparams=None, on_compute_entropy=None, constrain_W=True) -> None:
        super().__init__(N, D, H, variationalparams)
        
        self.on_compute_alpha = None
        self.on_compute_entropy = on_compute_entropy
        self.likelihood_scale = 1
        self.prior_scale = 1 
        self.proposal_scale = 1
        self.constrain_W = constrain_W
        

    def make_W(self):
        if self.constrain_W:
            return self.W / torch.sqrt((self.W**2).sum(0))
        else:
            return self.W
        

    def make_sigmasqr(self, mu, Sigma, X, W):
        N = X.shape[0]
        D, H = W.shape
            
        if isbatchofsquarematrices(Sigma):
            assert Sigma.shape == (N, H, H)
            assert mu.shape == (N, H)
            
            sigmasqr = (btrace(torch.matmul(Sigma, torch.mm(W.T, W))).sum() + \
                torch.square(torch.mm(W, mu.T).T - X).sum()) / (self.D * Sigma.shape[0])
        else:
            assert Sigma.shape == (N, H)
            assert mu.shape == (N, H)

            sigmasqr = ((Sigma * (W**2).sum(0)).sum() + \
                torch.square(torch.mm(W, mu.T).T - X).sum()) / (self.D * Sigma.shape[0])
        
        return sigmasqr


    def make_alpha(self, mu, Sigma):
        N, H = mu.shape
        
        if isbatchofsquarematrices(Sigma):
            assert Sigma.shape == (N, H, H)
            Sigma_hh = torch.diagonal(Sigma, dim1=-2, dim2=-1)
        else:
            assert Sigma.shape == (N, H)
            Sigma_hh = Sigma
            
        Sigma_hh_sqrt = torch.sqrt(Sigma_hh)
        alpha = torch.sum((2.0 / math.sqrt(2*math.pi)) * Sigma_hh_sqrt  * torch.exp(-0.5 * mu**2 / Sigma_hh) + \
            mu * erf(math.sqrt(2.0) / 2.0 * mu / Sigma_hh_sqrt),
            dim=0) / Sigma.shape[0]
        return alpha
        
    
    def elbo_on_manifold(self, X, indexes):
        W = self.make_W()
        mu, L, Sigma = self.variationalparams(X, indexes)
        sigmasqr = self.make_sigmasqr(mu, Sigma, X, W)
        alpha = self.make_alpha(mu, Sigma)
        if self.on_compute_alpha is not None:
            self.on_compute_alpha(alpha)
        likelihood_entropy = 0.5 * self.D * torch.log(2*math.pi*math.e*sigmasqr)
        prior_entropy = torch.log(2*math.e*alpha).sum()
        proposal_entropy = multivar_normal_entropy(L).sum() / Sigma.shape[0]
        elbo = - self.likelihood_scale*likelihood_entropy \
               - self.prior_scale*prior_entropy \
               + self.proposal_scale*proposal_entropy
        if self.on_compute_entropy is not None:
            self.on_compute_entropy(likelihood_entropy, prior_entropy, proposal_entropy)
        return elbo


    def forward(self, X, indexes):
        return self.elbo_on_manifold(X, indexes)



def compute_alpha(model, dataset):
    batch_size = 512
    alphas = []
    def on_compute_alpha(alpha):
        alphas.append(alpha.cpu().detach().numpy())
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.on_compute_alpha = on_compute_alpha
    for batch_idx, (data, indexes) in enumerate(dataloader):
        data = data.type(torch.get_default_dtype()).to(model.device)
        model(data, indexes)

    model.on_compute_alpha = None
    alphas = np.stack(alphas).mean(axis=0)
    return alphas

