""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter



def isbatchofsquarematrices(A):
    return A.dim() >= 3 and A.shape[-1] == A.shape[-2]


def btrace(a):
    return torch.diagonal(a, dim1=-2, dim2=-1).sum(-1) 


def bdiag(a):
    return torch.diagonal(a, dim1=-2, dim2=-1)


def bprod(a):
    return torch.diagonal(a, dim1=-2, dim2=-1).prod(-1) 


def multivar_normal_entropy(L):
    """ Entropy of multivariate Gaussian with covariance matrix 
        given by its lower-triangular (Cholesky) decomposition.
    """
    # L is lower triangular matrix from the Cholesky decomposition
    if isbatchofsquarematrices(L):
        L = bdiag(L)
    Sigma_logdet = 2 * torch.log(torch.abs(L)).sum(-1)  # more stable
    entropy = 0.5 * Sigma_logdet + 0.5 * L.shape[-1] * math.log(2*math.pi*math.e)
    return entropy


def unit_Laplace_logprob(x):
    """ Log-prob of Laplace samples with b=1, mu=0
    """
    return -torch.abs(x) - math.log(2.0)


def unit_Gaussian_logprob(x):
    """ Log-prob of Gaussian samples with mu=0, sigmasqr=1
    """
    return -0.5*math.log(2*math.pi) -0.5*(x**2)


def zeromean_Gaussian_logprob(x, sigmasqr):
    """ Log-prob of zero mean univariate Gaussian samples
    """
    return -0.5*math.log(2*math.pi) -0.5*torch.log(sigmasqr) -0.5*(1/sigmasqr)*(x**2)


def cholesky_jitter(A, maxattempts=10):
    """ Iteratively try adding larger diagonal to perform Cholesky decomposition
        Maybe use https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/cholesky.py
    """
    try:
        return torch.linalg.cholesky(A)
    except Exception as err:
        if err.__class__.__name__ != "_LinAlgError":
            raise
    
    for i in range(maxattempts):
        jitter = 1e-9 * 10**i
        print("Cholesky: adding diagonal {:.2e}".format(jitter))
        try:
            return torch.linalg.cholesky(A + jitter*torch.eye(A.shape[-1], device=A.device))
        except Exception as err:
            if err.__class__.__name__ != "_LinAlgError":
                raise
            

class VariationalParams(Module):
    def __init__(self, N, D, H) -> None:
        super().__init__()
        self.N = N  # number of data points
        self.D = D  # observed space dimensionality
        self.H = H  # latent space dimensionality


    def forward(self, X, indexes):
        raise NotImplementedError()
    


class FullCovarGaussianVariationalParams(VariationalParams):
    def __init__(self, N, D, H) -> None:
        super().__init__(N, D, H)
        # posterior mean
        self.mu = Parameter(torch.zeros(size=(self.N, self.H)))
        # posterior covariance lower triangular parameters
        self.Sigma_param = Parameter(torch.eye(self.H).reshape(1, self.H, self.H).repeat(self.N, 1, 1))
        

    def forward(self, X, indexes=None):
        if indexes is None:
            L = torch.tril(self.Sigma_param)
            Sigma = torch.bmm(L, torch.transpose(L, -1, -2))
            return self.mu, L, Sigma
        else:
            L = torch.tril(self.Sigma_param[indexes])
            Sigma = torch.bmm(L, torch.transpose(L, -1, -2))
            return self.mu[indexes], L, Sigma


    def set(self, indexesto, other, indexesfrom):
        with torch.no_grad():
            self.mu[indexesto] = other.mu[indexesfrom]
            self.Sigma_param[indexesto] = other.Sigma_param[indexesfrom]
        


class DiagCovarGaussianVariationalParams(VariationalParams):
    def __init__(self, N, D, H) -> None:
        super().__init__(N, D, H)
        # posterior mean
        self.mu = Parameter(torch.zeros(size=(self.N, self.H)))
        # posterior covariance lower triangular parameters
        self.Sigma_param = Parameter(torch.ones(size=(self.N, self.H)))
        

    def forward(self, X, indexes=None):
        if indexes is None:
            L = self.Sigma_param
            Sigma = L**2
            return self.mu, L, Sigma
        else:
            L = self.Sigma_param[indexes]
            Sigma = L**2
            return self.mu[indexes], L, Sigma


    def set(self, indexesto, other, indexesfrom):
        with torch.no_grad():
            self.mu[indexesto] = other.mu[indexesfrom]
            self.Sigma_param[indexesto] = other.Sigma_param[indexesfrom]
    

class AmortizedVariationalParams(VariationalParams):
    pass


class AmortizedDiagCovarGaussianVariationalParams(AmortizedVariationalParams):
    def __init__(self, N, D, H, rank=5) -> None:
        super().__init__(N, D, H)
        self.rank = rank
        nnSize = (self.D, 2*self.D, 4*self.D, self.H)
        
        # Shared NN
        self.nnShared = nn.Sequential(
                nn.Linear(self.D, nnSize[0]), 
                nn.ReLU(),
                )

        # NN for means
        self.nnMeans = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU(),
                nn.Linear(nnSize[2], nnSize[3]), 
                )
        
        # NN for diagonal covars
        self.nnDiagCovars = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU(),
                nn.Linear(nnSize[2], nnSize[3]), 
                #nn.Softplus(),
                )
        
        

    def forward(self, X, indexes):
        shared = self.nnShared(X)
        mu = self.nnMeans(shared)
        L = self.nnDiagCovars(shared) + 0.001
        Sigma = L**2
        return mu, L, Sigma
    


class AmortizedGaussianVariationalParams(AmortizedVariationalParams):
    def __init__(self, N, D, H, rank=5) -> None:
        super().__init__(N, D, H)
        self.rank = rank
        nnSize = (self.D, 10*self.D, 10*self.D, self.H)
        
        # NN for means
        self.nnMeans = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU6(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU6(),
                nn.Linear(nnSize[2], nnSize[3]), 
                )
        
        # NN for diagonal covars
        self.nnDiagCovars = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU6(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU6(),
                nn.Linear(nnSize[2], nnSize[3]), 
                )
        
        # NN for low-rank covar
        self.nnLowRankCovar = nn.Sequential(
                nn.Linear(nnSize[0], nnSize[1]), 
                nn.ReLU6(),
                nn.Linear(nnSize[1], nnSize[2]), 
                nn.ReLU6(),
                nn.Linear(nnSize[2], rank*nnSize[3]), 
                )
        

    def forward(self, X, indexes):
        mu = self.nnMeans(X)
        V = self.nnLowRankCovar(X).reshape((X.shape[0], self.H, -1))
        Sigma = torch.bmm(V, V.transpose(-1, -2)) + torch.diag_embed(self.nnDiagCovars(X)**2)
        L = cholesky_jitter(Sigma)
        return mu, L, Sigma



class ResBlock(nn.Module):
    def __init__(self, auto_block=None, channels=None, mid_channels=None):
        nn.Module.__init__(self)

        self.auto_block = auto_block
        if self.auto_block is None:
            assert (channels is not None) and (mid_channels is not None)
            self.auto_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(channels, mid_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.auto_block(x)
    

class AmortizedResNetVariationalParams(AmortizedVariationalParams):
    def __init__(self, N, D, H, minsigma=1e-3) -> None:
        super().__init__(N, D, H)
        self.minsigma = minsigma
        self.Dsqrt = int(math.sqrt(D))
        
        self.nn_common = nn.Sequential(
            nn.Linear(D, 2*D),
            ResBlock(nn.Sequential(
                nn.ReLU(),
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
            )),
            ResBlock(nn.Sequential(
                nn.ReLU(),
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
            )),
        )

        self.nn_mean = nn.Linear(2*D, H)
        self.nn_covar_param = nn.Linear(2*D, H)

        
    def forward(self, X, indexes):
        common = self.nn_common(X)
        mu = self.nn_mean(common)
        L = torch.nn.Softplus()(self.nn_covar_param(common)) + self.minsigma
        Sigma = L**2
        return mu, L, Sigma
    

class AmortizedResNetTwoHeadsVariationalParams(AmortizedVariationalParams):
    def __init__(self, N, D, H, minsigma=1e-3) -> None:
        super().__init__(N, D, H)
        self.minsigma = minsigma
        self.Dsqrt = int(math.sqrt(D))
        
        self.nn_common = nn.Sequential(
            nn.Linear(D, 2*D),
            nn.ReLU(),
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
        )

        self.nn_mean = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
            nn.Linear(2*D, H),
        )

        self.nn_covar_param = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
            nn.Linear(2*D, H),
        )

        
    def forward(self, X, indexes):
        common = self.nn_common(X)
        mu = self.nn_mean(common)
        L = torch.nn.Softplus()(self.nn_covar_param(common)) + self.minsigma
        Sigma = L**2
        return mu, L, Sigma
        

class AmortizedResNetLowRankVariationalParams(AmortizedVariationalParams):
    def __init__(self, N, D, H, rank=5, minsigma=1e-3) -> None:
        super().__init__(N, D, H)
        self.rank = rank
        self.minsigma = minsigma
        
        self.nn_common = nn.Sequential(
            nn.Linear(D, 2*D),
            nn.ReLU(),
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
        )

        self.nn_mean = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
            nn.Linear(2*D, H),
        )

        self.nn_diag_covar = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
            nn.Linear(2*D, H),
        )

        self.nn_low_rank_param = nn.Sequential(
            ResBlock(nn.Sequential(
                nn.Linear(2*D, 3*D),
                nn.ReLU(),
                nn.Linear(3*D, 2*D),
                nn.ReLU(),
            )),
            nn.Linear(2*D, rank*H),
        )

        self.nn_mean = nn.Linear(2*D, H)
        self.nn_diag_covar = nn.Linear(2*D, H)

        
    def forward(self, X, indexes):
        common = self.nn_common(X)
        mu = self.nn_mean(common)
        V = self.nn_low_rank_param(common).reshape((X.shape[0], self.H, -1))
        Sigma = torch.bmm(V, V.transpose(-1, -2)) + torch.diag_embed((self.nn_diag_covar(common)+self.minsigma)**2)
        L = cholesky_jitter(Sigma)
        return mu, L, Sigma


class LinearLVModel(Module):
    def __init__(self, N, D, H, variationalparams=None) -> None:
        super().__init__()
        self.N = N  # number of data points
        self.D = D  # observed space dimensionality
        self.H = H  # latent space dimensionality

        # model parameters
        self.W = Parameter(torch.normal(0, 1, size=(self.D, self.H)))
        self.normalize_W()
        
        # variational Gaussian params
        if variationalparams is None:
            variationalparams = FullCovarGaussianVariationalParams(N, D, H)
        self.variationalparams = variationalparams
        

    @property
    def device(self):
        return self.W.device
    

    def normalize_W(self):
        with torch.no_grad():
            self.W.copy_(self.W / torch.sqrt((self.W**2).sum(0)))
    

    def sample_posterior(self, X, indexes, nsamples, only_means=False):
        mu, L, Sigma = self.variationalparams(X, indexes)
        if only_means:
            return mu
        
        # Draw nsamples from the posterior for every data point
        rn = torch.randn(size=(nsamples, X.shape[0], self.H), device=X.device)
        if L.ndim == 2:
            z_samples = mu + L * rn
        else:
            assert L.ndim == 3
            z_samples = mu + torch.matmul(L, rn.unsqueeze(-1)).squeeze(-1)
        return z_samples, L, Sigma

    
    def sample_data(self, z_samples, only_means=False):
        xstar = torch.matmul(self.make_W(), z_samples.unsqueeze(-1)).squeeze(-1)
        if only_means:
            return xstar
        
        # Draw noisy data samples
        sigmasqr = self.make_sigmasqr()
        x_samples = xstar + torch.normal(0, torch.sqrt(sigmasqr), size=xstar.shape)
        return x_samples
    
        
