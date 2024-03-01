""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import torch
import numpy as np
from numpy.linalg import norm

def gini_coefficient(z):
    # Gini coefficient of absolute values in latent encodings
    z = np.abs(z)
    sorted_z = np.sort(z, axis=1)
    n = sorted_z.shape[-1]
    cumsums = np.cumsum(sorted_z, axis=-1)
    ginis = (n + 1 - 2 * np.sum(cumsums, axis=1) / cumsums[:, -1]) / n
    average_gini = np.mean(ginis)
    std = np.std(ginis)
    return average_gini, std


def posterior_sparseness(model, x, indexes, nsamples=10, type="Gini"):
     # Make data and reconstruction plots
    ndatapoints = x.shape[0]
    # Draw nsamples from the posterior for every data point
    z_samples, L, Sigma = model.sample_posterior(x.type(torch.get_default_dtype()).to(model.device), 
                                                 indexes, nsamples=nsamples)
    
    if type == "Gini":
        z = z_samples.cpu().detach().numpy()
        z = np.reshape(z, [-1, z.shape[-1]])
        return gini_coefficient(z)
    