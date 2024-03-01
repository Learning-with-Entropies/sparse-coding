""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import os
from datetime import datetime
import random
import math
import numpy as np
import torch
from datasets.bars import BarsDataset
from datasets.images import OlshausenDataset
from datasets.utils import to_bases, to_images
from models.optimalmanifoldELBO import *
from models.stochasticELBO import StochasticSC, LinearGaussianModel
from models.analyticELBO import AnalyticELBOSC
from utils.training import train_Adam, train_LBFGS
from utils.training import optimize_latents, adjust_parameters
from utils.training import CallbackList
import utils.plotting as plu
from utils.statistics import posterior_sparseness

import matplotlib.pyplot as plt
import utils.trochjson as tj
from tqdm import tqdm

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#device = "cpu"

# Uncomment to debug NANs
#torch.autograd.set_detect_anomaly(True)



if False:
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


### Load a dataset

dataset = OlshausenDataset(N=200*1024)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4*128)
N = dataset.N
D = dataset.D
H = 10**2  # number of sources

if dataloader.batch_size < H:
    print("Warning: dataloader.batch_size < H")

### Construct a model

likelihood_entropies = []
prior_entropies = []
proposal_entropies = []
def on_compute_entropy(likelihood_entropy, prior_entropy, proposal_entropy):
    likelihood_entropies.append(likelihood_entropy.item())
    prior_entropies.append(prior_entropy.item())
    proposal_entropies.append(proposal_entropy.item())

modeltype = 1
if modeltype == 1:
    model = OptimalManifoldSC(N, D, H, 
        variationalparams=DiagCovarGaussianVariationalParams(N, D, H), 
        #variationalparams=AmortizedResNetVariationalParams(N, D, H), 
        on_compute_entropy=on_compute_entropy, constrain_W=True).to(device)
elif modeltype == 2:
    model = AnalyticELBOSC(N, D, H, 
            variationalparams=AmortizedGaussianVariationalParams(N, D, H)).to(device)
elif modeltype == 3:
    model = StochasticSC(N, D, H, 
            variationalparams=AmortizedDiagCovarGaussianVariationalParams(N, D, H), nsamples=10).to(device)
elif modeltype == 4:
    model = LinearGaussianModel(N, D, H, 
            variationalparams=AmortizedDiagCovarGaussianVariationalParams(N, D, H), nsamples=1).to(device)

### Prepare the output directory
path = "./out/images/" + model.__class__.__name__ + "/" + datetime.now().strftime('%y.%m.%d-%H:%M:%S') + "/"
if not os.path.exists(path):
    os.makedirs(path)
prefix = path
os.system("cp {} {}".format(__file__, path))

### Plotting routines
test_x = []
test_indexes = []
for patch, index in dataset:
    test_x.append(patch)
    test_indexes.append(index)
    if index > 1000:
        break
test_x = torch.tensor(np.stack(test_x, axis=0))
test_indexes = torch.tensor(test_indexes)
def plot_training_progress(model, epoch, elbos=None):
    alphas = compute_alpha(model, dataset)
    plu.plot(elbos, filename=prefix+"elbos.pdf")
    plu.plot_W(model, filename=prefix + "Ws-{}.png".format(epoch), order=alphas)
    plu.plot_samples(model, test_x[:10], test_indexes[:10], filename=prefix + "samples-{}.pdf".format(epoch))

### Training

model.normalize_W()

plot_training_progress(model, 0)

epoch_elbos = []
epoch_likelihood_entropies = []
epoch_prior_entropies = []
epoch_proposal_entropies = []
epoch_gini = []

def on_after_epoch(model, epoch, elbos=None):
    elbos = -np.array(likelihood_entropies)-np.array(prior_entropies)+np.array(proposal_entropies)
    plot_training_progress(model, epoch, elbos)
    epoch_gini.append(posterior_sparseness(model, test_x, test_indexes))

    # Compute ELBO for this epoch
    print("Computing epoch ELBO")
    batch_likelihood_entropies = []
    batch_prior_entropies = []
    batch_proposal_entropies = []
    
    def collect_elbo(likelihood_entropy, prior_entropy, proposal_entropy):
        batch_likelihood_entropies.append(likelihood_entropy.item())
        batch_prior_entropies.append(prior_entropy.item())
        batch_proposal_entropies.append(proposal_entropy.item())

    oce = model.on_compute_entropy
    model.on_compute_entropy = collect_elbo
    for batch_idx, (data, indexes) in tqdm(enumerate(dataloader)):
        data = data.type(torch.get_default_dtype()).to(model.device)
        if isinstance(model.variationalparams, AmortizedVariationalParams):
            elbo = model(data, indexes)
        else:
            elbo = optimize_latents(model, data, indexes, init=True, max_iter=100)
    model.on_compute_entropy = oce

    batch_likelihood_entropies = np.array(batch_likelihood_entropies)
    batch_prior_entropies = np.array(batch_prior_entropies)
    batch_proposal_entropies = np.array(batch_proposal_entropies)

    batch_elbos = -batch_likelihood_entropies -batch_prior_entropies +batch_proposal_entropies
    epoch_elbos.append(np.ma.masked_invalid(np.array(batch_elbos)).mean())
    plu.plot(batch_elbos, prefix+"batch-elbos-{}.pdf".format(epoch))
    plu.plot(epoch_elbos, prefix+"epoch-elbos.pdf")

    epoch_likelihood_entropies.append(np.ma.masked_invalid(np.array(batch_likelihood_entropies)).mean())
    epoch_prior_entropies.append(np.ma.masked_invalid(np.array(batch_prior_entropies)).mean())
    epoch_proposal_entropies.append(np.ma.masked_invalid(np.array(batch_proposal_entropies)).mean())
    plu.plot(epoch_likelihood_entropies, filename=prefix + "epoch_likelihood_entropies.pdf")
    plu.plot(epoch_prior_entropies, filename=prefix + "epoch_prior_entropies.pdf")
    plu.plot(epoch_proposal_entropies, filename=prefix + "epoch_proposal_entropies.pdf")
    plu.plot_errorbar(epoch_gini, filename=prefix + "epoch_gini.pdf")
    
if isinstance(model.variationalparams, AmortizedVariationalParams):
    # Gradient-based optimization
    epoch = 0
    model.prior_scale = 1
    nepochs = 200

    if isinstance(model, (OptimalManifoldSC, AnalyticELBOSC)):
        def set_annealing(model, epoch, *args, **kwds):
            model.prior_scale = max(1.0, 2-0.1*epoch)

        elbos, gradsizes, gradtotalvars = train_Adam(model, dataloader, lr=1e-3, 
                                                        nepochs=nepochs, 
                                                        on_epoch_finish=CallbackList((on_after_epoch, set_annealing)))
        plu.plot(gradsizes, prefix + "gradsizes.pdf")
        plu.plot_entropies(likelihood_entropies, 
                        prior_entropies, 
                        proposal_entropies, 
                        prefix + "entropies.pdf")
        
    elif isinstance(model, (StochasticSC, LinearGaussianModel)):
        elbos, gradmeansizes, gradtotalvars = train_Adam(model, dataloader, lr=1e-3, 
                                                            nepochs=nepochs, 
                                                            on_epoch_finish=plot_training_progress)
        plu.plot(gradmeansizes, prefix + "gradmeansizes.pdf")
        plu.plot(gradtotalvars, prefix + "gradtotalvars.pdf")
    
else:
    # EM-like optimization
    for epoch in range(10):
        print("Optimizing ELBO")
        for batch_idx, (data, indexes) in enumerate(dataloader):
            model.prior_scale = max(1.0, 2*(5-epoch))  # prior annealing
            #model.likelihood_scale = ([-(1/(x-7)) for x in range(6)] + [1.0]*4)[epoch]  # beta-VAE
            data = data.type(torch.get_default_dtype()).to(model.device)
            elbo = optimize_latents(model, data, indexes, max_iter=100)
            elbo = adjust_parameters(model, data, indexes, lr=0.5)
            if not model.constrain_W:
                model.normalize_W()
            print("Epoch {:4d} \t Batch {:4d} \t Scale: {:.2f} \t ELBO: {:.6f}".format(epoch+1, batch_idx, model.prior_scale, elbo))
        on_after_epoch(model, epoch+1)

#torch.save(model, prefix+"model.pt")    



