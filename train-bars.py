""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets.bars import BarsDataset
from models.optimalmanifoldELBO import OptimalManifoldSC, DiagCovarGaussianVariationalParams, FullCovarGaussianVariationalParams, AmortizedGaussianVariationalParams
from models.stochasticELBO import StochasticSC
from models.analyticELBO import AnalyticELBOSC
from utils.training import *
from utils.plotting import *

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#device = "cpu"


if False:
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


N = 1000
H = 2*5  # number of sources
D = int(H / 2)**2  # data dimensionality
dataset = BarsDataset(N, D, H, latent="Laplace")
#H += 2  # additional letent dimensions

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)

likelihood_entropies = []
prior_entropies = []
proposal_entropies = []
def on_compute_entropy(likelihood_entropy, prior_entropy, proposal_entropy):
    likelihood_entropies.append(likelihood_entropy.item())
    prior_entropies.append(prior_entropy.item())
    proposal_entropies.append(proposal_entropy.item())

model = OptimalManifoldSC(N, D, H, 
        variationalparams=FullCovarGaussianVariationalParams(N, D, H), 
        on_compute_entropy=on_compute_entropy).to(device)

path = "./out/bars/" + model.__class__.__name__ + "/" + datetime.now().strftime('%y.%m.%d-%H:%M:%S') + "/"
if not os.path.exists(path):
    os.makedirs(path)
prefix = path

plot_training_data(dataset.x[:H], filename=prefix + "training-data.pdf")

def save_plots(model, epoch, elbos=None):
    if epoch % 10 == 0:
        plot(elbos, filename=prefix + "elbos.pdf")
        plot_Ws(model, filename=prefix + "Ws-{:05}.pdf".format(epoch))
    
    if epoch % 10 == 0:
        indexes = range(10)
        x = dataset.x[indexes]
        plot_samples(model, x, indexes, filename=prefix + "X-{:05}.pdf".format(epoch))

save_plots(model, epoch=0)
if isinstance(model, (OptimalManifoldSC, AnalyticELBOSC)):
    model.prior_scale = 1
    elbos, gradsizes = train_LBFGS(model, dataloader, nepochs=1000, on_epoch=save_plots, 
        history_size=200, 
        max_iter=200)
    save_plots(model, 10000)
    plt.plot(gradsizes)
    plt.gca().set_yscale("log")
    plt.title("gradient size")
    plt.savefig(prefix + "gradsizes.pdf")
    plt.close()

    # Set ground-truth W and compute max ELBO
    with torch.no_grad():
        model.W.copy_(torch.Tensor(dataset.gen.W))
    data = dataset.x.type(torch.get_default_dtype()).to(model.device)
    gt_elbo = optimize_latents(model, data, indexes=range(len(data)), init=False, max_iter=200)
    print("Ground truth ELBO: {}".format(gt_elbo))
    np.savetxt(prefix + "groundtruth-elbo.txt", np.array([gt_elbo]), fmt="%f", delimiter=",")


