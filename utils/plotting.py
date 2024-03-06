""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import math
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from datasets.utils import to_bases, to_images



def plot(x, filename, title=None):
    if x is not None:
        x = np.asarray(x)
        np.savetxt(filename[:-4]+".txt", x, fmt="%f", delimiter=",")
        plt.plot(x)
        if title is not None:
            plt.title(title)
        plt.savefig(filename)
        plt.close()


def plot_errorbar(x, filename, title=None):
    if x is not None:
        x = np.asarray(x)
        np.savetxt(filename[:-4]+".txt", x, fmt="%f", delimiter=",")
        plt.plot(x[:, 0])
        plt.errorbar(range(x.shape[0]), x[:, 0], yerr=x[:, 1])
        if title is not None:
            plt.title(title)
        plt.savefig(filename)
        plt.close()


def plot_elbos(elbos, filename):
    if elbos is not None:
        elbos = np.asarray(elbos)
        np.savetxt(filename[:-4]+".txt", elbos, fmt="%f", delimiter=",")
        plt.plot(elbos)
        plt.title("ELBO")
        plt.savefig(filename)
        plt.close()


def plot_entropies(likelihood_entropies, prior_entropies, proposal_entropies, filename):

    likelihood_entropies = np.array(likelihood_entropies)
    prior_entropies = np.array(prior_entropies)
    proposal_entropies = np.array(proposal_entropies)
    
    np.savetxt(filename[:-4]+".txt", 
               np.stack((likelihood_entropies, 
                         prior_entropies, 
                         proposal_entropies)), 
               fmt="%f", delimiter=",")
        
    plt.plot(-np.array(likelihood_entropies), label="-H(likelihood)")
    plt.plot(-np.array(prior_entropies), label="-H(prior)")
    plt.plot(np.array(proposal_entropies), label="H(proposal)")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


def plot_Ws(model, filename=None):
    W = model.make_W().to("cpu").detach().numpy()
    images = to_images(W)
    d = int(len(images)/2)
    fig, axs = plt.subplots(2, d, figsize=(d, 2))
    cmap = matplotlib.colormaps["gray"]
    normalizer = Normalize(-np.max(np.abs(W)), np.max(np.abs(W)))
    im = cm.ScalarMappable(norm=normalizer)
    for ax, img in zip(axs.flat, images):
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.imshow(img, cmap=cmap,norm=normalizer)
    if filename is not None:
        fig.savefig(filename)
        plt.close()


def plot_training_data(x, filename=None):
    x = x.T.to("cpu").detach().numpy()
    images = to_images(x)
    d = int(len(images)/2)
    fig, axs = plt.subplots(2, d, figsize=(d, 2))
    cmap = matplotlib.colormaps["gray"]
    normalizer = Normalize(-np.max(np.abs(x)), np.max(np.abs(x)))
    im = cm.ScalarMappable(norm=normalizer)
    for ax, img in zip(axs.flat, images):
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.imshow(img, cmap=cmap,norm=normalizer)
    if filename is not None:
        fig.savefig(filename)
        plt.close()


def plot_W(model, filename=None, order=None):
    W = model.make_W().to("cpu").detach().numpy()
    if filename is not None:
        np.savetxt(filename[:-4]+".txt", W, fmt="%f", delimiter=",")
    images = to_images(W)
    d = int(math.sqrt(len(images)))

    if order is not None:    
        if filename is not None:
            np.savetxt(filename[:-4]+"-order.txt", order, fmt="%f", delimiter=",")
        order = np.argsort(order)
        images = images[np.flip(order)]

    fig, axs = plt.subplots(d, d, figsize=(d, d))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    cmap = matplotlib.colormaps["gray"]
    normalizer = Normalize(-np.max(np.abs(W)), np.max(np.abs(W)))
    for ax, img in zip(axs.flat, images):
        ax.imshow(img, cmap=cmap, norm=normalizer)
        ax.set_axis_off()
    if filename is not None:
        fig.savefig(filename)
        plt.close()


def plot_samples(model, x, indexes, filename, nsamples=10):
    # Make data and reconstruction plots
    ndatapoints = x.shape[0]
    # Draw nsamples from the posterior for every data point
    z_samples, L, Sigma = model.sample_posterior(x.type(torch.get_default_dtype()).to(model.device), 
                                                 indexes, nsamples=nsamples)
    # Generate X
    xstar = model.sample_data(z_samples, only_means=True).to("cpu").detach().numpy()
    
    ximages = to_images(x.T)
    fig, axs = plt.subplots(ndatapoints, nsamples+1, figsize=(nsamples+1, ndatapoints))
    cmap = matplotlib.colormaps["gray"]
    amplitude = x.abs().max().item()
    normalizer = Normalize(-amplitude, amplitude)    
    for i in range(ndatapoints):
        xstarimages = to_images(xstar[:, i, :].T)
        axs[i, 0].imshow(ximages[i], cmap=cmap, norm=normalizer)
        axs[i, 0].set_axis_off()
        for j in range(nsamples):
            axs[i, j+1].imshow(xstarimages[j], cmap=cmap, norm=normalizer)
            axs[i, j+1].set_axis_off()
    fig.savefig(filename)
    plt.close()


def plot_patches(patches, filename, size):
    assert len(patches) == size[0] * size[1]
    
    fig, axs = plt.subplots(size[0], size[1], figsize=(size[1], size[0]))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    cmap = matplotlib.colormaps["gray"]
    amplitude = np.max(np.abs(patches))
    normalizer = Normalize(-amplitude, amplitude)
    for ax, img in zip(axs.flat, patches):
        d = int(math.sqrt(len(img)))
        ax.imshow(img.reshape(d, d), cmap=cmap, norm=normalizer)
        ax.set_axis_off()
    fig.savefig(filename)
    plt.close()
