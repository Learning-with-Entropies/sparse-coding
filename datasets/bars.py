""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import to_bases



class BarsDataGenerator():
    def __init__(self, N, D, H, latent=None) -> None:
        self.N = N  # number of data points
        self.D = D  # observed space dimensionality
        self.H = H  # latent space dimensionality
        if latent is None:
            latent = "Bernoulli"
        assert latent == "Bernoulli" or latent == "Laplace"
        self.latent = latent
        self.pi = 0.2  # source activation probability
        self.sigma = 0.1  # observation noise (scale)
        
        self.W = to_bases(self.make_bars_images())

    def make_bars_images(self):
        img = []
        M = int(self.H / 2)
        for i in range(M):
            gf = np.zeros((M, M))
            gf[:, i] = 1
            img.append(gf)
        for i in range(M):
            gf = np.zeros((M, M))
            gf[i, :] = 1
            img.append(gf)
        images = np.stack(img, axis=0)
        return images

    def sample_data(self, N):
        if self.latent == "Bernoulli":
            s = np.array(np.random.uniform(size=(N, self.H)) < self.pi, dtype=int)
        if self.latent == "Laplace":
            s = np.random.laplace(size=(N, self.H))
        x = np.dot(self.W, s.T).T
        x += np.random.normal(scale=self.sigma, size=x.shape)
        return s, x


class BarsDataset(Dataset):
    def __init__(self, N, D, H, latent=None) -> None:
        super().__init__()
        self.gen = BarsDataGenerator(N, D, H, latent)
        self.s, x = self.gen.sample_data(N)
        self.x = torch.tensor(x)

    def __len__(self):
        return self.gen.N

    def __getitem__(self, idx):
        return self.x[idx], idx

    def to(self, device):
        self.x = self.x.to(device)
        return self
    

