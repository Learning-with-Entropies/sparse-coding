""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import math
import h5py
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, IterableDataset
import os


class VanHaterenDataset(Dataset):
    """ Pre-processed image patches derived from the original van Hateren dataset
    """
    def __init__(self, filename, N=1000000) -> None:
        super().__init__()
        self.filename = filename
        h5file = h5py.File(self.filename)
        self.x = h5file["wdata"][:N].astype(np.float32)
        self.N = self.x.shape[0]
        self.D = self.x.shape[-1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx], idx

    def to(self, device):
        self.x = self.x.to(device)
        return self



class OlshausenDataset(IterableDataset):
    """ This is the exact dataset Bruno Olshausen used in his paper on sparse coding.
        Matlab data files are taken from http://www.rctn.org/bruno/sparsenet/
    """
    def __init__(self, patchsize=16, N=1000000) -> None:
        super().__init__()
        self.patchsize = patchsize
        self.N = N
        self.D = patchsize**2
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "olshausen/IMAGES.mat")        
        mat_images = sio.loadmat(filename)
        images = mat_images['IMAGES']
        self.images = np.moveaxis(images, -1, 0)
        numimages, H, W = self.images.shape
        self.rx = np.random.randint(0, W-patchsize, N)
        self.ry = np.random.randint(0, H-patchsize, N)
        self.ri = np.random.randint(0, numimages, N)

    def __len__(self):
        return self.N

    def __iter__(self):
        for i in range(self.N):
            patch = self.images[self.ri[i], 
                                self.rx[i]:self.rx[i]+self.patchsize, 
                                self.ry[i]:self.ry[i]+self.patchsize]
            yield patch.flatten(), i


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = OlshausenDataset(N=100)
    print(ds.x.shape)
    plt.imshow(ds.images[0], cmap="gray")
    plt.show()

