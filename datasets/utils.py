""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

import numpy as np



def to_bases(images):
    N, D1, D2 = images.shape
    return images.reshape((N, D1 * D2)).T


def to_images(W):
    # W: D*H
    sqrtD = int(np.sqrt(W.shape[0]))
    return W.T.reshape((-1, sqrtD, sqrtD))