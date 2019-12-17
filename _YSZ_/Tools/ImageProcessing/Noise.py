import torch
import numpy as np
from scipy import sparse


class Noise:
    def __init__(self, noiseType="rand", epsilon=0.2):
        self.noiseType = noiseType
        self.epsilon = epsilon

    def get_noise(self, size):
        if self.noiseType == "rand":
            noise = torch.rand(size, out=None)
        else:
            raise RuntimeError('noise type not defined')
        return noise

    def add_noise(self, X, decay):
        noise = self.get_noise(X.size())
        X = X + decay * noise
        return X
