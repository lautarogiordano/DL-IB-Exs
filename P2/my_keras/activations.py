import numpy as np
from scipy.special import expit

class ActivationFunc():
    def __call__(self):
        return None

    def prime(self):
        return None

class RELU(ActivationFunc):
    def __call__(self, z):
        return np.maximum(0, z)

    def prime(self, z):
        return 1*(self(z).astype(bool))

class sigmoid(ActivationFunc):
    def __call__(self, z):
        return expit(z)

    def prime(self, z):
        return expit(z)*(1-expit(z))

class tanh(ActivationFunc):
    def __call__(self, z):
        return np.tanh(z)

    def prime(self, z):
        return 1 -self(z)**2

class linear(ActivationFunc):
    def __call__(self, z):
        return z

    def prime(self, z):
        return 1