import numpy as np
from scipy.special import expit

class ActivationFunc():
    def __call__():
        return None

    def prime():
        return None

class RELU(ActivationFunc):
    def __call__(z):
        return np.maximum(0, z)

    def prime(z):
        return 1*(RELU(z).astype(bool))

class sigmoid(ActivationFunc):
    def __call__(z):
        return expit(z)

    def prime(z):
        return sigmoid(z)*(1-sigmoid(z))

class tanh(ActivationFunc):
    def __call__(z):
        return np.tanh(z)

    def prime(z):
        return 1.-np.tanh(z)**2