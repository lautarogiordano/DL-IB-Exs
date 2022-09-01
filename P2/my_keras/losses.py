import numpy as np
from .metrics import MSE as MSE_fun

class Loss():
    def __init__(self):
        return None

    def __call__(self):
        return None

    def gradient(self):
        return None

class MSE_XOR(Loss):
    def __init__(self):
        return None

    def __call__(self, ypred, ytrue):
        return MSE_fun(ypred, ytrue)

    def gradient(self, ypred, ytrue):
        m = ypred.shape[0]
        self.grad = (ypred - ytrue)/m
        return self.grad