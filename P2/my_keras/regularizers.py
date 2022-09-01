import numpy as np
from scipy import linalg

class Regularizer():
    def __init__(self):
        return None

    def __call__(self):
        return None
    
    def grad(self):
        return None

class L2(Regularizer):
    def __init__(self, lambda_=1e-2):
        self.lambda_ = lambda_

    def __call__(self, W):
        return (self.lambda_/2) * linalg.norm(W)**2

    def grad(self, W):
        return self.lambda_ * W

class L1(Regularizer):
    def __init__(self, lambda_=1e-2):
        self.lambda_ = lambda_
    
    def __call__(self, W):
        return (self.lambda_) * np.sum(np.abs(W))

    def grad(self, W):
        return self.lambda_ * np.sign(W)
