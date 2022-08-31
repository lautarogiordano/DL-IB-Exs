import numpy as np
from scipy import linalg

def MSE(ypred, ytrue):
    m = ypred.shape[0]
    return linalg.norm(ypred - ytrue)**2 / (2*m)

def accuracy(ypred, ytrue):
    return np.mean(ypred==ytrue)

def acc_XOR(ypred, ytrue, umbral=.8):
    ypred[ypred>umbral] = 1
    ypred[ypred<-umbral] = -1

    return np.mean(ypred==ytrue)