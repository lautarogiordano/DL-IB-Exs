import numpy as np
from scipy import linalg

def MSE(ypred, ytrue):
    m = ypred.shape[0]
    return linalg.norm(ypred - ytrue)**2 / (2*m)

def MSE_i(yb_pred, yb):
    diff = np.copy(yb_pred)
    m = diff.shape[0]
    idx = np.arange(0, m)
    diff[idx, yb[:, 0]] -= 1

    return (linalg.norm(diff)**2 / (2*m))


def acc_img(ypred, ytrue):
    ypreds = np.argmax(ypred, axis=1)
    return np.mean(ypreds==ytrue[:,0])

def acc_XOR(ypred, ytrue, umbral=.8):
    ypred[ypred>umbral] = 1
    ypred[ypred<-umbral] = -1

    return np.mean(ypred==ytrue)