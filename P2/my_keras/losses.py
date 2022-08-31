import numpy as np
import metrics


class Loss():
    def __call__():
        return None

    def gradient():
        return None

class MSE(Loss):
    def __call__(ypred, ytrue):
        return metrics.MSE(ypred, ytrue)

    def gradient(ypred, ytrue):
        m = ypred.shape[0]
        return (ypred - ytrue)/m