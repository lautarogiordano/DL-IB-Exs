import numpy as np
from .losses import MSE_XOR

class Optimizers():
    def __init__(self, alpha=1e-5):
        self.alpha = alpha
    
    def __call__(self, X, y, model):
        return None
    
    def updateWeights(self, W, gradW):
        return None

class SGD(Optimizers):
    def __init__(self, alpha=1e-5):
        super().__init__(alpha)

    
    def __call__(self, X, y, model, loss=MSE_XOR(), batch_size=None):
        m = X.shape[0]
        if batch_size is None or batch_size > m:
            batch_size = m
        n_batches = int(m/batch_size)
        shuffle = np.random.permutation(m)
        X = X[shuffle]
        y = y[shuffle]

        loss_epoch = 0

        for batch in range(n_batches):
            Xb = X[batch*batch_size:(batch+1)*batch_size, :]
            yb = y[batch*batch_size:(batch+1)*batch_size]

            yb_pred = model.forward(Xb)

            loss_epoch += model.backward(Xb, yb_pred, yb, loss)
        
        return loss_epoch/n_batches
    
    def updateRule(self, W, gradW, reg):
        return W - self.alpha * (gradW + reg.grad(W))