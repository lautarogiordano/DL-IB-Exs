from lib2to3.pgen2.token import OP
import numpy as np

class Optimizers():
    def __init__(self, alpha=1e-5):
        self.alpha = alpha
    
    def __call__(self, X, y, model):
        return None
    
    def updateWeights(self, W, gradW):
        return None

class SGD(Optimizers):
    def __init__(self, batch_size, alpha=1e-5):
        super().__init__(alpha)
        self.batch_size = batch_size

    def getBatch(self, X):
        m = X.shape[0]
        n_batches = int(m/self.batch_size)
        i=0
        assert n_batches > 0, "Batch size bigger than dataset"

        while i < n_batches:
            yield X[i*self.batch_size:(i+1)*self.batch_size, :]
            i+=1
    
    def __call__(self, X, y, model):
        #Decidir si hacer batches
        #model.backward(Xb, yb)   #Le paso los batch
        pass
    
    def updateWeights(self, W, gradW):
        W -= self.alpha * gradW