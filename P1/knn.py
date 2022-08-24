import numpy as np
from scipy import linalg
from collections import Counter

class KNN:
    def __init__(self):
        pass

    def loadData(self, X, y):
        self.X = X.reshape(len(X), X[0].size).astype(np.int16)
        self.m = X.shape[0]
        self.y = y
        self.idx = []

    def predict(self, X, k):
        self.size = X.shape[0]
        self.y_guess = np.zeros((self.size,))
        self.norm = np.zeros((self.m,))
        
        X = X.reshape(len(X), X[0].size).astype(np.int16)
        
        for i in range(self.size):
            for j in range(self.m):
                #norm es un array con la distancia de la foto i a cada ejemplo de train j
                self.norm[j] = linalg.norm(self.X[j,:] - X[i,:])

            #Voy a obtener los índices de las k distancias más cercanas 
            self.idx = np.argpartition(self.norm, k)[:k]
            labels = np.array(self.y[self.idx])
            most_common = Counter(labels).most_common(1)  
            self.y_guess[i] = most_common[0][0]
            
        return self.y_guess

    def accuracy(self, yguess, ytrue):
        return float(np.sum(ytrue==yguess))/float(len(ytrue))
