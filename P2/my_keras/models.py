import numpy as np
import layers
import losses
import optimizers
import regularizers

class Network():
    def __init__(self):
        self.layers = []
        self.n_layers = 0

    def add(self, layer=layers.Dense):
        #Inicializo una nueva capa
        #self.layers.append(algo)
        self.n_layers += 1
        pass

    def getLayer(self, n):
        return self.layers[n]
    
    def forward(self, j=None):
        if j is None:
            j = self.n_layers
        pass

    def backward(self, Xb, yb, loss):
        current = self.n_layers
        #yb_pred es el resultado de hacer el forward con el batch
        yb_pred = self.layers[-1]()

        grad = loss.grad(yb_pred, yb)

        while current >= 0:
            #Update weights hace todo (dentro de layers): 
            # 1- Multiplica el grad entrante por 
            # el gradiente de la funcion de activacion.
            # 2- Si la layer tiene pesos la multiplico por el gradiente y 
            # actualizo los pesos. 
            # 3- Le saco los indices de los bias o concat si es necesario
            # 4- El gradiente que queda se lo asigno a grad y sigo para atrás.
            # 5- current=0 la input layer solo calcula gradientes respecto a W1.
            grad = self.layers[current].updateWeights(grad)


    def fit(X, y, epochs=10, batch_size=None, 
            loss=losses.MSE(), opt=optimizers.SGD(alpha = 1e-3)):
        #Loop en epocas
        #LLama al optimizador y hace backward
        pass

    def predict(self, X):
        return self.forward()
