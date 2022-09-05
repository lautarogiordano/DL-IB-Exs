from .layers2 import ConcatInput, Input, Dense
from .metrics import acc_XOR
import numpy as np


class Network():
    def __init__(self):
        self.layers = []
        self.n_layers = 0

    def add(self, Layer, scale=1e-2):
        self.layers.append(Layer)
        self.n_layers += 1
        

    def getLayer(self, n):
        return self.layers[n]

    def printLayers(self):
        for layer in range(self.n_layers):
            print("Layer {}: {}, neurons: {}".format(layer, type(self.layers[layer]),
                                                     self.layers[layer].n_neurons))
    
    def forward(self, Xb, up_to=None):
        if up_to is None:
            up_to = self.n_layers

        S = None #S es el ultimo output del forward pass
        S = self.layers[0].forward(Xb)

        if up_to > 1:
            for layer in range(1, up_to):
                if isinstance(self.layers[layer], ConcatInput):
                    S = self.layers[layer].forward(Xb, S)
                else:
                    S = self.layers[layer].forward(S)
        
        return S

    def regTerm(self):
        reg = 0
        for layer in range(self.n_layers):
            if isinstance(self.layers[layer], Dense):
                reg += self.layers[layer].getReg()
        return reg

    def backward(self, Xb, yb_pred, yb, loss):
        current = self.n_layers - 1
        
        assert yb_pred.shape[0] == yb.shape[0], "Hay un problemita con el forward pass."
        
        loss_b = loss(yb_pred, yb) + self.regTerm()
        
        grad = loss.gradient(yb_pred, yb)
    
        while current >= 0:
            #Update weights hace todo (dentro de layers): 
            # 1- Multiplica el grad entrante por 
            # el gradiente de la funcion de activacion.
            # 2- Si la layer tiene pesos la multiplico por el gradiente y 
            # actualizo los pesos. 
            # 3- Le saco los indices de los bias o concat si es necesario
            # 4- El gradiente que queda se lo asigno a grad y sigo para atrás.
            # 5- current=0 la input layer solo calcula gradientes respecto a W1.

            if isinstance(self.layers[current], ConcatInput):
                grad = self.layers[current].updateWeights(grad)
                
            elif isinstance(self.layers[current], Dense): 
                #Para capas densas actualizo los pesos
                if current == 1:
                    last = np.copy(Xb)
                else:
                    last = self.layers[current-1]()
                grad = self.layers[current].updateWeights(grad, last, self.opt.updateRule)
            
            current -= 1

        return loss_b

    def fit(self, X, y, loss, opt, metric, testdata = None, epochs=10, batch_size=None):
        self.opt = opt
        loss_hist = []

        for epoch in range(epochs):
            loss_hist.append(self.opt(X, y, self, loss, batch_size))
            if epochs > 10 and epoch % int(epochs/10) == 0:
                if testdata is not None:
                    acc_test = 100*metric(self.predict(testdata[0]), testdata[1])
                    print("Epoch {}, Train loss: {:.4f}\n Accuracy test: {:.2f}".format(epoch, loss_hist[-1], acc_test))
                else:
                    print("Epoch {}, Train loss: {:.4f}".format(epoch, loss_hist[-1]))
        
        return loss_hist



    def predict(self, X):
        return self.forward(X)
