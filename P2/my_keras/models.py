from .layers import Input, Dense, LastLayer
import numpy as np


class Network():
    def __init__(self):
        self.layers = []
        self.n_layers = 0

    def add(self, Layer, scale=1e-2):
        self.layers.append(Layer)
        self.n_layers += 1
        #Creo la matriz de pesos de la layer anterior si es necesario
        if self.n_layers > 1:
            last_layer = self.layers[self.n_layers-2]
            if isinstance(last_layer, (Input, Dense)):
                n_out = Layer.n_neurons
                last_layer.setW(n_out, scale)

    def getLayer(self, n):
        return self.layers[n]

    def printLayers(self):
        for layer in range(self.n_layers):
            print("Layer {}: {}".format(layer, type(self.layers[layer])))
    
    def forward(self, Xb, up_to=None):
        if up_to is None:
            up_to = self.n_layers

        S = None #S es el ultimo output del forward pass
        S = self.layers[0].forward(Xb)
        if up_to > 1:
            for layer in range(1, up_to):
                S = self.layers[layer].forward(S)
        
        return S

    def regTerm(self):
        reg = 0
        for layer in range(self.n_layers):
            if isinstance(self.layers[layer], Dense):
                reg += self.layers[layer].getReg()
        return reg

    def backward(self, Xb, yb, loss):
        current = self.n_layers - 1
        #yb_pred es el resultado de hacer el forward con el batch
        yb_pred = self.layers[-1]()
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
            if isinstance(self.layers[current], LastLayer):
                grad = self.layers[current].getGrad(grad)
            if isinstance(self.layers[current], Dense): 
                #Para capas densas o de input actualizo los pesos
                grad = self.layers[current].updateWeights(grad, self.opt.updateRule)
            if isinstance(self.layers[current], Input):
                #Para el gradiente del input necesito pasarle los datos de entrada
                grad = self.layers[current].updateWeights(Xb, grad, self.opt.updateRule)
            current -= 1

        return loss_b

    def fit(self, X, y, loss, opt, epochs=10, batch_size=None):
        self.opt = opt
        loss_hist = []

        for epoch in range(epochs):
            loss_hist.append(self.opt(X, y, self, loss, batch_size))
            if epoch % int(epochs/10) == 0:
                print("Epoch: {}, Train loss: {}".format(epoch, loss_hist[-1]))


    def predict(self, X):
        return self.forward(X)
