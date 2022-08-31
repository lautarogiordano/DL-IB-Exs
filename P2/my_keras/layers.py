import numpy as np
import activations

class BaseLayer():
    def __init__():
        return None
    
    def __call__():
        return None



class Input(BaseLayer):
    def __init__(self, n_features, n_out, scale=1e-2):
        #n_features = X.shape[1]+1 (el +1 lo agregamos en model.clean(X)), m = X.shape[0]
        self.W = np.random.uniform(-scale, scale, size=(n_features, n_out))

    def __call__(self, X):
        return np.dot(self.W.T, X)    



class Layer(BaseLayer):
    def __init__(self, n_neurons, n_out, activation=activations.RELU()):
        self.n_neurons = n_neurons
        self.activation = activation
        
    
    def __call__(self, yleft):
        #S es la salida de la layer, puede ser la ultima
        self.S = self.activation(yleft)
        return self.S

    def getWeights(self):
        return None
    
    def updateWeights(self, grad):
        pass

    def reg(self):
        pass



class Dense(Layer):
    def __init__(self, n_neurons, n_out, activation=activations.RELU(), scale=1e-2):
        super().__init__(n_neurons, n_out, activation)
        self.W = np.random.uniform(-scale, scale, size=(n_neurons+1, n_out))

    def dot(self, X):
        return np.dot(self.W.T, X)

    def __call__(self, X):
        #y es hacer Wt.X antes de aplicarle la funcion de activacion.
        #La necesito para calcular los gradientes
        self.y = self.dot(self, X)
        self.S = self.activation(self.y)
        return self.S

    def getWeights(self):
        return self.W
    
    def updateWeights(self, grad):
        #Pensar que hacer acá
        #
        #
        grad = grad * self.activation.prime(self.y)



class Concat(Layer):
    def __init__():
        return None

    def __call__(Layer1, Layer2):
        pass
