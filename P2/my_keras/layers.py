import numpy as np

class BaseLayer():
    def __init__(self):
        return None
    
    #call me devuelve cuanto vale la salida S de la capa
    def __call__(self):
        return None

    #forward calcula (y actualiza) el valor de S pasandole
    #el input de la capa anterior
    def forward(self):
        return None



class Input(BaseLayer):
    def __init__(self, n_features, reg):
        #n_features = X.shape[1]+1 (el +1 lo agregamos en model.clean(X)), m = X.shape[0]
        self.n_neurons = n_features
        self.reg = reg

    def setW(self, n_out, scale=1e-2):
        self.W = np.random.uniform(-scale, scale, size=(self.n_neurons, n_out))

    def forward(self, X):
        self.S = np.dot(X, self.W)
        return self.S
    
    def __call__(self):
        return self.S

    def updateWeights(self, Xb, grad, updateRule):
        gradW = np.dot(Xb.T, grad)     
        self.W = updateRule(self.W, gradW, self.reg)


class Layer(BaseLayer): #Interfaz para neurona con activacion
    def __init__(self, n_neurons, activation):
        self.n_neurons = n_neurons
        self.activation = activation
    
    def __call__(self):
        return None

    def forward(self):
        return None
    


class LastLayer(Layer): #esta es basicamente mi neurona de salida,
                            #tiene activacion pero no pesos
    def __init__(self, n_neurons, activation):
        super().__init__(n_neurons, activation)
        
    def __call__(self):
        return self.S
    
    def forward(self, Sprev):
        #S es la salida de la layer, puede ser la ultima
        #Sprev = Sprev[:,0] #Le saco la dimension trivial
        self.y = Sprev
        self.S = self.activation(Sprev)
        return self.S

    def getGrad(self, grad):
        grad = grad * self.activation.prime(self.y)
        return grad

    def reg(self):
        pass



class Dense(Layer):
    def __init__(self, n_neurons, activation, reg):
        super().__init__(n_neurons, activation)
        self.reg = reg

    def setW(self, n_out, scale=1e-2):
        self.W = np.random.uniform(-scale, scale, size=(self.n_neurons+1, n_out))

    def __call__(self):
        return self.S

    def getReg(self):
        return self.reg(self.W)

    def forward(self, Sprev):
        #y es la activacion de las neuronas por la salida de la capa previa.
        #Luego le agrego la columna de unos para matchear con W
        #La necesito para calcular los gradientes
        self.y = self.activation(Sprev)
        self.y = np.hstack((np.ones((self.y.shape[0], 1)), self.y))
        self.S = np.dot(self.y, self.W)
        return self.S
    
    def updateWeights(self, grad, updateRule):
        gradW = np.dot(self.y.T, grad)     
        grad = np.dot(grad, self.W.T)
        grad = grad * self.activation.prime(self.y)
        grad = grad[:, 1:]
        self.W = updateRule(self.W, gradW, self.reg)
        return grad
        



class Concat(Layer):
    def __init__(self):
        return None

    def __call__(self, Layer1, Layer2):
        pass
