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
    def __init__(self, n_features):
        #n_features = X.shape[1]+1 (el +1 lo agregamos en model.clean(X)), m = X.shape[0]
        self.n_neurons = n_features

    
    def forward(self, X):
        return X
    
    def __call__(self, X):
        return X


class WLayer(BaseLayer): #Interfaz para neurona con activacion
    def __init__(self, n_in, n_neurons, activation, scale=1e-2):
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.activation = activation
        self.W = np.random.uniform(-scale, scale, size=(self.n_in + 1, self.n_neurons))
            
    def __call__(self):
        return None

    def forward(self):
        return None
    
class Dense(WLayer):
    def __init__(self, n_in, n_neurons, activation, reg, scale=1e-2):
        super().__init__(n_in, n_neurons, activation, scale)
        self.reg = reg

    def __call__(self):
        return self.S

    def getReg(self):
        return self.reg(self.W)

    def forward(self, Sprev):
        Sprevio = np.hstack((np.ones((Sprev.shape[0], 1)), Sprev))
        self.y = np.dot(Sprevio, self.W)
        self.S = self.activation(self.y)
        return self.S
    
    def updateWeights(self, grad, Sprev, updateRule):
        Sprevio = np.hstack((np.ones((Sprev.shape[0], 1)), Sprev))
        grad = grad * self.activation.prime(self.y)
        gradW = np.dot(Sprevio.T, grad)     
        grad = np.dot(grad, self.W.T)
        grad = grad[:, 1:]
        self.W = updateRule(self.W, gradW, self.reg)
        return grad

class ConcatInput(WLayer):
    def __init__(self, n_neurons, n1, n2):
        self.n_neurons = n_neurons
        self.n1 = n1
        self.n2 = n2
        self.n = self.n1 + self.n2
        

    def __call__(self):
        return self.S

    def forward(self, X, S):
        self.S = np.hstack((S, X[:, 1:]))
        return self.S

    def updateWeights(self, grad):
        #Solo saco la parte de reinyeccion del gradiente
        grad = grad[:,:self.n1]
        return grad