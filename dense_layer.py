import numpy as np

class DenseLayer:
    def __init__(self, n_l, n_l_minus):
        self.weights = np.random.randn(n_l, n_l_minus) * 0.01 #multiplica por 0.01 para que sejam valores menores
        self.biases = np.zeros((n_l, 1))
        
        self.d_weights = None
        self.d_biases = None
        self.input = None   

    def forward(self, input):
        self.input = input

        # Z[l] = W[l] * A[l - 1] + b[l]
        Z = np.dot(self.weights, input) + self.biases
        return Z

    def backward(self, dZ):
        m = self.input.shape[1]

        self.d_weights = 1/m * np.matmul(dZ, self.input.T)
        self.d_biases = 1/m * np.sum(dZ, axis=1, keepdims=True)

        d_input = np.matmul(self.weights.T, dZ)

        return d_input
        

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases  -= learning_rate * self.d_biases
        pass


