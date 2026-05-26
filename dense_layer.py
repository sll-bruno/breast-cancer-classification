import numpy as np

class DenseLayer:
    def __init__(self, n_l, n_l_minus, activation_function=None):
        self.weights = np.random.randn(n_l, n_l_minus) * 0.1 #multiplica por 0.01 para que sejam valores menores
        self.biases = np.random.randn(n_l, 1) * 0.1

        self.d_weights = None
        self.d_biases = None
        self.input = None   
        self.Z = None

        self.activation_function = activation_function

    def forward(self, input):
        self.input = input

        # Z[l] = W[l] * A[l - 1] + b[l]
        self.Z = np.dot(self.weights, input) + self.biases
        
        if self.activation_function is not None:
            A = self.activation_function.forward(self.Z)
            return A
        return self.Z

    def backward(self, dA):
        m = self.input.shape[1]

        if self.activation_function is not None:
            dZ = self.activation_function.backward(dA)
        else:
            dZ = dA

        self.d_weights = 1/m * np.matmul(dZ, self.input.T)
        self.d_biases = 1/m * np.sum(dZ, axis=1, keepdims=True)

        d_input = np.matmul(self.weights.T, dZ)

        return d_input
        

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases  -= learning_rate * self.d_biases
        pass


