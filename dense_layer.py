import numpy as np

class DenseLayer:
    def __init__(self, n_l, n_l_minus, activation_function=None, init_method="he"):
        
        self.weights, self.biases = self.initialize_parameters(n_l, n_l_minus, init_method)

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
    
    def forward_baseline(self, input_base):
        self.input_base = input_base
        self.Z_base = np.dot(self.weights, input_base) + self.biases
        
        if self.activation_function is not None:
            A_base = self.activation_function.forward_baseline(self.Z_base)
            return A_base
        return self.Z_base
    
    def backward_deepshap(self, incoming_shap):
        if self.activation_function is not None:
            dZ = self.activation_function.backward_deepshap(incoming_shap)
        else:
            dZ = incoming_shap

        d_input = np.matmul(self.weights.T, dZ)

        return d_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases  -= learning_rate * self.d_biases
        pass
    
    def initialize_parameters(self, n_l, n_l_minus, method):
        if method == "random":
            weights = np.random.randn(n_l, n_l_minus) * 0.5
        elif method == "xavier":
            weights = np.random.randn(n_l, n_l_minus) * np.sqrt(1/n_l_minus)
        elif method == "he":
            weights = np.random.randn(n_l, n_l_minus) * np.sqrt(2/n_l_minus) # N ~ (0, 2/n_l_minus) media é 0 e variancia é 2/n_l_minus, o que ajuda a evitar o problema do gradiente desaparecendo em redes profundas com ReLU. A ideia é que a variância das ativações seja mantida ao longo das camadas, evitando que os gradientes se tornem muito pequenos ou muito grandes durante a retropropagação.
        else:
            raise ValueError("Invalid initialization method")
        
        # Inicialização uniforme centrada em 0
        # bound = 1 / np.sqrt(n_l_minus)
        # biases = np.random.uniform(low=-bound, high=bound, size=(n_l, 1))
        biases = np.zeros((n_l, 1))
        return weights, biases

